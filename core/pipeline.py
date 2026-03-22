"""
Generation Pipeline — orchestrates multi-stage generation.

Replaces the single-pass generation with a structured Pipeline:
Stage 1: Base generation
Stage 2: Quality analysis -> Refinement (img2img) if needed
Stage 3: Upscale/Detail enhancement if requested/needed
"""
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional

from llm.base import GenerationPlan
from core.quality_analyzer import QualityAnalyzer, QualityScore
from core.queue_manager import JobStatus

logger = logging.getLogger(__name__)


class PipelineStage(Enum):
    BASE = "base"
    REFINE = "refine"
    UPSCALE = "upscale"
    DETAIL = "detail"
    FACE_FIX = "face_fix"


@dataclass
class PipelineResult:
    """Result of a pipeline execution."""
    stage: PipelineStage
    image_bytes: bytes
    quality_score: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class GenerationPipeline:
    """
    Executes a multi-stage generation.
    Handles fallbacks, auto-correction, and stage transitions.
    """

    def __init__(self, engine, config: dict):
        self.engine = engine  # AIDirectorEngine reference
        self.config = config.get("pipeline", {})
        
        self.max_retries = self.config.get("max_retries", 2)
        self.quality_threshold = self.config.get("quality_threshold", 6.0)
        self.auto_upscale = self.config.get("auto_upscale", False)
        
        # Modules used by the pipeline
        self.qa = QualityAnalyzer(engine.llm)

    async def execute(self, job_id, plan: GenerationPlan, reference_image=None, pose_image=None, status_callback=None) -> List[bytes]:
        """
        Execute the full pipeline for a GenerationPlan.
        Returns the final image bytes.
        """
        results_history = []
        
        # --- Stage 1: Base Generation ---
        if status_callback:
            await status_callback(JobStatus.BUILDING)
            
        workflow = None
        if hasattr(self.engine, "workflow_manager"):
            workflow, success = self.engine.workflow_manager.build_from_template(plan, reference_image, pose_image)
            if not success:
                logger.warning(f"Failed to build from template {plan.workflow_template}.")
                # Smarter fallback: try to find a general purpose one or one matching the arch
                available = list(self.engine.workflow_manager.templates.keys())
                if available:
                    fallback = available[0]
                    # Prefer 'turbo' for speed or 'sdxl' for compatibility if first one looks like an edit template
                    for opt in available:
                        if "turbo" in opt.lower() or "sdxl" in opt.lower():
                            fallback = opt
                            break
                    
                    logger.warning(f"Using fallback template: {fallback}")
                    plan.workflow_template = fallback
                    workflow, success = self.engine.workflow_manager.build_from_template(plan, reference_image, pose_image)
                else:
                    raise RuntimeError("No workflow templates found in data/workflows/")

        if workflow is None or not success:
            raise RuntimeError(f"Workflow building failed for template: {plan.workflow_template}")
        
        if status_callback:
            await status_callback(JobStatus.GENERATING)
            
        output_images = await self._run_workflow(workflow)
        
        if not output_images:
            raise RuntimeError("Base generation returned no images")
            
        current_image = output_images[0]
        
        # If this is just an upscale or edit task from the user, we skip auto-correction loop
        if plan.action in ("upscale", "edit", "vary"):
            return output_images
            
        # --- Stage 2: Quality Check & Auto-Correction ---
        retry_count = 0
        while retry_count < self.max_retries:
            if status_callback:
                # Assuming queue_manager has been updated with QUALITY_CHECK
                # If not, use ANALYZING as fallback
                await status_callback(getattr(JobStatus, "QUALITY_CHECK", JobStatus.ANALYZING))
                
            score = await self.qa.analyze(current_image)
            logger.info(f"Quality Score: {score.overall} (Faces: {score.faces}, Artifacts: {score.artifacts})")
            
            # Record score for this workflow template
            if hasattr(self.engine, "workflow_manager") and hasattr(self.engine.workflow_manager, "scorer"):
                self.engine.workflow_manager.scorer.record_score(plan.workflow_template, score)
            
            # Save history
            plan.user_overrides.setdefault("pipeline_history", []).append({
                "stage": "base" if retry_count == 0 else "refine",
                "score": score.overall,
                "issues": score.issues
            })
            
            # Check if we need to refine
            if score.overall >= self.quality_threshold and score.artifacts <= 5.0 and not score.needs_face_fix():
                logger.info("Image meets quality standards. Proceeding.")
                break
                
            logger.info(f"Image below threshold. Refining (attempt {retry_count + 1}/{self.max_retries})...")
            
            # Prepare refinement plan
            refine_plan = self._create_refinement_plan(plan, score)
            
            if status_callback:
                await status_callback(getattr(JobStatus, "REFINING", JobStatus.GENERATING))
                
            # Upload current image as reference for img2img
            try:
                # Assuming engine.comfyui.upload_image_bytes exists and returns a filename
                remote_name = "refine_temp.png"
                if hasattr(self.engine.comfyui, "upload_image_bytes"):
                    remote_name = await self.engine.comfyui.upload_image_bytes(current_image, f"qa_refine_{job_id}.png")
                
                # Try to find a refinement template
                available_templates = list(self.engine.workflow_manager.templates.keys())
                refine_template = next((t for t in available_templates if "img2img" in t.lower() or "refine" in t.lower()), None)
                
                if refine_template:
                    refine_plan.workflow_template = refine_template
                    refine_plan.denoise = 0.45
                    workflow, success = self.engine.workflow_manager.build_from_template(refine_plan, remote_name)
                    if not success:
                        workflow = None
                else:
                    logger.warning("No img2img template found for refinement loop. Skipping auto-correction.")
                    workflow = None
                    
                if workflow:
                    output_images = await self._run_workflow(workflow)
                    if output_images:
                        current_image = output_images[0]
                
            except Exception as e:
                logger.warning(f"Refinement pipeline failed: {e}. Keeping previous image.")
                break
                
            retry_count += 1
            
        # Optional: Auto-Detail/Enhance stage
        if self.auto_upscale and current_image and getattr(score, "sharpness", 10.0) < 6.0:
            if status_callback:
                 await status_callback(getattr(JobStatus, "ENHANCING", JobStatus.UPSCALING))
            try:
                remote_name = "upscale_temp.png"
                if hasattr(self.engine.comfyui, "upload_image_bytes"):
                    remote_name = await self.engine.comfyui.upload_image_bytes(current_image, f"qa_detail_{job_id}.png")
                
                # Try to find an upscale template
                available_templates = list(self.engine.workflow_manager.templates.keys())
                upscale_template = next((t for t in available_templates if "upscale" in t.lower() or "detail" in t.lower()), None)
                
                if upscale_template:
                    plan.workflow_template = upscale_template
                    workflow, success = self.engine.workflow_manager.build_from_template(plan, remote_name)
                    if success and workflow:
                        output_images = await self._run_workflow(workflow)
                        if output_images:
                            current_image = output_images[0]
            except Exception as e:
                logger.warning(f"Enhancement pipeline failed: {e}. Keeping previous image.")
            
        return [current_image]

    def _create_refinement_plan(self, original_plan: GenerationPlan, score: QualityScore) -> GenerationPlan:
        """Create a modified plan specifically for refining flaws."""
        import copy
        refine_plan = copy.deepcopy(original_plan)
        
        # Adjust prompt based on issues detected
        issue_types = [i.lower() for i in score.issues]
        
        if any("face" in i for i in issue_types) or score.needs_face_fix():
            refine_plan.enhanced_prompt = "detailed beautiful face, perfect eyes, highres, " + (refine_plan.enhanced_prompt or "")
            refine_plan.negative_prompt = "bad face, deformed eyes, lazy eye, " + refine_plan.negative_prompt
            
        if score.hands is not None and score.hands < 6.0:
            refine_plan.negative_prompt += ", bad hands, extra fingers, missing fingers, deformed hands"
            
        if score.artifacts is not None and score.artifacts > 5.0:
            refine_plan.negative_prompt += ", artifacts, jpeg artifacts, blurry, noisy"
            
        # Change seed to try a new variation
        refine_plan.seed = -1
        
        return refine_plan

    async def _run_workflow(self, workflow: dict) -> List[bytes]:
        """Helper to run a workflow and extract images."""
        prompt_id = await self.engine.comfyui.queue_prompt(workflow)
        output_data = await self.engine.comfyui.wait_for_result(prompt_id)
        
        result_images = []
        for node_id, images in output_data.items():
            result_images.extend(images)
            
        return result_images
