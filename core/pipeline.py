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
from core.quality_analyzer import QualityAnalyzer
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
            
        workflow = self.engine.builder.build(plan, reference_image, pose_image)
        
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
                
                # Build refine workflow based on specific issues
                from core.workflow_builder import WorkflowBuilder
                if hasattr(self.engine.builder, "build_face_fix_workflow") and score.needs_face_fix():
                    workflow = self.engine.builder.build_face_fix_workflow(remote_name)
                    if status_callback:
                        await status_callback(getattr(JobStatus, "FACE_FIXING", JobStatus.REFINING))
                elif hasattr(self.engine.builder, "build_img2img_workflow"):
                    # General refinement for artifacts/composition
                    refine_plan.denoise = 0.45
                    workflow = self.engine.builder.build_img2img_workflow(refine_plan, remote_name)
                else:
                    # Fallback to standard build
                    refine_plan.action = "edit"
                    workflow = self.engine.builder.build(refine_plan, remote_name, pose_image)
                    
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
                
                if hasattr(self.engine.builder, "build_detail_workflow"):
                    workflow = self.engine.builder.build_detail_workflow(plan, remote_name)
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
