import os
import json
import logging
import random
from pathlib import Path
from typing import Dict, Any, Tuple

from llm.base import GenerationPlan
from core.workflow_scorer import WorkflowScorer

logger = logging.getLogger(__name__)

class WorkflowManager:
    """
    Discovers, describes, and heuristically injects parameters into 
    user-provided ComfyUI workflow JSON files.
    """
    def __init__(self, data_dir: str):
        self.workflows_dir = Path(data_dir) / "workflows"
        self.workflows_dir.mkdir(parents=True, exist_ok=True)
        self.scorer = WorkflowScorer(data_dir)
        self.templates = {}
        self.refresh()

    def refresh(self):
        """Scan directory for .json files and matching .txt descriptions."""
        self.templates.clear()
        if not self.workflows_dir.exists():
            return
            
        for json_file in self.workflows_dir.glob("*.json"):
            name = json_file.stem
            
            # Load metadata/description
            description = f"User defined workflow template: {name}"
            txt_file = self.workflows_dir / f"{name}.txt"
            if txt_file.exists():
                try:
                    description = txt_file.read_text(encoding="utf-8").strip()
                except Exception as e:
                    logger.warning(f"Could not read {txt_file}: {e}")
            
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    workflow_data = json.load(f)
                    self.templates[name] = {
                        "name": name,
                        "description": description,
                        "path": str(json_file),
                        "data": workflow_data
                    }
                logger.info(f"Loaded workflow template: {name}")
            except Exception as e:
                logger.error(f"Failed to load workflow {json_file}: {e}")

    def get_available_templates_for_llm(self) -> str:
        """Return formatted string of available templates for LLM intent analysis."""
        if not self.templates:
            return ""
            
        lines = ["\nAVAILABLE WORKFLOW TEMPLATES (You MUST choose one of these):"]
        for name, info in self.templates.items():
            score_summary = self.scorer.get_template_stats_string(name)
            lines.append(f"  - {name}: {info['description']} [Score: {score_summary}]")
        return "\n".join(lines)

    def has_templates(self) -> bool:
        return len(self.templates) > 0

    def build_from_template(self, plan: GenerationPlan, reference_image: str = None, pose_image: str = None) -> Tuple[Dict[str, Any], bool]:
        """
        Attempt to build the workflow using the selected template.
        Returns (workflow_json, success_bool).
        """
        if not plan.workflow_template or plan.workflow_template not in self.templates:
            return {}, False

        import copy
        template = self.templates[plan.workflow_template]
        workflow = copy.deepcopy(template["data"])
        
        logger.info(f"Building from user workflow template: {plan.workflow_template}")
        
        # Inject standard parameters heuristically
        self._inject_model(workflow, plan.checkpoint)
        self._inject_prompts(workflow, plan.enhanced_prompt, plan.negative_prompt)
        self._inject_seed_and_settings(workflow, plan)
        self._inject_dimensions(workflow, plan.width, plan.height)
        self._inject_images(workflow, reference_image, pose_image)
        
        return workflow, True

    def _inject_model(self, workflow: dict, checkpoint: str):
        if not checkpoint: return
        # Finding CheckpointLoaderSimple or UNETLoader
        for node_id, node in workflow.items():
            if not isinstance(node, dict): continue
            c_type = node.get("class_type", "")
            if c_type in ("CheckpointLoaderSimple", "UNETLoader"):
                inputs = node.get("inputs", {})
                if "ckpt_name" in inputs:
                    inputs["ckpt_name"] = checkpoint
                elif "unet_name" in inputs:
                    inputs["unet_name"] = checkpoint

    def _inject_prompts(self, workflow: dict, pos_prompt: str, neg_prompt: str):
        # Find KSampler to trace back which CLIPTextEncode is pos vs neg
        sampler_node = None
        for node_id, node in workflow.items():
            if not isinstance(node, dict): continue
            if node.get("class_type") in ("KSampler", "KSamplerAdvanced", "SamplerCustom"):
                sampler_node = node
                break
                
        if not sampler_node:
            # Fallback: strict guessing
            clip_nodes = []
            for node_id, node in workflow.items():
                if isinstance(node, dict) and node.get("class_type") == "CLIPTextEncode":
                    clip_nodes.append(node)
                    
            if len(clip_nodes) >= 1 and pos_prompt:
                if "text" in clip_nodes[0].get("inputs", {}): 
                    clip_nodes[0]["inputs"]["text"] = pos_prompt
            if len(clip_nodes) >= 2 and neg_prompt:
                if "text" in clip_nodes[1].get("inputs", {}):
                    clip_nodes[1]["inputs"]["text"] = neg_prompt
            return

        inputs = sampler_node.get("inputs", {})
        pos_link = inputs.get("positive")
        neg_link = inputs.get("negative")
        
        if isinstance(pos_link, list) and len(pos_link) > 0:
            pos_node_id = str(pos_link[0])
            if pos_node_id in workflow and pos_prompt:
                if "text" in workflow[pos_node_id].get("inputs", {}):
                    workflow[pos_node_id]["inputs"]["text"] = pos_prompt

        if isinstance(neg_link, list) and len(neg_link) > 0:
            neg_node_id = str(neg_link[0])
            if neg_node_id in workflow and neg_prompt:
                if "text" in workflow[neg_node_id].get("inputs", {}):
                    workflow[neg_node_id]["inputs"]["text"] = neg_prompt

    def _inject_seed_and_settings(self, workflow: dict, plan: GenerationPlan):
        seed = getattr(plan, 'seed', -1)
        if seed == -1:
            seed = random.randint(1, 0xffffffff)
            
        for node_id, node in workflow.items():
            if not isinstance(node, dict): continue
            c_type = node.get("class_type", "")
            inputs = node.get("inputs", {})
            if c_type in ("KSampler", "KSamplerAdvanced", "SamplerCustom"):
                if "seed" in inputs: inputs["seed"] = seed
                if "noise_seed" in inputs: inputs["noise_seed"] = seed

    def _inject_dimensions(self, workflow: dict, width: int, height: int):
        if not width or not height: return
        for node_id, node in workflow.items():
            if not isinstance(node, dict): continue
            c_type = node.get("class_type", "")
            if c_type == "EmptyLatentImage":
                inputs = node.get("inputs", {})
                if "width" in inputs: inputs["width"] = width
                if "height" in inputs: inputs["height"] = height

    def _inject_images(self, workflow: dict, reference_image: str, pose_image: str):
        """Heuristically assign visual inputs to LoadImage nodes."""
        if not reference_image and not pose_image:
            return
            
        for node_id, node in workflow.items():
            if not isinstance(node, dict): continue
            
            c_type = node.get("class_type", "")
            if c_type == "LoadImage":
                # Look at ComfyUI native _meta.title if available
                title = str(node.get("_meta", {}).get("title", "")).lower()
                inputs = node.get("inputs", {})
                
                if ("pose" in title or "depth" in title or "canny" in title) and pose_image:
                    inputs["image"] = pose_image
                    inputs["_injected"] = True
                elif ("ref" in title or "style" in title or "ip" in title) and reference_image:
                    inputs["image"] = reference_image
                    inputs["_injected"] = True
                else:
                    # Fallbacks based on presence of only one image type
                    if reference_image and not inputs.get("_injected") and not pose_image:
                        inputs["image"] = reference_image
                        inputs["_injected"] = True
                    elif pose_image and not inputs.get("_injected") and not reference_image:
                        inputs["image"] = pose_image
                        inputs["_injected"] = True
                    elif reference_image and not inputs.get("_injected"):
                        # If both provided but node is ambiguous, inject reference first
                        inputs["image"] = reference_image
                        inputs["_injected"] = True

