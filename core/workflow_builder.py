"""
Dynamic Workflow Builder — constructs ComfyUI JSON from GenerationPlan.

Supports multiple model architectures:
  - SDXL: CheckpointLoaderSimple (standard)
  - Flux: UNETLoader + DualCLIPLoader + VAELoader
  - Hunyuan: UNETLoader + separate CLIP loading

Each feature (ControlNet, IP-Adapter, LoRA) is a modular add-on.
"""
import copy
import json
import logging
import time
from pathlib import Path
from typing import Optional

from llm.base import GenerationPlan

logger = logging.getLogger(__name__)

# Path to the base workflow template
BASE_WORKFLOW_PATH = Path(__file__).parent.parent / "data" / "workflow_base.json"


class WorkflowBuilder:
    """Builds ComfyUI API-format JSON workflows from a GenerationPlan."""

    def __init__(self, config: dict, registry=None):
        self.config = config
        self.defaults = config.get("defaults", {})
        self.registry = registry  # ModelRegistry for architecture-aware building

    def build(
        self,
        plan: GenerationPlan,
        reference_image: Optional[str] = None,
        pose_image: Optional[str] = None,
    ) -> dict:
        """
        Build a complete ComfyUI workflow from a GenerationPlan.
        
        Automatically detects the model architecture and builds
        the correct workflow pipeline (SDXL, Flux, Hunyuan, etc.)
        """
        logger.info(f"Building workflow: action={plan.action}, arch={plan.model_arch}, style={plan.style_category}")
        
        if plan.action == "upscale":
            return self._build_upscale_workflow(plan, reference_image)
        
        # Build the right base workflow for the architecture
        arch = plan.model_arch
        
        if arch == "flux":
            workflow = self._create_flux_workflow(plan)
        elif arch == "hunyuan":
            workflow = self._create_hunyuan_workflow(plan)
        else:
            workflow = self._create_sdxl_workflow(plan)
        
        # Add ControlNet if needed (only for SDXL currently)
        if plan.use_controlnet and pose_image and arch == "sdxl":
            workflow = self._add_controlnet(workflow, plan, pose_image)
        
        # Add IP-Adapter if needed (only for SDXL currently)
        if plan.use_ipadapter and reference_image and arch == "sdxl":
            workflow = self._add_ipadapter(workflow, plan, reference_image)
        
        # Add LoRAs if specified
        if plan.loras:
            workflow = self._add_loras(workflow, plan)
        
        logger.info(f"Workflow built with {len(workflow)} nodes: {list(workflow.keys())}")
        return workflow

    # ------------------------------------------------------------------
    # SDXL Workflow (CheckpointLoaderSimple — standard)
    # ------------------------------------------------------------------
    def _create_sdxl_workflow(self, plan: GenerationPlan) -> dict:
        """Create standard SDXL text-to-image workflow."""
        seed = plan.seed if plan.seed != -1 else int(time.time() * 1000) % 1000000000
        
        checkpoint = plan.checkpoint or self.defaults.get(
            "ckpt", "Juggernaut-XL_v9_RunDiffusion.safetensors"
        )
        
        workflow = {
            "base_model": {
                "inputs": {"ckpt_name": checkpoint},
                "class_type": "CheckpointLoaderSimple",
            },
            "positive_prompt": {
                "inputs": {
                    "text": plan.enhanced_prompt or "masterpiece, best quality",
                    "clip": ["base_model", 1],
                },
                "class_type": "CLIPTextEncode",
            },
            "negative_prompt": {
                "inputs": {
                    "text": plan.negative_prompt,
                    "clip": ["base_model", 1],
                },
                "class_type": "CLIPTextEncode",
            },
            "empty_latent": {
                "inputs": {
                    "width": plan.width,
                    "height": plan.height,
                    "batch_size": 1,
                },
                "class_type": "EmptyLatentImage",
            },
            "sampler": {
                "inputs": {
                    "seed": seed,
                    "steps": plan.steps,
                    "cfg": plan.cfg,
                    "sampler_name": plan.sampler,
                    "scheduler": plan.scheduler,
                    "denoise": plan.denoise,
                    "model": ["base_model", 0],
                    "positive": ["positive_prompt", 0],
                    "negative": ["negative_prompt", 0],
                    "latent_image": ["empty_latent", 0],
                },
                "class_type": "KSampler",
            },
            "vae_decode": {
                "inputs": {
                    "samples": ["sampler", 0],
                    "vae": ["base_model", 2],
                },
                "class_type": "VAEDecode",
            },
            "save_image": {
                "inputs": {
                    "filename_prefix": "AI_Director",
                    "images": ["vae_decode", 0],
                },
                "class_type": "SaveImage",
            },
        }
        
        return workflow

    # ------------------------------------------------------------------
    # Flux Workflow (UNETLoader + DualCLIPLoader + VAELoader)
    # ------------------------------------------------------------------
    def _create_flux_workflow(self, plan: GenerationPlan) -> dict:
        """
        Create Flux-architecture workflow.
        
        Flux uses separate loaders:
          - UNETLoader for the diffusion model
          - DualCLIPLoader for two CLIP models (t5xxl + clip_l)
          - VAELoader for the VAE
        
        Flux also uses FluxGuidance node for CFG-like guidance.
        """
        seed = plan.seed if plan.seed != -1 else int(time.time() * 1000) % 1000000000
        
        unet_name = plan.checkpoint
        if not unet_name:
            unet_name = "flux2_dev_fp8mixed"  # Default Flux model
        
        # Detect CLIP and VAE from registry or use defaults
        clip_name1 = "t5xxl_fp8_e4m3fn.safetensors"
        clip_name2 = "clip_l.safetensors"
        vae_name = "ae.safetensors"
        
        if self.registry:
            clips = self.registry.get_clip_models_for_arch("flux")
            if clips and len(clips) >= 2:
                clip_name1 = clips[0]
                clip_name2 = clips[1]
            vae = self.registry.get_vae_for_arch("flux")
            if vae:
                vae_name = vae
        
        workflow = {
            "unet_loader": {
                "inputs": {
                    "unet_name": unet_name,
                    "weight_dtype": "default",
                },
                "class_type": "UNETLoader",
            },
            "clip_loader": {
                "inputs": {
                    "clip_name1": clip_name1,
                    "clip_name2": clip_name2,
                    "type": "flux",
                },
                "class_type": "DualCLIPLoader",
            },
            "vae_loader": {
                "inputs": {
                    "vae_name": vae_name,
                },
                "class_type": "VAELoader",
            },
            "positive_prompt": {
                "inputs": {
                    "text": plan.enhanced_prompt or "masterpiece, best quality",
                    "clip": ["clip_loader", 0],
                },
                "class_type": "CLIPTextEncode",
            },
            "negative_prompt": {
                "inputs": {
                    "text": plan.negative_prompt,
                    "clip": ["clip_loader", 0],
                },
                "class_type": "CLIPTextEncode",
            },
            "flux_guidance": {
                "inputs": {
                    "guidance": max(plan.cfg, 3.5),  # Flux guidance (different from CFG)
                    "conditioning": ["positive_prompt", 0],
                },
                "class_type": "FluxGuidance",
            },
            "empty_latent": {
                "inputs": {
                    "width": plan.width,
                    "height": plan.height,
                    "batch_size": 1,
                },
                "class_type": "EmptySD3LatentImage",
            },
            "sampler": {
                "inputs": {
                    "seed": seed,
                    "steps": plan.steps,
                    "cfg": 1.0,  # Flux uses guidance node instead of CFG
                    "sampler_name": plan.sampler,
                    "scheduler": plan.scheduler,
                    "denoise": plan.denoise,
                    "model": ["unet_loader", 0],
                    "positive": ["flux_guidance", 0],
                    "negative": ["negative_prompt", 0],
                    "latent_image": ["empty_latent", 0],
                },
                "class_type": "KSampler",
            },
            "vae_decode": {
                "inputs": {
                    "samples": ["sampler", 0],
                    "vae": ["vae_loader", 0],
                },
                "class_type": "VAEDecode",
            },
            "save_image": {
                "inputs": {
                    "filename_prefix": "AI_Director_Flux",
                    "images": ["vae_decode", 0],
                },
                "class_type": "SaveImage",
            },
        }
        
        return workflow

    # ------------------------------------------------------------------
    # Hunyuan Workflow (UNETLoader + separate CLIP/VAE)
    # ------------------------------------------------------------------
    def _create_hunyuan_workflow(self, plan: GenerationPlan) -> dict:
        """
        Create Hunyuan-architecture workflow.
        
        Similar to Flux but may use different clip loading.
        Falls back to a UNET-based approach.
        """
        seed = plan.seed if plan.seed != -1 else int(time.time() * 1000) % 1000000000
        
        unet_name = plan.checkpoint
        if not unet_name:
            unet_name = "hunyuan_3d_v2.1.safetensors"
        
        # Hunyuan can also use CheckpointLoaderSimple in some cases
        # For UNET-only models, we use UNETLoader
        if self.registry and self.registry.is_diffusion_model(unet_name):
            # UNET-only path
            workflow = {
                "unet_loader": {
                    "inputs": {
                        "unet_name": unet_name,
                        "weight_dtype": "default",
                    },
                    "class_type": "UNETLoader",
                },
                "clip_loader": {
                    "inputs": {
                        "clip_name1": "t5xxl_fp8_e4m3fn.safetensors",
                        "clip_name2": "clip_l.safetensors",
                        "type": "hunyuan_video",
                    },
                    "class_type": "DualCLIPLoader",
                },
                "vae_loader": {
                    "inputs": {"vae_name": "ae.safetensors"},
                    "class_type": "VAELoader",
                },
                "positive_prompt": {
                    "inputs": {
                        "text": plan.enhanced_prompt or "masterpiece, best quality",
                        "clip": ["clip_loader", 0],
                    },
                    "class_type": "CLIPTextEncode",
                },
                "negative_prompt": {
                    "inputs": {
                        "text": plan.negative_prompt,
                        "clip": ["clip_loader", 0],
                    },
                    "class_type": "CLIPTextEncode",
                },
                "empty_latent": {
                    "inputs": {
                        "width": plan.width,
                        "height": plan.height,
                        "batch_size": 1,
                    },
                    "class_type": "EmptyLatentImage",
                },
                "sampler": {
                    "inputs": {
                        "seed": seed,
                        "steps": plan.steps,
                        "cfg": plan.cfg,
                        "sampler_name": plan.sampler,
                        "scheduler": plan.scheduler,
                        "denoise": plan.denoise,
                        "model": ["unet_loader", 0],
                        "positive": ["positive_prompt", 0],
                        "negative": ["negative_prompt", 0],
                        "latent_image": ["empty_latent", 0],
                    },
                    "class_type": "KSampler",
                },
                "vae_decode": {
                    "inputs": {
                        "samples": ["sampler", 0],
                        "vae": ["vae_loader", 0],
                    },
                    "class_type": "VAEDecode",
                },
                "save_image": {
                    "inputs": {
                        "filename_prefix": "AI_Director_Hunyuan",
                        "images": ["vae_decode", 0],
                    },
                    "class_type": "SaveImage",
                },
            }
        else:
            # Fallback to CheckpointLoaderSimple
            workflow = self._create_sdxl_workflow(plan)
            workflow["save_image"]["inputs"]["filename_prefix"] = "AI_Director_Hunyuan"
        
        return workflow

    # ------------------------------------------------------------------
    # Modular Add-Ons (ControlNet, IP-Adapter, LoRA)
    # ------------------------------------------------------------------
    def _add_controlnet(
        self, workflow: dict, plan: GenerationPlan, pose_image: str
    ) -> dict:
        """Add ControlNet nodes to the workflow."""
        cn_model = self.defaults.get(
            "controlnet_model", "thibaud_xl_openpose_256lora.safetensors"
        )
        
        # Determine preprocessor based on type
        preprocessor_class = "DWPreprocessor"
        preprocessor_inputs = {
            "detect_hand": "enable",
            "detect_body": "enable",
            "detect_face": "enable",
            "resolution": plan.width,
            "image": ["load_pose", 0],
        }
        
        if plan.controlnet_type == "depth":
            preprocessor_class = "MiDaS-DepthMapPreprocessor"
            preprocessor_inputs = {
                "a": 6.283185307179586,
                "bg_threshold": 0.1,
                "resolution": plan.width,
                "image": ["load_pose", 0],
            }
        elif plan.controlnet_type == "canny":
            preprocessor_class = "CannyEdgePreprocessor"
            preprocessor_inputs = {
                "low_threshold": 100,
                "high_threshold": 200,
                "resolution": plan.width,
                "image": ["load_pose", 0],
            }
        elif plan.controlnet_type == "lineart":
            preprocessor_class = "LineArtPreprocessor"
            preprocessor_inputs = {
                "coarse": "disable",
                "resolution": plan.width,
                "image": ["load_pose", 0],
            }
        
        workflow["load_pose"] = {
            "inputs": {"image": pose_image, "upload": "image"},
            "class_type": "LoadImage",
        }
        
        workflow["preprocessor"] = {
            "inputs": preprocessor_inputs,
            "class_type": preprocessor_class,
        }
        
        workflow["controlnet_loader"] = {
            "inputs": {"control_net_name": cn_model},
            "class_type": "ControlNetLoader",
        }
        
        workflow["controlnet_apply"] = {
            "inputs": {
                "strength": plan.controlnet_strength,
                "start_percent": 0.0,
                "end_percent": 1.0,
                "positive": ["positive_prompt", 0],
                "negative": ["negative_prompt", 0],
                "control_net": ["controlnet_loader", 0],
                "image": ["preprocessor", 0],
            },
            "class_type": "ControlNetApplyAdvanced",
        }
        
        # Rewire sampler to use ControlNet conditioning
        workflow["sampler"]["inputs"]["positive"] = ["controlnet_apply", 0]
        workflow["sampler"]["inputs"]["negative"] = ["controlnet_apply", 1]
        
        logger.info(f"Added ControlNet: {plan.controlnet_type} (strength: {plan.controlnet_strength})")
        return workflow

    def _add_ipadapter(
        self, workflow: dict, plan: GenerationPlan, reference_image: str
    ) -> dict:
        """Add IP-Adapter nodes for style reference."""
        ipa_model = plan.ipadapter_model or "ip-adapter-plus_sdxl_vit-h.safetensors"
        clip_vision = "CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors"
        
        workflow["load_style"] = {
            "inputs": {"image": reference_image, "upload": "image"},
            "class_type": "LoadImage",
        }
        
        workflow["ipadapter_model"] = {
            "inputs": {"ipadapter_file": ipa_model},
            "class_type": "IPAdapterModelLoader",
        }
        
        workflow["clip_vision"] = {
            "inputs": {"clip_name": clip_vision},
            "class_type": "CLIPVisionLoader",
        }
        
        # Get the current model source for the sampler
        current_model = workflow["sampler"]["inputs"]["model"]
        
        workflow["ipadapter_apply"] = {
            "inputs": {
                "weight": plan.ipadapter_weight,
                "weight_type": "linear",
                "combine_embeds": "concat",
                "start_at": 0.0,
                "end_at": 1.0,
                "embeds_scaling": "V only",
                "model": current_model,
                "ipadapter": ["ipadapter_model", 0],
                "image": ["load_style", 0],
                "clip_vision": ["clip_vision", 0],
            },
            "class_type": "IPAdapterAdvanced",
        }
        
        # Rewire sampler model through IP-Adapter
        workflow["sampler"]["inputs"]["model"] = ["ipadapter_apply", 0]
        
        logger.info(f"Added IP-Adapter: {ipa_model} (weight: {plan.ipadapter_weight})")
        return workflow

    def _add_loras(self, workflow: dict, plan: GenerationPlan) -> dict:
        """Add LoRA loader nodes — supports chaining multiple LoRAs."""
        prev_model = workflow["sampler"]["inputs"]["model"]
        
        # Find the CLIP source — different for each architecture
        if "clip_loader" in workflow:
            prev_clip = ["clip_loader", 0]
        elif "base_model" in workflow:
            prev_clip = ["base_model", 1]
        else:
            prev_clip = ["unet_loader", 0]  # Shouldn't happen but fallback
        
        for i, lora_info in enumerate(plan.loras[:3]):  # Max 3 LoRAs
            lora_name = lora_info.get("name", "")
            lora_weight = lora_info.get("weight", 0.8)
            node_id = f"lora_{i}"
            
            workflow[node_id] = {
                "inputs": {
                    "lora_name": lora_name,
                    "strength_model": lora_weight,
                    "strength_clip": lora_weight,
                    "model": prev_model,
                    "clip": prev_clip,
                },
                "class_type": "LoraLoader",
            }
            
            prev_model = [node_id, 0]
            prev_clip = [node_id, 1]
        
        # Rewire sampler and prompts through the LoRA chain
        workflow["sampler"]["inputs"]["model"] = prev_model
        workflow["positive_prompt"]["inputs"]["clip"] = prev_clip
        workflow["negative_prompt"]["inputs"]["clip"] = prev_clip
        
        lora_names = [l.get("name", "?") for l in plan.loras[:3]]
        logger.info(f"Added {len(plan.loras[:3])} LoRAs: {lora_names}")
        return workflow

    def _build_upscale_workflow(
        self, plan: GenerationPlan, input_image: Optional[str] = None
    ) -> dict:
        """Build a simple upscale workflow using latent upscale + second pass."""
        checkpoint = plan.checkpoint or self.defaults.get(
            "ckpt", "Juggernaut-XL_v9_RunDiffusion.safetensors"
        )
        seed = plan.seed if plan.seed != -1 else int(time.time() * 1000) % 1000000000
        
        workflow = {
            "base_model": {
                "inputs": {"ckpt_name": checkpoint},
                "class_type": "CheckpointLoaderSimple",
            },
            "load_image": {
                "inputs": {"image": input_image or "input.png", "upload": "image"},
                "class_type": "LoadImage",
            },
            "positive_prompt": {
                "inputs": {
                    "text": plan.enhanced_prompt or "masterpiece, best quality, highres, 8k",
                    "clip": ["base_model", 1],
                },
                "class_type": "CLIPTextEncode",
            },
            "negative_prompt": {
                "inputs": {
                    "text": plan.negative_prompt,
                    "clip": ["base_model", 1],
                },
                "class_type": "CLIPTextEncode",
            },
            "vae_encode": {
                "inputs": {
                    "pixels": ["load_image", 0],
                    "vae": ["base_model", 2],
                },
                "class_type": "VAEEncode",
            },
            "upscale_latent": {
                "inputs": {
                    "upscale_method": "nearest-exact",
                    "width": plan.width * 2,
                    "height": plan.height * 2,
                    "crop": "disabled",
                    "samples": ["vae_encode", 0],
                },
                "class_type": "LatentUpscale",
            },
            "sampler": {
                "inputs": {
                    "seed": seed,
                    "steps": 20,
                    "cfg": plan.cfg,
                    "sampler_name": plan.sampler,
                    "scheduler": plan.scheduler,
                    "denoise": 0.4,
                    "model": ["base_model", 0],
                    "positive": ["positive_prompt", 0],
                    "negative": ["negative_prompt", 0],
                    "latent_image": ["upscale_latent", 0],
                },
                "class_type": "KSampler",
            },
            "vae_decode": {
                "inputs": {
                    "samples": ["sampler", 0],
                    "vae": ["base_model", 2],
                },
                "class_type": "VAEDecode",
            },
            "save_image": {
                "inputs": {
                    "filename_prefix": "AI_Director_Upscale",
                    "images": ["vae_decode", 0],
                },
                "class_type": "SaveImage",
            },
        }
        
        logger.info("Built upscale workflow")
        return workflow
