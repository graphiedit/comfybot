"""
Model Registry — auto-discovers and catalogs all available models in ComfyUI.

Queries the ComfyUI API to find installed checkpoints, LoRAs, ControlNets,
IP-Adapters, and diffusion models (Flux, Hunyuan, etc.).
Enriches with metadata from catalog YAML files.
"""
import json
import logging
from pathlib import Path
from typing import Optional

import aiohttp
import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model architecture definitions — maps model filenames to their architecture
# so the workflow builder knows which pipeline to construct.
# ---------------------------------------------------------------------------
MODEL_ARCHITECTURES = {
    # Flux models
    "flux": {
        "arch": "flux",
        "default_cfg": 1.0,
        "default_steps": 4,
        "default_sampler": "euler",
        "default_scheduler": "simple",
        "clip_type": "flux",
        "clip_models": ["t5xxl_fp8_e4m3fn.safetensors", "clip_l.safetensors"],
        "vae": "ae.safetensors",
        "styles": ["general", "artistic", "creative"],
    },
    # Hunyuan models
    "hunyuan": {
        "arch": "hunyuan",
        "default_cfg": 1.0,
        "default_steps": 30,
        "default_sampler": "dpmpp_2m",
        "default_scheduler": "normal",
        "clip_type": "hunyuan",
        "vae": "",
        "styles": ["general", "creative", "artistic"],
    },
    # Lightning / Turbo variants (fast SDXL)
    "lightning": {
        "arch": "sdxl",
        "default_cfg": 2.0,
        "default_steps": 4,
        "default_sampler": "dpmpp_sde",
        "default_scheduler": "karras",
        "styles": ["general", "fast"],
    },
    # Turbo variants
    "turbo": {
        "arch": "sdxl",
        "default_cfg": 2.0,
        "default_steps": 4,
        "default_sampler": "dpmpp_sde",
        "default_scheduler": "karras",
        "styles": ["general", "fast"],
    },
}


def detect_architecture(filename: str) -> dict:
    """
    Detect the model architecture from the filename.
    Returns architecture info dict or None if standard SDXL.
    """
    name_lower = filename.lower()
    
    # Check for specific arch patterns
    if "flux" in name_lower:
        info = MODEL_ARCHITECTURES["flux"].copy()
        # Detect Flux variants
        if "turbo" in name_lower:
            info["default_steps"] = 4
            info["default_cfg"] = 1.0
            info["styles"] = ["general", "fast", "turbo"]
        elif "dev" in name_lower:
            info["default_steps"] = 20
            info["default_cfg"] = 1.0
        elif "schnell" in name_lower:
            info["default_steps"] = 4
            info["default_cfg"] = 1.0
        return info
    
    if "hunyuan" in name_lower:
        return MODEL_ARCHITECTURES["hunyuan"].copy()
    
    if "lightning" in name_lower:
        return MODEL_ARCHITECTURES["lightning"].copy()
    
    if "z_image" in name_lower or "z-image" in name_lower:
        info = MODEL_ARCHITECTURES["flux"].copy()
        if "turbo" in name_lower:
            info["default_steps"] = 4
        return info
        
    if "turbo" in name_lower:
        return MODEL_ARCHITECTURES["turbo"].copy()
        
    # Check for Qwen image models (these are diffusion_models too)
    if "qwen" in name_lower and "image" in name_lower:
        info = MODEL_ARCHITECTURES["flux"].copy()
        info["styles"] = ["general", "editing"]
        return info
    
    if "z_image" in name_lower or "z-image" in name_lower:
        info = MODEL_ARCHITECTURES["flux"].copy()
        if "turbo" in name_lower:
            info["default_steps"] = 4
        return info
    
    return None  # Standard SDXL


# ---------------------------------------------------------------------------
# Default workflow profile — used when a model has no explicit profile.
# These define HOW to wire up the ComfyUI workflow for a given architecture.
# ---------------------------------------------------------------------------
DEFAULT_WORKFLOW_PROFILES = {
    "flux": {
        "clip_loader": "DualCLIPLoader",
        "clip_type": "flux",
        "clip_name1": "qwen_2.5_vl_7b_fp8_scaled.safetensors",
        "clip_name2": "qwen_3_4b.safetensors",
        "vae": "flux2-vae.safetensors",
        "latent_type": "EmptySD3LatentImage",
        "use_guidance_node": True,
        "weight_dtype": "default",
    },
    "hunyuan": {
        "clip_loader": "DualCLIPLoader",
        "clip_type": "hunyuan_video",
        "clip_name1": "qwen_2.5_vl_7b_fp8_scaled.safetensors",
        "clip_name2": "qwen_3_4b.safetensors",
        "vae": "ae.safetensors",
        "latent_type": "EmptyLatentImage",
        "use_guidance_node": False,
        "weight_dtype": "default",
    },
    "sdxl": {
        # SDXL uses CheckpointLoaderSimple — CLIP/VAE are baked in.
        "clip_loader": None,
        "vae": None,
        "latent_type": "EmptyLatentImage",
        "use_guidance_node": False,
    },
}

# Default catalog data embedded in code
DEFAULT_MODEL_CATALOG = {
    "checkpoints": [
        {
            "filename": "Juggernaut-XL_v9_RunDiffusion.safetensors",
            "styles": ["realistic", "cinematic", "portrait", "product", "landscape"],
            "quality": 9,
            "default_cfg": 7.0,
            "default_sampler": "dpmpp_2m_sde",
            "default_scheduler": "karras",
            "base_model": "sdxl",
            "arch": "sdxl",
        },
        {
            "filename": "DreamShaperXL_Lightning.safetensors",
            "styles": ["general", "artistic", "fantasy", "fast"],
            "quality": 8,
            "default_cfg": 2.0,
            "default_sampler": "dpmpp_sde",
            "default_scheduler": "karras",
            "default_steps": 4,
            "base_model": "sdxl",
            "arch": "sdxl",
        },
    ],
    "diffusion_models": [
        {
            "filename": "flux2_dev_fp8mixed.safetensors",
            "display_name": "Flux.1 Dev (fp8)",
            "styles": ["general", "artistic", "creative", "realistic"],
            "quality": 9,
            "arch": "flux",
            "default_cfg": 1.0,
            "default_steps": 20,
            "default_sampler": "euler",
            "default_scheduler": "simple",
            "workflow_profile": {
                "clip_loader": "DualCLIPLoader",
                "clip_type": "flux",
                "clip_name1": "qwen_2.5_vl_7b_fp8_scaled.safetensors",
                "clip_name2": "qwen_3_4b.safetensors",
                "vae": "flux2-vae.safetensors",
                "latent_type": "EmptySD3LatentImage",
                "use_guidance_node": True,
                "weight_dtype": "fp8_e4m3fn",
            },
        },
        {
            "filename": "flux-2-klein-9b-fp8.safetensors",
            "display_name": "Flux.2 Klein 9B (fp8)",
            "styles": ["general", "artistic", "creative"],
            "quality": 8,
            "arch": "flux",
            "default_cfg": 1.0,
            "default_steps": 20,
            "default_sampler": "euler",
            "default_scheduler": "simple",
            "workflow_profile": {
                "clip_loader": "DualCLIPLoader",
                "clip_type": "flux",
                "clip_name1": "qwen_2.5_vl_7b_fp8_scaled.safetensors",
                "clip_name2": "qwen_3_4b.safetensors",
                "vae": "flux2-vae.safetensors",
                "latent_type": "EmptySD3LatentImage",
                "use_guidance_node": True,
                "weight_dtype": "fp8_e4m3fn",
            },
        },
        {
            "filename": "z_image_turbo_bf16.safetensors",
            "display_name": "Z-Image Turbo",
            "styles": ["general", "fast", "turbo"],
            "quality": 7,
            "arch": "flux",
            "default_cfg": 1.0,
            "default_steps": 4,
            "default_sampler": "euler",
            "default_scheduler": "simple",
            "workflow_profile": {
                "clip_loader": "DualCLIPLoader",
                "clip_type": "flux",
                "clip_name1": "qwen_2.5_vl_7b_fp8_scaled.safetensors",
                "clip_name2": "qwen_3_4b.safetensors",
                "vae": "flux2-vae.safetensors",
                "latent_type": "EmptySD3LatentImage",
                "use_guidance_node": True,
                "weight_dtype": "fp8_e4m3fn",
            },
        },
        {
            "filename": "z_image_bf16.safetensors",
            "display_name": "Z-Image",
            "styles": ["general", "artistic"],
            "quality": 8,
            "arch": "flux",
            "default_cfg": 1.0,
            "default_steps": 20,
            "default_sampler": "euler",
            "default_scheduler": "simple",
            "workflow_profile": {
                "clip_loader": "DualCLIPLoader",
                "clip_type": "flux",
                "clip_name1": "qwen_2.5_vl_7b_fp8_scaled.safetensors",
                "clip_name2": "qwen_3_4b.safetensors",
                "vae": "flux2-vae.safetensors",
                "latent_type": "EmptySD3LatentImage",
                "use_guidance_node": True,
                "weight_dtype": "default",
            },
        },
        {
            "filename": "qwen_image_edit_2509_fp8_e4m3fn.safetensors",
            "display_name": "Qwen Image Edit",
            "styles": ["editing", "general"],
            "quality": 7,
            "arch": "flux",
            "default_cfg": 1.0,
            "default_steps": 20,
            "default_sampler": "euler",
            "default_scheduler": "simple",
            "workflow_profile": {
                "clip_loader": "DualCLIPLoader",
                "clip_type": "flux",
                "clip_name1": "qwen_2.5_vl_7b_fp8_scaled.safetensors",
                "clip_name2": "qwen_3_4b.safetensors",
                "vae": "qwen_image_vae.safetensors",
                "latent_type": "EmptySD3LatentImage",
                "use_guidance_node": True,
                "weight_dtype": "default",
            },
        },
        {
            "filename": "hunyuan_3d_v2.1.safetensors",
            "display_name": "Hunyuan 3D v2.1",
            "styles": ["3d", "creative"],
            "quality": 7,
            "arch": "hunyuan",
            "default_cfg": 1.0,
            "default_steps": 30,
            "default_sampler": "dpmpp_2m",
            "default_scheduler": "normal",
            "workflow_profile": {
                "clip_loader": "DualCLIPLoader",
                "clip_type": "hunyuan_video",
                "clip_name1": "qwen_2.5_vl_7b_fp8_scaled.safetensors",
                "clip_name2": "qwen_3_4b.safetensors",
                "vae": "ae.safetensors",
                "latent_type": "EmptyLatentImage",
                "use_guidance_node": False,
                "weight_dtype": "default",
            },
        },
    ],
}

DEFAULT_LORA_CATALOG = {
    "loras": [
        {
            "id": "bhairavi_devi",
            "filename": "Bhairavi_Devi_Hindu_Art.safetensors",
            "keywords": ["hindu", "devi", "bhairavi", "indian art", "deity", "mythology"],
            "trigger_words": ["bhairavi devi"],
            "default_weight": 0.8,
            "compatible_models": ["flux", "sdxl"],
        },
        {
            "id": "flux_turbo",
            "filename": "Flux_2-Turbo-LoRA_comfyui.safetensors",
            "keywords": ["speed", "turbo", "fast"],
            "trigger_words": [],
            "default_weight": 1.0,
            "compatible_models": ["flux"],
        },
        {
            "id": "qwen_image_lightning",
            "filename": "Qwen-Image-2512-Lightning-4steps-V1.0-fp32.safetensors",
            "keywords": ["speed", "lightning", "fast", "qwen"],
            "trigger_words": [],
            "default_weight": 1.0,
            "compatible_models": ["flux"],
        },
        {
            "id": "qwen_edit_lightning",
            "filename": "Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors",
            "keywords": ["speed", "lightning", "fast", "edit", "qwen"],
            "trigger_words": [],
            "default_weight": 1.0,
            "compatible_models": ["flux"],
        },
        {
            "id": "indian_miniature",
            "filename": "indian_miniature_manuscript.safetensors",
            "keywords": ["indian", "miniature", "manuscript", "traditional", "painting"],
            "trigger_words": ["indian miniature"],
            "default_weight": 0.8,
            "compatible_models": ["flux", "sdxl"],
        },
        {
            "id": "pixel_art_z",
            "filename": "pixel_art_style_z_image_turbo.safetensors",
            "keywords": ["pixel art", "retro", "8bit", "game"],
            "trigger_words": ["pixel art style"],
            "default_weight": 0.8,
            "compatible_models": ["flux"],
        },
    ]
}


class ModelRegistry:
    """
    Central registry of all available models, LoRAs, ControlNets, and IP-Adapters.
    
    Combines:
    1. Live data from ComfyUI API (what's actually installed)
    2. Model architecture detection (Flux, Hunyuan, SDXL, etc.)
    3. Metadata from catalog YAML files (style tags, compatibility, defaults)
    """

    def __init__(self, config: dict):
        self.comfyui_url = config.get("comfyui", {}).get("url", "http://127.0.0.1:8188")
        self.server_addr = self.comfyui_url.replace("http://", "").replace("https://", "")
        
        # Data directory for catalog files
        self.data_dir = Path(__file__).parent / "data"
        self.data_dir.mkdir(exist_ok=True)
        
        # Cached data
        self._checkpoints = []        # Standard checkpoints (CheckpointLoaderSimple)
        self._diffusion_models = []   # UNET-only models (UNETLoader) — Flux, Hunyuan, etc.
        self._loras = []
        self._controlnets = []
        self._ipadapters = []
        self._clip_vision = []
        self._clip_models = []        # For DualCLIPLoader
        self._vae_models = []         # For VAELoader
        
        # Catalog metadata
        self._model_catalog = {}
        self._lora_catalog = {}
        
        self._loaded = False

    async def refresh(self):
        """Re-scan ComfyUI for available models and load catalogs."""
        logger.info("Refreshing model registry...")
        
        # Load catalog files
        self._load_catalogs()
        
        # Query ComfyUI API for installed models
        try:
            self._checkpoints = await self._query_comfyui_models("CheckpointLoaderSimple", "ckpt_name")
            self._loras = await self._query_comfyui_models("LoraLoader", "lora_name")
            self._controlnets = await self._query_comfyui_models("ControlNetLoader", "control_net_name")
            
            # Diffusion models (Flux, Hunyuan, etc.) — loaded via UNETLoader
            try:
                self._diffusion_models = await self._query_comfyui_models("UNETLoader", "unet_name")
            except Exception:
                logger.info("UNETLoader node not found — diffusion_models discovery skipped")
                self._diffusion_models = []
            
            # CLIP models for DualCLIPLoader
            try:
                self._clip_models = await self._query_comfyui_models("DualCLIPLoader", "clip_name1")
            except Exception:
                self._clip_models = []
            
            # VAE models for VAELoader
            try:
                self._vae_models = await self._query_comfyui_models("VAELoader", "vae_name")
            except Exception:
                self._vae_models = []
            
            # IP-Adapter models
            try:
                self._ipadapters = await self._query_comfyui_models("IPAdapterModelLoader", "ipadapter_file")
            except Exception:
                logger.info("IPAdapter nodes not installed — IP-Adapter features disabled")
                self._ipadapters = []
            
            # CLIP Vision models
            try:
                self._clip_vision = await self._query_comfyui_models("CLIPVisionLoader", "clip_name")
            except Exception:
                self._clip_vision = []
            
            self._loaded = True
            logger.info(
                f"Registry loaded: {len(self._checkpoints)} checkpoints, "
                f"{len(self._diffusion_models)} diffusion models, "
                f"{len(self._loras)} LoRAs, {len(self._controlnets)} ControlNets, "
                f"{len(self._ipadapters)} IP-Adapters"
            )
            
            if self._diffusion_models:
                logger.info(f"Diffusion models found: {self._diffusion_models}")
            
        except Exception as e:
            logger.error(f"Failed to query ComfyUI: {e}")
            logger.info("Using catalog data only (ComfyUI may not be running)")
            self._loaded = True

    def _load_catalogs(self):
        """Load model and LoRA catalogs from YAML files."""
        model_file = self.data_dir / "model_catalog.yaml"
        lora_file = self.data_dir / "lora_catalog.yaml"
        
        if model_file.exists():
            with open(model_file, "r") as f:
                self._model_catalog = yaml.safe_load(f) or {}
        else:
            self._model_catalog = DEFAULT_MODEL_CATALOG
            with open(model_file, "w") as f:
                yaml.dump(DEFAULT_MODEL_CATALOG, f, default_flow_style=False)
            logger.info(f"Created default model catalog at {model_file}")
        
        if lora_file.exists():
            with open(lora_file, "r") as f:
                self._lora_catalog = yaml.safe_load(f) or {}
        else:
            self._lora_catalog = DEFAULT_LORA_CATALOG
            with open(lora_file, "w") as f:
                yaml.dump(DEFAULT_LORA_CATALOG, f, default_flow_style=False)
            logger.info(f"Created default LoRA catalog at {lora_file}")

    async def _query_comfyui_models(self, node_type: str, input_name: str) -> list:
        """Query ComfyUI API for available models of a specific type."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.comfyui_url}/object_info/{node_type}",
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
                    models = data[node_type]["input"]["required"][input_name][0]
                    return models
        except Exception as e:
            logger.warning(f"Could not query {node_type}: {e}")
            return []

    def get_model_arch(self, model_name: str) -> str:
        """
        Determine the architecture of a model.
        Returns: 'sdxl', 'flux', 'hunyuan', etc.
        """
        # Check catalog first
        for ckpt in self._model_catalog.get("checkpoints", []):
            if ckpt["filename"] == model_name:
                return ckpt.get("arch", "sdxl")
        
        for dm in self._model_catalog.get("diffusion_models", []):
            if dm["filename"] == model_name:
                return dm.get("arch", "flux")
        
        # Check if it's in the diffusion_models list (UNET-only)
        if model_name in self._diffusion_models:
            detected = detect_architecture(model_name)
            if detected:
                return detected.get("arch", "flux")
            return "flux"  # Default for diffusion_models
        
        # Auto-detect from filename
        detected = detect_architecture(model_name)
        if detected:
            return detected.get("arch", "sdxl")
        
        return "sdxl"

    def get_model_defaults(self, model_name: str) -> dict:
        """
        Get default settings for a model (cfg, steps, sampler, scheduler).
        Works for both standard checkpoints and diffusion models.
        """
        # Check catalog first
        for ckpt in self._model_catalog.get("checkpoints", []):
            if ckpt["filename"] == model_name:
                return {
                    "cfg": ckpt.get("default_cfg", 7.0),
                    "steps": ckpt.get("default_steps", 30),
                    "sampler": ckpt.get("default_sampler", "dpmpp_2m_sde"),
                    "scheduler": ckpt.get("default_scheduler", "karras"),
                }
        
        for dm in self._model_catalog.get("diffusion_models", []):
            if dm["filename"] == model_name:
                return {
                    "cfg": dm.get("default_cfg", 1.0),
                    "steps": dm.get("default_steps", 20),
                    "sampler": dm.get("default_sampler", "euler"),
                    "scheduler": dm.get("default_scheduler", "simple"),
                }
        
        # Auto-detect from filename
        detected = detect_architecture(model_name)
        if detected:
            return {
                "cfg": detected.get("default_cfg", 7.0),
                "steps": detected.get("default_steps", 30),
                "sampler": detected.get("default_sampler", "dpmpp_2m_sde"),
                "scheduler": detected.get("default_scheduler", "karras"),
            }
        
        return {}

    def get_workflow_profile(self, model_name: str) -> dict:
        """
        Get the workflow profile for a model — defines how to wire up CLIP, VAE,
        latent nodes, and guidance for this specific model.

        Lookup order:
        1. Explicit 'workflow_profile' in the model catalog entry
        2. Architecture default from DEFAULT_WORKFLOW_PROFILES
        3. Hardcoded fallback
        """
        # Check catalog for explicit profiles
        for dm in self._model_catalog.get("diffusion_models", []):
            if dm["filename"] == model_name and "workflow_profile" in dm:
                return dm["workflow_profile"]

        for ckpt in self._model_catalog.get("checkpoints", []):
            if ckpt["filename"] == model_name and "workflow_profile" in ckpt:
                return ckpt["workflow_profile"]

        # Fall back to arch-level default
        arch = self.get_model_arch(model_name)
        if arch in DEFAULT_WORKFLOW_PROFILES:
            return DEFAULT_WORKFLOW_PROFILES[arch].copy()

        # Ultimate fallback
        return DEFAULT_WORKFLOW_PROFILES.get("sdxl", {}).copy()

    def is_diffusion_model(self, model_name: str) -> bool:
        """Check if a model is a diffusion model (UNET-only, needs separate CLIP/VAE)."""
        if model_name in self._diffusion_models:
            return True
        arch = self.get_model_arch(model_name)
        return arch in ("flux", "hunyuan")

    def get_clip_models_for_arch(self, arch: str, model_name: str = "") -> list:
        """Get the CLIP models appropriate for a model (profile-aware)."""
        # Try model-specific profile first
        if model_name:
            profile = self.get_workflow_profile(model_name)
            if profile.get("clip_name1") and profile.get("clip_name2"):
                return [profile["clip_name1"], profile["clip_name2"]]

        # Fall back to arch-level defaults
        arch_profile = DEFAULT_WORKFLOW_PROFILES.get(arch, {})
        if arch_profile.get("clip_name1") and arch_profile.get("clip_name2"):
            return [arch_profile["clip_name1"], arch_profile["clip_name2"]]

        return []  # SDXL uses the checkpoint's built-in CLIP

    def get_vae_for_arch(self, arch: str, model_name: str = "") -> str:
        """Get the VAE model appropriate for a model (profile-aware)."""
        # Try model-specific profile first
        if model_name:
            profile = self.get_workflow_profile(model_name)
            if profile.get("vae"):
                return profile["vae"]

        # Fall back to arch-level defaults
        arch_profile = DEFAULT_WORKFLOW_PROFILES.get(arch, {})
        if arch_profile.get("vae"):
            return arch_profile["vae"]

        return ""  # SDXL uses the checkpoint's built-in VAE


    def get_available_models_for_llm(self) -> dict:
        """
        Get a structured dict of all available models for the LLM to reason about.
        
        Enriches raw filenames with catalog metadata (styles, compatibility, etc.)
        Includes both standard checkpoints AND diffusion models.
        """
        # Enrich checkpoints with metadata
        checkpoints = []
        catalog_ckpts = {c["filename"]: c for c in self._model_catalog.get("checkpoints", [])}
        
        for ckpt in self._checkpoints:
            if ckpt in catalog_ckpts:
                entry = catalog_ckpts[ckpt].copy()
                entry.setdefault("arch", "sdxl")
                checkpoints.append(entry)
            else:
                detected = detect_architecture(ckpt)
                checkpoints.append({
                    "filename": ckpt,
                    "styles": detected["styles"] if detected else ["general"],
                    "arch": detected["arch"] if detected else ("sdxl" if "xl" in ckpt.lower() else "sd15"),
                    "base_model": detected["arch"] if detected else "sdxl",
                })
        
        # Enrich diffusion models with metadata
        diffusion_models = []
        catalog_dms = {d["filename"]: d for d in self._model_catalog.get("diffusion_models", [])}
        
        for dm in self._diffusion_models:
            if dm in catalog_dms:
                entry = catalog_dms[dm].copy()
                diffusion_models.append(entry)
            else:
                detected = detect_architecture(dm)
                diffusion_models.append({
                    "filename": dm,
                    "arch": detected["arch"] if detected else "flux",
                    "styles": detected["styles"] if detected else ["general"],
                    "note": "Uses UNETLoader (separate CLIP + VAE required)",
                })
        
        # If no data from API, use catalogues
        if not checkpoints:
            checkpoints = self._model_catalog.get("checkpoints", [])
        if not diffusion_models:
            diffusion_models = self._model_catalog.get("diffusion_models", [])
        
        # Enrich LoRAs with metadata
        loras = []
        catalog_loras = {l["filename"]: l for l in self._lora_catalog.get("loras", [])}
        
        for lora_file in self._loras:
            if lora_file in catalog_loras:
                entry = catalog_loras[lora_file].copy()
                # Categorize based on keywords
                keywords = [k.lower() for k in entry.get("keywords", [])]
                if any(k in keywords for k in ["style", "art", "painting", "illustration", "anime"]):
                    entry["category"] = "Style LoRA"
                else:
                    entry["category"] = "Subject/Character LoRA"
                loras.append(entry)
            else:
                loras.append({
                    "filename": lora_file,
                    "keywords": [],
                    "trigger_words": [],
                    "default_weight": 0.8,
                    "compatible_models": ["sdxl"],
                    "category": "Unknown",
                })
        
        if not loras:
            loras = self._lora_catalog.get("loras", [])
        
        # ControlNets
        controlnets = []
        for cn in self._controlnets:
            cn_type = "unknown"
            cn_lower = cn.lower()
            if "openpose" in cn_lower or "pose" in cn_lower:
                cn_type = "openpose"
            elif "depth" in cn_lower:
                cn_type = "depth"
            elif "canny" in cn_lower:
                cn_type = "canny"
            elif "lineart" in cn_lower or "line" in cn_lower:
                cn_type = "lineart"
            elif "scribble" in cn_lower:
                cn_type = "scribble"
            controlnets.append({"filename": cn, "type": cn_type})
        
        # IP-Adapters
        ipadapters = []
        for ipa in self._ipadapters:
            ipa_type = "standard"
            ipa_lower = ipa.lower()
            if "plus" in ipa_lower:
                ipa_type = "plus"
            elif "faceid" in ipa_lower or "face" in ipa_lower:
                ipa_type = "faceid"
            elif "light" in ipa_lower:
                ipa_type = "light"
            ipadapters.append({"filename": ipa, "type": ipa_type})
        
        return {
            "checkpoints": checkpoints,
            "diffusion_models": diffusion_models,
            "loras": loras,
            "controlnets": controlnets,
            "ipadapters": ipadapters,
            "clip_vision": self._clip_vision,
        }

    def get_best_checkpoint(self, style_category: str) -> Optional[str]:
        """Get the best checkpoint or diffusion model for a given style category."""
        catalog_ckpts = self._model_catalog.get("checkpoints", [])
        catalog_dms = self._model_catalog.get("diffusion_models", [])
        
        # Combine all models for scoring
        all_models = catalog_ckpts + catalog_dms
        
        # Score each model by style match
        best = None
        best_score = -1
        
        for model in all_models:
            # Check if model is actually available
            filename = model["filename"]
            is_available = not self._loaded or filename in self._checkpoints or filename in self._diffusion_models
            
            if is_available:
                if style_category in model.get("styles", []):
                    score = model.get("quality", 5)
                    if score > best_score:
                        best = filename
                        best_score = score
        
        if not best and self._diffusion_models:
            best = self._diffusion_models[0]
        elif not best and self._checkpoints:
            best = self._checkpoints[0]
        elif not best and all_models:
            best = all_models[0]["filename"]
        
        return best

    def score_model(self, model_entry: dict, style_category: str, prompt_keywords: list = None) -> float:
        """
        Score a model based on its fit for the request.
        Factors: Style match (40%), Quality (30%), Architecture/Tags (20%), Base (10%)
        """
        score = 0.0
        
        # 1. Style Match (up to 40 points)
        styles = model_entry.get("styles", ["general"])
        if style_category in styles:
            score += 40.0
        elif "general" in styles:
            score += 10.0
            
        # 2. Quality rating (up to 30 points)
        quality = model_entry.get("quality", 5)
        score += quality * 3.0  # 1-10 -> 3-30
        
        # 3. Keyword matching (up to 20 points)
        prompt_keywords = prompt_keywords or []
        tags = model_entry.get("tags", [])
        matches = sum(1 for kw in prompt_keywords if kw.lower() in [t.lower() for t in tags])
        score += min(20.0, matches * 5.0)
        
        # 4. Modern Architecture bonus (up to 10 points)
        arch = model_entry.get("arch", "sdxl")
        if arch in ("flux", "hunyuan"):
            score += 10.0
        elif arch == "sdxl":
            score += 5.0
            
        return score

    def get_best_models_ranked(self, style_category: str, prompt_keywords: list = None, top_n: int = 3) -> list:
        """Get the top N models ranked by score for a given style and prompt."""
        candidates = []
        
        # Add Checkpoints
        for ckpt in self._model_catalog.get("checkpoints", []):
            if not self._checkpoints or ckpt["filename"] in self._checkpoints:
                score = self.score_model(ckpt, style_category, prompt_keywords)
                candidates.append((ckpt["filename"], ckpt.get("arch", "sdxl"), score))
                
        # Add Diffusion Models
        for dm in self._model_catalog.get("diffusion_models", []):
            if not self._diffusion_models or dm["filename"] in self._diffusion_models:
                score = self.score_model(dm, style_category, prompt_keywords)
                candidates.append((dm["filename"], dm.get("arch", "flux"), score))
                
        # Sort by score descending
        candidates.sort(key=lambda x: x[2], reverse=True)
        
        return candidates[:top_n]

    def get_best_model(self, style_category: str) -> tuple:
        """
        Get the best model (checkpoint or diffusion model) for a style.
        Returns (filename, arch).
        """
        ranked = self.get_best_models_ranked(style_category, top_n=1)
        if ranked:
            return ranked[0][0], ranked[0][1]
        
        # Fallback
        if self._checkpoints:
            return self._checkpoints[0], "sdxl"
        
        return "", "sdxl"

    def get_compatible_loras(self, model_arch: str, prompt_keywords: list = None) -> list:
        """Find the best LoRAs for a given architecture and prompt."""
        prompt_keywords = prompt_keywords or []
        matches = []
        
        for lora in self._lora_catalog.get("loras", []):
            if not self._loras or lora["filename"] in self._loras:
                # Check architecture compatibility
                compat = lora.get("compatible_models", ["sdxl"])
                if model_arch in compat:
                    # Check if keywords match
                    lora_kws = lora.get("keywords", [])
                    has_match = any(kw.lower() in [k.lower() for k in prompt_keywords] for kw in lora_kws)
                    
                    if has_match:
                        matches.append(lora)
                        
        return matches

    def get_checkpoint_defaults(self, checkpoint: str) -> dict:
        """Get default settings for a specific checkpoint."""
        return self.get_model_defaults(checkpoint)

    def resolve_checkpoint(self, filename: str) -> Optional[str]:
        """
        Resolve a partial filename to its exact filename in the registry.
        E.g. "z_image_turbo_bf16" -> "z_image_turbo_bf16.safetensors"
        """
        if not self._checkpoints and not self._diffusion_models:
            return filename  # Trust it if API not loaded
            
        exact = filename in self._checkpoints or filename in self._diffusion_models
        if exact:
            return filename
            
        # Try substring match (case insensitive)
        query = filename.lower()
        if not query.endswith(".safetensors"):
            query += ".safetensors"
            
        for dm in self._diffusion_models:
            if dm.lower() == query or filename.lower() in dm.lower():
                return dm
                
        for ckpt in self._checkpoints:
            if ckpt.lower() == query or filename.lower() in ckpt.lower():
                return ckpt
                
        return None

    def validate_checkpoint(self, filename: str) -> bool:
        """Check if a checkpoint actually exists in ComfyUI."""
        return self.resolve_checkpoint(filename) is not None

    def validate_lora(self, filename: str) -> bool:
        """Check if a LoRA actually exists in ComfyUI."""
        if not self._loras:
            return True
        return filename in self._loras

    def get_lora_compatibility(self, lora_name: str) -> list:
        """Get the base models compatible with this LoRA."""
        # Check catalog
        for lora in self._lora_catalog.get("loras", []):
            if lora["filename"] == lora_name:
                return lora.get("compatible_models", ["sdxl"])
        
        # Default for unknown LoRAs
        return ["sdxl"]

    def has_ipadapter_support(self) -> bool:
        """Check if IP-Adapter nodes and models are available."""
        return len(self._ipadapters) > 0 and len(self._clip_vision) > 0

    def get_default_ipadapter(self) -> Optional[str]:
        """Get the default IP-Adapter model filename."""
        if self._ipadapters:
            for ipa in self._ipadapters:
                if "plus" in ipa.lower():
                    return ipa
            return self._ipadapters[0]
        return None

    def get_default_clip_vision(self) -> Optional[str]:
        """Get the default CLIP Vision model filename."""
        if self._clip_vision:
            for cv in self._clip_vision:
                if "vit-h" in cv.lower() or "ViT-H" in cv:
                    return cv
            return self._clip_vision[0]
        return None

    def add_checkpoint_to_catalog(self, filename: str, styles: list = None, **kwargs):
        """Add or update a checkpoint in the catalog."""
        catalog_ckpts = self._model_catalog.setdefault("checkpoints", [])
        
        for ckpt in catalog_ckpts:
            if ckpt["filename"] == filename:
                if styles:
                    ckpt["styles"] = styles
                ckpt.update(kwargs)
                self._save_model_catalog()
                return
        
        entry = {"filename": filename, "styles": styles or ["general"], **kwargs}
        catalog_ckpts.append(entry)
        self._save_model_catalog()

    def add_diffusion_model_to_catalog(self, filename: str, arch: str, styles: list = None, **kwargs):
        """Add or update a diffusion model in the catalog."""
        catalog_dms = self._model_catalog.setdefault("diffusion_models", [])
        
        for dm in catalog_dms:
            if dm["filename"] == filename:
                dm["arch"] = arch
                if styles:
                    dm["styles"] = styles
                dm.update(kwargs)
                self._save_model_catalog()
                return
        
        entry = {"filename": filename, "arch": arch, "styles": styles or ["general"], **kwargs}
        catalog_dms.append(entry)
        self._save_model_catalog()

    def add_lora_to_catalog(self, filename: str, keywords: list = None, **kwargs):
        """Add or update a LoRA in the catalog."""
        catalog_loras = self._lora_catalog.setdefault("loras", [])
        
        for lora in catalog_loras:
            if lora["filename"] == filename:
                if keywords:
                    lora["keywords"] = keywords
                lora.update(kwargs)
                self._save_lora_catalog()
                return
        
        entry = {
            "id": filename.replace(".safetensors", "").lower()[:30],
            "filename": filename,
            "keywords": keywords or [],
            "trigger_words": kwargs.get("trigger_words", []),
            "default_weight": kwargs.get("default_weight", 0.8),
            "compatible_models": kwargs.get("compatible_models", ["sdxl"]),
        }
        catalog_loras.append(entry)
        self._save_lora_catalog()

    def _save_model_catalog(self):
        """Persist model catalog to YAML."""
        model_file = self.data_dir / "model_catalog.yaml"
        with open(model_file, "w") as f:
            yaml.dump(self._model_catalog, f, default_flow_style=False)

    def _save_lora_catalog(self):
        """Persist LoRA catalog to YAML."""
        lora_file = self.data_dir / "lora_catalog.yaml"
        with open(lora_file, "w") as f:
            yaml.dump(self._lora_catalog, f, default_flow_style=False)
