"""
Style Presets — structured configurations for art styles.
"""
import yaml
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Dict, Optional

from llm.base import GenerationPlan

logger = logging.getLogger(__name__)

@dataclass
class StylePreset:
    name: str = ""
    display_name: str = ""
    default_model: str = ""
    recommended_loras: List[Dict[str, float]] = field(default_factory=list)
    prompt_modifiers: str = ""
    negative_modifiers: str = ""
    default_cfg: float = 7.0
    default_steps: int = 30
    description: str = ""


# Fallback presets if YAML is missing
DEFAULT_PRESETS = [
    StylePreset(
        name="cinematic",
        display_name="🎬 Cinematic",
        default_model="Juggernaut-XL_v9_RunDiffusion.safetensors",
        prompt_modifiers="cinematic lighting, anamorph lens, 8k uhd, photorealistic, dramatic movie still, highly detailed",
        negative_modifiers="cartoon, illustration, 3d render, low quality, bad lighting",
        default_cfg=7.0,
        default_steps=35,
        description="Dramatic, movie-like quality with strong lighting"
    ),
    StylePreset(
        name="anime",
        display_name="🎌 Anime",
        default_model="",  # Auto-select best anime model
        prompt_modifiers="masterpiece, best quality, ultra-detailed, anime style, 2d illustration, vibrant colors",
        negative_modifiers="realistic, 3d, photo, worst quality, low quality",
        default_cfg=6.5,
        default_steps=25,
        description="High quality 2D anime style"
    ),
    StylePreset(
        name="photorealistic",
        display_name="📸 Photorealistic",
        default_model="Juggernaut-XL_v9_RunDiffusion.safetensors",
        prompt_modifiers="raw photo, 8k uhd, dslr, soft lighting, high quality, film grain, Fujifilm XT4",
        negative_modifiers="(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck",
        default_cfg=6.0,
        default_steps=30,
        description="Extremely realistic photography"
    )
]

class StylePresetManager:
    """Manages loaded style presets."""

    def __init__(self, data_dir=None):
        if not data_dir:
            self.data_dir = Path(__file__).parent / "data"
        else:
            self.data_dir = Path(data_dir)
            
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.presets: Dict[str, StylePreset] = {}
        self._load_presets()

    def _load_presets(self):
        """Load presets from YAML file."""
        preset_file = self.data_dir / "style_presets.yaml"
        
        if not preset_file.exists():
            # Create default file
            with open(preset_file, "w") as f:
                dump_data = [asdict(p) for p in DEFAULT_PRESETS]
                yaml.dump(dump_data, f, default_flow_style=False)
            
            for p in DEFAULT_PRESETS:
                self.presets[p.name] = p
            logger.info("Created default style presets.")
            return

        try:
            with open(preset_file, "r") as f:
                data = yaml.safe_load(f)
                
            if data and isinstance(data, list):
                for item in data:
                    preset = StylePreset(**item)
                    self.presets[preset.name] = preset
            logger.info(f"Loaded {len(self.presets)} style presets.")
        except Exception as e:
            logger.error(f"Failed to load style presets from YAML: {e}")
            for p in DEFAULT_PRESETS:
                self.presets[p.name] = p

    def get_preset(self, name: str) -> Optional[StylePreset]:
        """Get a preset by name."""
        return self.presets.get(name.lower().strip())

    def list_presets(self) -> List[StylePreset]:
        """List all available presets."""
        return list(self.presets.values())

    def apply_preset(self, plan: GenerationPlan, preset: StylePreset) -> GenerationPlan:
        """Apply a preset to a GenerationPlan."""
        # Only override if the plan doesn't have an explicit value
        if not plan.checkpoint and preset.default_model:
            plan.checkpoint = preset.default_model
            
        # Append modifiers to the prompt
        if preset.prompt_modifiers:
            plan.enhanced_prompt = f"{plan.enhanced_prompt}, {preset.prompt_modifiers}"
            
        if preset.negative_modifiers:
            # Check if common negative words exist
            if plan.negative_prompt and plan.negative_prompt != GenerationPlan.negative_prompt:
                plan.negative_prompt = f"{plan.negative_prompt}, {preset.negative_modifiers}"
            else:
                plan.negative_prompt = preset.negative_modifiers
                
        # Set settings
        if getattr(plan, "_using_defaults", True):
            plan.cfg = preset.default_cfg
            plan.steps = preset.default_steps
            
        return plan
