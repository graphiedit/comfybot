"""
Abstract LLM Provider — swappable backend for AI intelligence.

Supports: Ollama (local), Gemini, OpenAI, or any custom provider.
To add a new provider, subclass LLMProvider and register it.
"""
import abc
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class StyleAnalysis:
    """Result of analyzing a reference image."""
    style: str = ""
    keywords: str = ""
    lora_search: list = field(default_factory=list)
    raw: str = ""


@dataclass
class GenerationPlan:
    """Structured output from intent analysis — describes HOW to generate."""
    action: str = "generate"                    # generate | upscale | edit | vary
    style_category: str = "realistic"           # realistic | anime | cinematic | fantasy | etc
    checkpoint: str = ""                        # which model to use (empty = auto)
    model_arch: str = "sdxl"                    # sdxl | flux | hunyuan (determines workflow type)
    loras: list = field(default_factory=list)    # [{"name": "...", "weight": 0.8}]
    use_controlnet: bool = False
    controlnet_type: str = ""                   # openpose | depth | canny | lineart
    controlnet_strength: float = 1.0
    use_ipadapter: bool = False
    ipadapter_model: str = ""                   # which ip-adapter model
    ipadapter_weight: float = 0.6
    enhanced_prompt: str = ""
    negative_prompt: str = "worst quality, low quality, bad anatomy, bad hands, text, error, blurry, jpeg artifacts, watermark"
    steps: int = 30
    cfg: float = 7.0
    sampler: str = "dpmpp_2m_sde"
    scheduler: str = "karras"
    width: int = 1024
    height: int = 1024
    seed: int = -1
    denoise: float = 1.0
    # User overrides tracking
    user_overrides: dict = field(default_factory=dict)
    # Conversation context
    reasoning: str = ""                         # Why the AI made these choices


class LLMProvider(abc.ABC):
    """Abstract base for all LLM providers."""

    def __init__(self, config: dict):
        self.config = config

    @abc.abstractmethod
    async def analyze_intent(
        self,
        prompt: str,
        available_models: dict,
        conversation_history: list = None,
        has_reference_image: bool = False,
    ) -> GenerationPlan:
        """
        Analyze user prompt and produce a structured generation plan.
        
        Args:
            prompt: The user's text prompt
            available_models: Dict of available checkpoints, loras, controlnets, ipadapters
            conversation_history: Previous messages for multi-turn context
            has_reference_image: Whether user attached an image
        
        Returns:
            GenerationPlan with all generation parameters decided
        """
        ...

    @abc.abstractmethod
    async def enhance_prompt(
        self,
        prompt: str,
        style_info: Optional[str] = None,
        lora_trigger_words: list = None,
    ) -> str:
        """Rewrite a basic prompt into a detailed Stable Diffusion prompt."""
        ...

    @abc.abstractmethod
    async def analyze_image(self, image_path: str) -> StyleAnalysis:
        """Analyze a reference image for style, keywords, and LoRA suggestions."""
        ...

    @abc.abstractmethod
    async def chat(
        self,
        message: str,
        conversation_history: list = None,
        system_context: str = None,
    ) -> str:
        """
        General chat — used when the bot needs to ask the user questions
        or explain its decisions.
        """
        ...

    @abc.abstractmethod
    async def refine_plan(
        self,
        plan: GenerationPlan,
        user_feedback: str,
        available_models: dict,
    ) -> GenerationPlan:
        """
        Refine a generation plan based on user feedback.
        E.g. "make it more anime" or "use a different model"
        """
        ...


# --- Provider Registry ---
_PROVIDERS = {}


def register_provider(name: str):
    """Decorator to register an LLM provider."""
    def decorator(cls):
        _PROVIDERS[name] = cls
        return cls
    return decorator


def create_llm_provider(config: dict) -> LLMProvider:
    """
    Factory function — creates the right LLM provider based on config.
    
    Config should have:
        llm:
          provider: "ollama"  # or "gemini", "openai"
          ollama: { ... }
          gemini: { ... }
    """
    provider_name = config.get("llm", {}).get("provider", "ollama")
    
    # Import providers to trigger registration
    from . import ollama_provider  # noqa: F401
    try:
        from . import gemini_provider  # noqa: F401
    except ImportError:
        pass
    
    if provider_name not in _PROVIDERS:
        available = list(_PROVIDERS.keys())
        raise ValueError(
            f"Unknown LLM provider '{provider_name}'. Available: {available}"
        )
    
    provider_config = config.get("llm", {}).get(provider_name, {})
    logger.info(f"Creating LLM provider: {provider_name}")
    return _PROVIDERS[provider_name](provider_config)
