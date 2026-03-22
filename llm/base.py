"""
Abstract LLM Provider — swappable backend for AI intelligence.

Supports: Ollama (local), Gemini, OpenAI, or any custom provider.
To add a new provider, subclass LLMProvider and register it.
"""
import abc
import logging
from dataclasses import dataclass, field
from typing import Optional, List

logger = logging.getLogger(__name__)

# Re-export for convenience
from llm.provider_manager import RateLimitError, AllProvidersFailedError  # noqa: F401


@dataclass
class StyleAnalysis:
    """Result of analyzing a reference image."""
    style: str = ""
    keywords: str = ""
    lora_search: list = field(default_factory=list)
    raw: str = ""


@dataclass
class PlanReview:
    """Result of reviewing a generation plan for completeness."""
    is_complete: bool = True
    needs_clarification: bool = False
    question: str = ""                         # Question to ask the user
    suggestions: List[str] = field(default_factory=list)  # Non-blocking tips
    warnings: List[str] = field(default_factory=list)     # Issues to flag
    confidence: float = 0.9                     # How confident the AI is


@dataclass
class GenerationPlan:
    """Structured output from intent analysis — describes HOW to generate."""
    action: str = "generate"                    # generate | upscale | edit | vary
    workflow_template: str = ""                 # If a user workflow template was chosen
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
    weight_dtype: str = "default"               # Corrected by ErrorDiagnosticAgent if needed
    guidance: float = 3.5                      # Flux guidance
    force_sdxl: bool = False                   # Flag to force SDXL fallback
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

    async def check_plan_completeness(
        self,
        plan: GenerationPlan,
        available_models: dict,
    ) -> PlanReview:
        """
        Review a generation plan for completeness and quality.
        Returns suggestions, warnings, or clarification questions.
        Default implementation — override for smarter behavior.
        """
        review = PlanReview()
        
        # Basic checks any provider can do
        if not plan.enhanced_prompt and not plan.user_overrides.get("action") == "dream":
            review.is_complete = False
            review.needs_clarification = True
            review.question = "Your prompt seems very short. Could you describe what you'd like in more detail?"
        
        if plan.width > 2048 or plan.height > 2048:
            review.warnings.append(f"Resolution {plan.width}×{plan.height} is very high and may cause OOM.")
        
        if plan.steps > 50:
            review.suggestions.append(f"Using {plan.steps} steps — this will be slow. 20-30 is often sufficient.")
        
        return review


# --- Provider Registry ---
_PROVIDERS = {}


def register_provider(name: str):
    """Decorator to register an LLM provider."""
    def decorator(cls):
        _PROVIDERS[name] = cls
        return cls
    return decorator


def _import_providers():
    """Import all provider modules to trigger registration."""
    try:
        from . import ollama_provider  # noqa: F401
    except ImportError as e:
        logger.warning(f"Could not import Ollama provider: {e}")
    try:
        from . import gemini_provider  # noqa: F401
    except ImportError as e:
        logger.warning(f"Could not import Gemini provider: {e}")


def create_single_provider(name: str, provider_config: dict) -> LLMProvider:
    """
    Create a single LLM provider by name.
    Used internally by ProviderManager.
    """
    _import_providers()
    
    # Normalize name
    provider_name = name.split("_")[0] if "_" in name else name  # e.g. "gemini_backup" → "gemini"
    
    if provider_name not in _PROVIDERS:
        available = list(_PROVIDERS.keys())
        raise ValueError(f"Unknown LLM provider '{provider_name}'. Available: {available}")
    
    logger.info(f"Creating LLM provider: {name} (type={provider_name})")
    return _PROVIDERS[provider_name](provider_config)


def create_llm_provider(config: dict):
    """
    Factory function — creates an LLM ProviderManager with failover.
    
    If config has multiple providers, creates a ProviderManager.
    If config has a single provider, still wraps it in a ProviderManager
    for consistent interface.
    
    Config supports both legacy and new format:
    
    Legacy:
        llm:
          provider: "ollama"
          ollama: { ... }
    
    New (multi-provider):
        llm:
          providers:
            - name: gemini
              api_key: ...
              priority: 1
            - name: ollama
              priority: 2
    """
    from llm.provider_manager import ProviderManager
    return ProviderManager(config)
