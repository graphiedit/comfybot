"""
LLM Base — Data structures and abstract interface for AI brain.

The AI brain's only job:
1. Understand what the user wants (intent)
2. Pick the best workflow template
3. Enhance the user's prompt
4. Chat conversationally
"""
import abc
import logging
import random
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple

logger = logging.getLogger(__name__)


@dataclass
class GenerationPlan:
    """What the AI decided to do — simple and focused."""
    action: str = "generate"            # generate | chat | upscale | edit | vary
    workflow_template: str = ""          # Which workflow to use
    enhanced_prompt: str = ""            # The enhanced prompt for image gen
    negative_prompt: str = "worst quality, low quality, bad anatomy, bad hands, text, error, blurry, jpeg artifacts, watermark"
    width: int = 1024
    height: int = 1024
    seed: int = -1                      # -1 = random
    reasoning: str = ""                 # Why the AI made these choices
    images: list = field(default_factory=list)  # Uploaded image filenames for ComfyUI
    needs_image: bool = False           # Does this plan need images from user?
    # Extended fields (used by workflow_builder, pipeline, etc.)
    style_category: str = "realistic"
    checkpoint: str = ""
    model_arch: str = "sdxl"
    loras: list = field(default_factory=list)
    use_controlnet: bool = False
    controlnet_type: str = ""
    controlnet_strength: float = 1.0
    use_ipadapter: bool = False
    ipadapter_model: str = ""
    ipadapter_weight: float = 0.6
    steps: int = 30
    cfg: float = 7.0
    sampler: str = "dpmpp_2m_sde"
    scheduler: str = "karras"
    denoise: float = 1.0
    guidance: float = 3.5
    weight_dtype: str = "default"
    user_overrides: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.seed == -1:
            self.seed = random.randint(0, 2**31)


@dataclass
class ChatResponse:
    """Response from AI chat — may include a generation trigger."""
    message: str = ""                   # The text reply
    should_generate: bool = False       # Did the AI decide to generate?
    generation_prompt: str = ""         # If generating, what prompt?
    workflow_hint: str = ""             # Optional workflow preference
    needs_image: bool = False           # Does the AI need images from user?
    questions: list = field(default_factory=list)  # Structured follow-up questions


class LLMProvider(abc.ABC):
    """Abstract base for all LLM providers."""

    def __init__(self, config: dict):
        self.config = config

    @abc.abstractmethod
    async def analyze_intent(
        self,
        prompt: str,
        workflows: Dict[str, str],
        user_history: Optional[List[str]] = None,
    ) -> GenerationPlan:
        """Analyze user's prompt and decide generation strategy.
        
        Args:
            prompt: The user's raw prompt
            workflows: Dict of {template_name: description}
            user_history: Recent prompts from this user
        """
        ...

    @abc.abstractmethod
    async def enhance_prompt(self, prompt: str, style_hints: str = "") -> str:
        """Take a basic prompt and make it detailed for image generation."""
        ...

    @abc.abstractmethod
    async def chat(
        self,
        message: str,
        conversation_history: List[Dict[str, str]],
        workflows: Dict[str, str],
    ) -> ChatResponse:
        """Have a conversation. May trigger image generation autonomously."""
        ...


# Provider registry for plugin-style registration
_PROVIDER_REGISTRY: Dict[str, type] = {}


def register_provider(name: str):
    """Decorator to register an LLM provider class."""
    def decorator(cls):
        _PROVIDER_REGISTRY[name.lower()] = cls
        return cls
    return decorator


def create_single_provider(name: str, config: dict) -> LLMProvider:
    """Create a single LLM provider by name."""
    name_lower = name.lower()
    if name_lower == "ollama":
        from llm.ollama_provider import OllamaProvider
        return OllamaProvider(config)
    elif name_lower == "gemini":
        from llm.gemini_provider import GeminiProvider
        return GeminiProvider(config)
    elif name_lower in _PROVIDER_REGISTRY:
        return _PROVIDER_REGISTRY[name_lower](config)
    else:
        raise ValueError(f"Unknown LLM provider: {name}")


def create_llm_provider(config: dict) -> LLMProvider:
    """Create the appropriate LLM provider based on config.
    
    If multiple providers are configured, returns a ProviderManager
    that handles automatic failover. Otherwise returns a single provider.
    """
    providers_config = config.get("llm", {}).get("providers", [])
    
    if not providers_config:
        raise ValueError("No LLM providers configured in config.yaml")
    
    # If multiple providers, use ProviderManager for failover
    if len(providers_config) > 1:
        from llm.provider_manager import ProviderManager
        manager = ProviderManager(config)
        return manager  # Must call manager.initialize() async later
    
    # Single provider — create directly
    provider_config = providers_config[0]
    name = provider_config.get("name", "").lower()
    return create_single_provider(name, provider_config)
