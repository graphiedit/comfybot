"""
LLM Provider Manager — manages multiple LLM providers with automatic failover.

Chains providers by priority and auto-switches on rate limits, errors,
or provider-specific failures. Tracks cooldowns and provides transparent
access to the best available provider.
"""
import asyncio
import logging
import time
from typing import Optional, List

logger = logging.getLogger(__name__)


class RateLimitError(Exception):
    """Raised when an LLM provider hits a rate limit."""
    def __init__(self, provider_name: str, retry_after: float = 60.0, message: str = ""):
        self.provider_name = provider_name
        self.retry_after = retry_after
        super().__init__(message or f"Rate limited by {provider_name} for {retry_after}s")


class AllProvidersFailedError(Exception):
    """Raised when all LLM providers have failed."""
    def __init__(self, errors: list):
        self.errors = errors
        summary = "; ".join(f"{e[0]}: {e[1]}" for e in errors)
        super().__init__(f"All LLM providers failed: {summary}")


class ProviderManager:
    """
    Manages multiple LLM providers with automatic failover.
    
    Wraps all LLMProvider methods and transparently routes to the best
    available provider, falling back to alternatives on rate limits or errors.
    
    Usage:
        manager = ProviderManager(config)
        await manager.initialize()
        result = await manager.analyze_intent(prompt, models)  # Auto-failover
    """

    def __init__(self, config: dict):
        self.config = config
        self.providers: List[dict] = []  # [{name, provider, priority, cooldown_until}]
        self._active_name: str = ""
        self._initialization_errors: list = []

    async def initialize(self):
        """Initialize all configured providers."""
        from llm.base import create_single_provider
        
        llm_config = self.config.get("llm", {})
        providers_config = llm_config.get("providers", [])
        
        if not providers_config:
            # Legacy single-provider config — wrap it
            providers_config = [{
                "name": llm_config.get("provider", "ollama"),
                "priority": 1,
                **{k: v for k, v in llm_config.items() if k not in ("provider", "providers")},
            }]
        
        for p_conf in sorted(providers_config, key=lambda x: x.get("priority", 99)):
            name = p_conf.get("name", "unknown")
            try:
                provider = create_single_provider(name, p_conf)
                self.providers.append({
                    "name": name,
                    "provider": provider,
                    "priority": p_conf.get("priority", 99),
                    "cooldown_until": 0,
                    "consecutive_failures": 0,
                })
                logger.info(f"Initialized LLM provider: {name} (priority {p_conf.get('priority', 99)})")
            except Exception as e:
                logger.warning(f"Failed to initialize provider '{name}': {e}")
                self._initialization_errors.append((name, str(e)))
        
        if not self.providers:
            raise RuntimeError(
                f"No LLM providers available. Errors: {self._initialization_errors}"
            )
        
        self._active_name = self.providers[0]["name"]
        logger.info(f"Active LLM provider: {self._active_name}")

    def get_available_providers(self) -> list:
        """Get providers that are not in cooldown, sorted by priority."""
        now = time.time()
        available = []
        for p in self.providers:
            if p["cooldown_until"] <= now:
                available.append(p)
            else:
                remaining = p["cooldown_until"] - now
                logger.debug(f"Provider '{p['name']}' in cooldown for {remaining:.0f}s")
        return sorted(available, key=lambda x: x["priority"])

    def mark_rate_limited(self, provider_entry: dict, retry_after: float = 60.0):
        """Mark a provider as rate-limited with a cooldown period."""
        provider_entry["cooldown_until"] = time.time() + retry_after
        provider_entry["consecutive_failures"] += 1
        logger.warning(
            f"Provider '{provider_entry['name']}' rate-limited for {retry_after}s "
            f"(failures: {provider_entry['consecutive_failures']})"
        )

    def mark_failed(self, provider_entry: dict, cooldown: float = 30.0):
        """Mark a provider as temporarily failed."""
        provider_entry["consecutive_failures"] += 1
        # Exponential backoff: 30s, 60s, 120s, 240s...
        backoff = cooldown * (2 ** min(provider_entry["consecutive_failures"] - 1, 4))
        provider_entry["cooldown_until"] = time.time() + backoff
        logger.warning(
            f"Provider '{provider_entry['name']}' failed. "
            f"Cooling down for {backoff:.0f}s"
        )

    def mark_success(self, provider_entry: dict):
        """Reset failure count on success."""
        provider_entry["consecutive_failures"] = 0

    @property
    def active_provider_name(self) -> str:
        return self._active_name

    async def call(self, method_name: str, *args, **kwargs):
        """
        Call an LLM method with automatic failover.
        
        Tries each available provider in priority order. On rate limit or
        error, switches to the next provider.
        """
        errors = []
        available = self.get_available_providers()
        
        if not available:
            # No providers available — find the one with shortest cooldown
            soonest = min(self.providers, key=lambda p: p["cooldown_until"])
            wait_time = soonest["cooldown_until"] - time.time()
            if wait_time > 0 and wait_time < 120:
                logger.info(f"All providers on cooldown. Waiting {wait_time:.0f}s for '{soonest['name']}'...")
                await asyncio.sleep(wait_time)
                available = [soonest]
            else:
                raise AllProvidersFailedError(
                    [(p["name"], "in cooldown") for p in self.providers]
                )
        
        for p_entry in available:
            provider = p_entry["provider"]
            
            if not hasattr(provider, method_name):
                continue
            
            try:
                method = getattr(provider, method_name)
                result = await method(*args, **kwargs)
                
                # Success!
                self.mark_success(p_entry)
                self._active_name = p_entry["name"]
                return result
                
            except RateLimitError as e:
                logger.warning(f"Provider '{p_entry['name']}' rate limited: {e}")
                self.mark_rate_limited(p_entry, e.retry_after)
                errors.append((p_entry["name"], f"rate limited: {e}"))
                continue
                
            except Exception as e:
                error_str = str(e).lower()
                
                # Check if this looks like a rate limit error (HTTP 429 or quota exceeded)
                if any(kw in error_str for kw in ["429", "rate limit", "quota", "resource_exhausted", "capacity"]):
                    retry_after = 60.0 # Default
                    
                    # Try to extract retry delay if present in error message
                    import re
                    match = re.search(r"retry in ([\d\.]+)", error_str)
                    if match:
                        retry_after = float(match.group(1))
                    elif "seconds" in error_str:
                        match = re.search(r"(\d+)\s*seconds?", error_str)
                        if match:
                            retry_after = float(match.group(1))
                    
                    self.mark_rate_limited(p_entry, retry_after)
                    errors.append((p_entry["name"], f"rate limited: {e}"))
                else:
                    logger.error(f"Provider '{p_entry['name']}' call failed: {e}", exc_info=True)
                    self.mark_failed(p_entry)
                    errors.append((p_entry["name"], str(e)))
                continue
        
        raise AllProvidersFailedError(errors)

    # --- Proxy methods for LLMProvider interface ---
    # These delegate to self.call() for transparent failover.

    async def analyze_intent(self, prompt, available_models, **kwargs):
        return await self.call("analyze_intent", prompt, available_models, **kwargs)

    async def enhance_prompt(self, prompt, style_hints=None, **kwargs):
        return await self.call("enhance_prompt", prompt, style_hints, **kwargs)

    async def analyze_image(self, image_path, **kwargs):
        return await self.call("analyze_image", image_path, **kwargs)

    async def chat(self, message, conversation_history=None, system_context=None, **kwargs):
        return await self.call("chat", message, conversation_history, system_context, **kwargs)

    async def refine_plan(self, plan, feedback, available_models, **kwargs):
        return await self.call("refine_plan", plan, feedback, available_models, **kwargs)

    async def check_plan_completeness(self, plan, available_models, **kwargs):
        """Check if a generation plan is complete and sensible."""
        return await self.call("check_plan_completeness", plan, available_models, **kwargs)

    def get_status_summary(self) -> str:
        """Get a human-readable status of all providers."""
        lines = []
        now = time.time()
        for p in self.providers:
            status = "✅ available"
            if p["cooldown_until"] > now:
                remaining = p["cooldown_until"] - now
                status = f"⏳ cooldown ({remaining:.0f}s)"
            if p["name"] == self._active_name:
                status += " [ACTIVE]"
            lines.append(f"  {p['name']} (priority {p['priority']}): {status}")
        return "\n".join(lines)
