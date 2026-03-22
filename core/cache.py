"""
Simple caching layer for LLM responses and embeddings.
"""
import copy
import logging
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class CacheEntry:
    def __init__(self, value: Any, ttl_seconds: int):
        self.value = value
        self.expiry = time.time() + ttl_seconds


class MemoryCache:
    """Basic in-memory temporal cache with LRU eviction."""

    def __init__(self, max_entries: int = 100, default_ttl: int = 3600):
        self._cache: Dict[str, CacheEntry] = {}
        self.max_entries = max_entries
        self.default_ttl = default_ttl

    def get(self, key: str) -> Optional[Any]:
        """Get a value if it exists and hasn't expired."""
        entry = self._cache.get(key)
        
        if not entry:
            return None
            
        if time.time() > entry.expiry:
            del self._cache[key]
            return None
            
        return copy.deepcopy(entry.value)

    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set a value in the cache, managing the max size."""
        # Evict if full
        if len(self._cache) >= self.max_entries and key not in self._cache:
            # Remove oldest (first item inserted)
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
            
        self._cache[key] = CacheEntry(
            copy.deepcopy(value), 
            ttl if ttl is not None else self.default_ttl
        )

    def clear(self):
        """Clear all entries."""
        self._cache.clear()

    def generate_prompt_key(self, prompt: str, overrides: dict = None) -> str:
        """Helper to create a deterministic key for a prompt."""
        key = prompt.lower().strip()
        if overrides:
            # Sort keys for deterministic string representation
            override_str = "|".join([f"{k}:{v}" for k, v in sorted(overrides.items()) if k != "seed"])
            key = f"{key}|{override_str}"
        return key
