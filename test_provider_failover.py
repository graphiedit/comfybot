"""Test script for the LLM ProviderManager failover logic."""
import sys
import asyncio
import time

sys.path.insert(0, ".")

from llm.provider_manager import ProviderManager, RateLimitError

# Mock LLM Providers for testing
class MockFastProvider:
    def __init__(self, config):
        self.config = config
        self.call_count = 0
        
    async def analyze_intent(self, *args, **kwargs):
        self.call_count += 1
        return "fast_result"

class MockRateLimitingProvider:
    def __init__(self, config):
        self.config = config
        self.call_count = 0
        
    async def analyze_intent(self, *args, **kwargs):
        self.call_count += 1
        raise RateLimitError("MockRateLimiter", retry_after=5.0)

class MockFailingProvider:
    def __init__(self, config):
        self.config = config
        self.call_count = 0
        
    async def analyze_intent(self, *args, **kwargs):
        self.call_count += 1
        raise ValueError("Something went fundamentally wrong")

async def test_provider_manager():
    print("="*50)
    print("TEST: LLM Provider Failover")
    print("="*50)
    
    # 1. Test normal auto-failover
    # We'll set up a manager without the real create_llm_provider
    # by directly injecting our mocks.
    manager = ProviderManager({})
    
    manager.providers = [
        {"name": "primary", "provider": MockRateLimitingProvider({}), "priority": 1, "cooldown_until": 0, "consecutive_failures": 0},
        {"name": "backup1", "provider": MockFailingProvider({}), "priority": 2, "cooldown_until": 0, "consecutive_failures": 0},
        {"name": "backup2", "provider": MockFastProvider({}), "priority": 3, "cooldown_until": 0, "consecutive_failures": 0},
    ]
    
    print("Calling analyze_intent on ProviderManager...")
    result = await manager.analyze_intent("test prompt", {})
    
    print(f"Result: {result}")
    assert result == "fast_result", "Should have returned the result from the working provider"
    
    print("Provider status:")
    print(manager.get_status_summary())
    
    print("\n✅ Failover test passed: Successfully skipped rate-limited and failing providers to reach the working one.")

if __name__ == "__main__":
    asyncio.run(test_provider_manager())
