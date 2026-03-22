import asyncio
import aiohttp
import json

async def test_ollama():
    url = "http://127.0.0.1:11434/api/chat"
    payload = {
        "model": "deepseek-r1:1.5b",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": False
    }
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(url, json=payload) as resp:
                print(f"Status: {resp.status}")
                text = await resp.text()
                print(f"Response: {text[:200]}")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_ollama())
