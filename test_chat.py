import asyncio
import logging
from llm.gemini_provider import GeminiProvider

logging.basicConfig(level=logging.INFO)

async def main():
    config = {'api_key': 'AIzaSyDCJyXFZkG6oEHNGXXlNs0u483Zr4-mGmo'}
    provider = GeminiProvider(config)
    history = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "Hello! I am your AI assistant."}
    ]
    
    print("Sending chat request with mention...")
    try:
        response = await provider.chat(
            message="<@1485046907943756264> i wnat you to use z image turbo model and genrate image of lord krishna holding flute in his hand",
            conversation_history=history,
            system_context="If they want to generate, tell them to use /generate command."
        )
        print("Response:", response)
    except Exception as e:
        print("Exception:", e)

asyncio.run(main())
