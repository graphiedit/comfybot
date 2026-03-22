import asyncio
import sys
import logging
from engine import AIDirectorEngine
import yaml

logging.basicConfig(level=logging.INFO)

async def main():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    engine = AIDirectorEngine(config)
    
    class FakeChannel:
        async def send(self, **kwargs):
            class Msg:
                id = 123
                async def edit(self, **kw):
                    pass
                async def delete(self):
                    pass
            return Msg()

    print("Generating...")
    # Using the same prompt the user might have used or just a generic one
    images = await engine.generate(
        prompt="a futuristic cyberpunk city, neon lights",
        user_id="123",
        channel=FakeChannel(),
        workflow_override="image_z_image_turbo" # start with turbo to see if it works, then check the one they used
    )
    if images:
        print(f"Generated {len(images)} images!")
    else:
        print("Failed to generate.")

asyncio.run(main())
