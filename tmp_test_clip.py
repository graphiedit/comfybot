import asyncio
import json
import logging
from core.workflow_manager import WorkflowManager
from core.comfyui_client import ComfyUIClient

logging.basicConfig(level=logging.INFO)

async def main():
    manager = WorkflowManager('data/workflows')
    client = ComfyUIClient("http://192.168.29.23:8188")
    await client.connect()

    types_to_try = ['qwen_image', 'flux2', 'longcat_image', 'hunyuan_image', 'default']

    for t in types_to_try:
        print(f"Testing type: {t}")
        # modify the template directly in memory
        manager.templates['image_z_image_turbo']['nodes'].append({
            "id": "patch_type", "type": "patch", "clip_type": t
        })
        
        # build workflow API format
        prompt_data = manager.build_workflow('image_z_image_turbo', prompt="test")
        
        # find the right node and patch it
        for node_id, node in prompt_data.items():
            if node['class_type'] == 'CLIPLoader':
                node['inputs']['type'] = t
                
        # submit
        try:
            workflow_id = await client.queue_prompt(prompt_data)
            print(f"[{t}] Queued: {workflow_id}")
            # we need to wait for result or error
            # but client.queue_prompt doesn't wait for completion.
            # let's just check the status in a loop
            import time
            time.sleep(2)  # wait for error message to arrive at server
            # let's do a short loop
        except Exception as e:
            print(f"[{t}] Failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())
