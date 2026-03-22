import asyncio
import json
import urllib.request
import urllib.parse
from core.workflow_manager import WorkflowManager
from core.comfyui_client import ComfyUIClient
import time

async def main():
    manager = WorkflowManager('data/workflows')
    client = ComfyUIClient("http://192.168.29.23:8188")
    await client.connect()

    types_to_try = ['default', 'qwen_image', 'flux2', 'longcat_image', 'hunyuan_video', 'hudyuan_image']

    for t in types_to_try:
        print(f"\n--- Testing type: {t} ---")
        prompt_data = manager.build_workflow('image_z_image_turbo', prompt="test")
        
        # find the right node and patch it
        patched = False
        for node_id, node in prompt_data.items():
            if node['class_type'] == 'CLIPLoader':
                node['inputs']['type'] = t
                patched = True
                
        if not patched:
            print("CLIPLoader not found!")
            continue

        try:
            workflow_id = await client.queue_prompt(prompt_data)
            print(f"[{t}] Queued: {workflow_id}")
            # we need to wait for result or error
            # wait 3 seconds and fetch history
            time.sleep(4)
            req = urllib.request.Request(f"http://192.168.29.23:8188/history/{workflow_id}")
            with urllib.request.urlopen(req) as response:
                history = json.loads(response.read())
            
            if workflow_id in history:
                status = history[workflow_id].get('status', {})
                if status.get('status_str') == 'error':
                    print(f"[{t}] Execution Failed: {status.get('messages', [['Unknown Error']])[0][0]}")
                else:
                    print(f"[{t}] Execution Succeeded or in progress!")
            else:
                # check queue
                req = urllib.request.Request("http://192.168.29.23:8188/queue")
                with urllib.request.urlopen(req) as response:
                    queue = json.loads(response.read())
                in_queue = False
                for q in queue.get('queue_running', []):
                    if q[1] == workflow_id:
                        in_queue = True
                        print(f"[{t}] Still running!")
                if not in_queue:
                    print(f"[{t}] Execution might have failed early or finished super fast.")
        except Exception as e:
            print(f"[{t}] Request Failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())
