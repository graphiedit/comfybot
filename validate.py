import asyncio
import yaml
import json
from core.workflow_manager import WorkflowManager
from core.comfyui_client import ComfyUIClient
from llm.base import GenerationPlan

async def validate():
    with open('config.yaml', 'r') as f:
        cfg = yaml.safe_load(f)
        
    client = ComfyUIClient(cfg['comfyui'])
    wm = WorkflowManager('data')
    
    with open('validate.log', 'w', encoding='utf-8') as out:
        for t_name in wm.templates:
            plan = GenerationPlan(
                workflow_template=t_name,
                enhanced_prompt="test",
                negative_prompt="test",
                width=1024,
                height=1024
            )
            
            wf, succ = wm.build_workflow(plan)
            if not succ:
                out.write(f"{t_name}: Build failed\n")
                continue
                
            try:
                pid = await client.queue_prompt(wf)
                out.write(f"✅ {t_name}: OK -> {pid}\n")
            except Exception as e:
                out.write(f"❌ {t_name}: FAILED\n")
                msg = str(e)
                if "rejected workflow:" in msg:
                    err_json = msg.split("rejected workflow: ")[1]
                    try:
                        data = json.loads(err_json)
                        node_errs = data.get("node_errors", {})
                        for nid, n_data in node_errs.items():
                            c_type = n_data.get("class_type")
                            out.write(f"  - Node {nid} ({c_type}):\n")
                            for err in n_data.get("errors", []):
                                out.write(f"    * {err.get('message')}: {err.get('details')}\n")
                    except:
                        out.write(f"  - {msg}\n")
                else:
                    out.write(f"  - {msg}\n")

if __name__ == '__main__':
    asyncio.run(validate())
