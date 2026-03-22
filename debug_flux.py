import json
from core.workflow_manager import WorkflowManager
from llm.base import GenerationPlan

wm = WorkflowManager('data')
plan = GenerationPlan(
    workflow_template='image_flux2_klein_image_edit_9b_distilled',
    enhanced_prompt='test', negative_prompt='test',
    width=1024, height=1024
)
wf, _ = wm.build_workflow(plan)
json.dump(wf, open('flux_dump.json', 'w', encoding='utf-8'), indent=2)

# Check for missing refs
for nid, nd in wf.items():
    ct = nd.get("class_type", "")
    for k, v in nd.get('inputs', {}).items():
        if isinstance(v, list) and len(v) == 2:
            ref = str(v[0])
            if ref not in wf:
                print(f"BROKEN: Node {nid} ({ct}) input '{k}' -> missing node {ref}")

# Check nodes with empty/missing critical inputs
for nid, nd in wf.items():
    ct = nd.get("class_type", "")
    if ct in ("VAEEncode", "ReferenceLatent"):
        print(f"Node {nid} ({ct}): {json.dumps(nd['inputs'])}")
