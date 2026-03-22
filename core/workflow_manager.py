"""
Workflow Manager — loads ComfyUI workflows, converts formats, injects parameters.

Handles the complete pipeline:
1. Scan data/workflows/ for *.json files with matching *.txt descriptions
2. Convert ComfyUI graph-editor format to API prompt format
3. Inject user parameters (prompt, seed, dimensions) into the workflow

IMPORTANT: The workflow JSON files from ComfyUI come in "graph-editor" format
with nodes[], links[], definitions.subgraphs[]. The API expects a flat dict:
{"node_id": {"class_type": "...", "inputs": {...}}}

This module handles that conversion automatically.
"""
import json
import copy
import random
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

from llm.base import GenerationPlan
from core.workflow_analyzer import WorkflowAnalyzer, WorkflowProfile

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════
# Widget-to-input mapping for known ComfyUI node types
# Maps class_type -> ordered list of widget input names
# This tells us how to convert positional widgets_values to named inputs
# ═══════════════════════════════════════════════════════════════
WIDGET_MAPS = {
    "KSampler": ["seed", "control_after_generate", "steps", "cfg", "sampler_name", "scheduler", "denoise"],
    "KSamplerAdvanced": ["add_noise", "noise_seed", "control_after_generate", "steps", "cfg", "sampler_name", "scheduler", "start_at_step", "end_at_step", "return_with_leftover_noise"],
    "CheckpointLoaderSimple": ["ckpt_name"],
    "UNETLoader": ["unet_name", "weight_dtype"],
    "CLIPLoader": ["clip_name", "type", "device"],
    "DualCLIPLoader": ["clip_name1", "clip_name2", "type"],
    "VAELoader": ["vae_name"],
    "CLIPTextEncode": ["text"],
    "EmptyLatentImage": ["width", "height", "batch_size"],
    "EmptySD3LatentImage": ["width", "height", "batch_size"],
    "EmptyFlux2LatentImage": ["width", "height", "batch_size"],
    "SaveImage": ["filename_prefix"],
    "PreviewImage": [],
    "ConditioningZeroOut": [],
    "ModelSamplingAuraFlow": ["shift"],
    "ModelSamplingFlux": ["max_shift", "base_shift", "width", "height"],
    "FluxGuidance": ["guidance"],
    "LoraLoader": ["lora_name", "strength_model", "strength_clip"],
    "LoraLoaderModelOnly": ["lora_name", "strength_model"],
    "LoadImage": ["image", "upload"],
    "ImageScale": ["upscale_method", "width", "height", "crop"],
    "LatentUpscale": ["upscale_method", "width", "height", "crop"],
    "SamplerCustom": ["add_noise", "noise_seed", "cfg", "positive", "negative", "sampler", "sigmas"],
    "BasicScheduler": ["scheduler", "steps", "denoise"],
    "BasicGuider": [],
    "RandomNoise": ["noise_seed"],
    "SamplerCustomAdvanced": [],
    "KSamplerSelect": ["sampler_name"],
    "Flux2Scheduler": ["steps", "shift_start", "shift_end"],
    "CFGGuider": ["cfg"],
    "ImageScaleToTotalPixels": ["upscale_method", "megapixels", "resolution_steps"],
    "GetImageSize": [],
    "ImageScaleToMaxDimension": ["upscale_method", "largest_size"],
    "ModelPatchLoader": ["name"],
    "QwenImageDiffsynthControlnet": ["strength"],
    "OpenposePreprocessor": ["detect_hand", "detect_body", "detect_face", "resolution", "bbox_detector"],
    "Canny": ["low_threshold", "high_threshold"],
    "TextEncodeQwenImageEditPlus": ["prompt"],
    "PrimitiveStringMultiline": ["value"],
    "CFGNorm": ["strength"],
    "FluxKontextImageScale": [],
    "ReferenceLatent": [],
    "VAEDecode": [],
    "VAEEncode": [],
}

# Input slot name maps — used to figure out input names when we only know the slot index
SLOT_MAPS = {
    "SaveImage": {0: "images"},
    "PreviewImage": {0: "images"},
    "VAEDecode": {0: "samples", 1: "vae"},
    "VAEEncode": {0: "pixels", 1: "vae"},
    "KSampler": {0: "model", 1: "positive", 2: "negative", 3: "latent_image"},
    "CLIPTextEncode": {0: "clip"},
    "ModelSamplingAuraFlow": {0: "model"},
    "ModelSamplingFlux": {0: "model"},
    "FluxGuidance": {0: "conditioning"},
    "ConditioningZeroOut": {0: "conditioning"},
    "CFGGuider": {0: "model", 1: "positive", 2: "negative"},
    "SamplerCustomAdvanced": {0: "noise", 1: "guider", 2: "sampler", 3: "sigmas", 4: "latent_image"},
    "EmptyFlux2LatentImage": {0: "width", 1: "height"},
    "ImageScaleToTotalPixels": {0: "image"},
    "GetImageSize": {0: "image"},
    "ReferenceLatent": {0: "positive", 1: "negative", 2: "latent"},
    "CFGNorm": {0: "model"},
    "LoraLoaderModelOnly": {0: "model"},
    "TextEncodeQwenImageEditPlus": {0: "clip", 1: "vae", 2: "image1", 3: "image2", 4: "image3", 5: "prompt"},
    "FluxKontextImageScale": {0: "image"},
    "ImageScaleToMaxDimension": {0: "image"},
    "QwenImageDiffsynthControlnet": {0: "model", 1: "positive", 2: "negative"},
    "ModelPatchLoader": {0: "model"},
}

# Nodes to always skip during conversion
SKIP_TYPES = {"MarkdownNote", "Note", "Reroute"}


class WorkflowManager:
    """Manages ComfyUI workflow templates."""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.workflows_dir = self.data_dir / "workflows"
        self.templates: Dict[str, dict] = {}          # name -> raw JSON data
        self.descriptions: Dict[str, str] = {}         # name -> text description
        self.profiles: Dict[str, WorkflowProfile] = {} # name -> analyzed profile
        self._analyzer = WorkflowAnalyzer()
        self._load_templates()

    def _load_templates(self):
        """Scan workflows directory for JSON files with optional .txt descriptions."""
        if not self.workflows_dir.exists():
            logger.warning(f"Workflows directory not found: {self.workflows_dir}")
            return

        for json_path in sorted(self.workflows_dir.glob("*.json")):
            name = json_path.stem
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.templates[name] = data

                # Look for matching .txt description
                txt_path = json_path.with_suffix(".txt")
                if txt_path.exists():
                    self.descriptions[name] = txt_path.read_text(encoding="utf-8").strip()
                else:
                    self.descriptions[name] = f"Workflow: {name}"

                # Auto-analyze workflow structure
                profile = self._analyzer.analyze(name, data, self.descriptions[name])
                self.profiles[name] = profile

                logger.info(
                    f"Loaded workflow: {name} "
                    f"(images={profile.min_images}, arch={profile.architecture}, "
                    f"caps={profile.capabilities})"
                )
            except Exception as e:
                logger.error(f"Failed to load workflow {name}: {e}")

    def get_workflow_list(self) -> Dict[str, str]:
        """Return dict of {name: description} for all available workflows."""
        return dict(self.descriptions)

    def get_workflow_list_rich(self) -> Dict[str, dict]:
        """Return enriched workflow metadata for the LLM."""
        result = {}
        for name, profile in self.profiles.items():
            result[name] = profile.to_dict()
        return result

    def get_workflow_summaries_for_llm(self) -> str:
        """Generate a formatted summary of all workflows for LLM system prompts."""
        if not self.profiles:
            return "No workflows available."
        parts = []
        for name, profile in self.profiles.items():
            parts.append(profile.to_llm_summary())
        return "\n\n".join(parts)

    def get_profile(self, name: str) -> Optional[WorkflowProfile]:
        """Get the analyzed profile for a workflow template."""
        return self.profiles.get(name)

    def find_workflows_for_intent(
        self, has_image: bool = False, num_images: int = 0,
        wants_edit: bool = False, wants_controlnet: bool = False,
    ) -> Dict[str, WorkflowProfile]:
        """Filter workflows by intent. Returns matching profiles sorted by relevance."""
        matches = {}
        for name, profile in self.profiles.items():
            score = 0

            # If user has no images, prefer text-only workflows
            if not has_image and not profile.requires_image:
                score += 10
            elif not has_image and profile.requires_image:
                score -= 100  # Can't use without images

            # If user has images, prefer image-capable workflows
            if has_image and profile.requires_image:
                score += 10
                if num_images >= profile.min_images:
                    score += 5

            # Intent matching
            if wants_edit and "image_edit" in profile.capabilities:
                score += 15
            if wants_controlnet and "controlnet" in profile.capabilities:
                score += 15

            if score > -50:  # Only include viable options
                matches[name] = profile

        return matches

    def add_workflow(self, name: str, json_data: dict, description: str = "") -> WorkflowProfile:
        """Add a new workflow at runtime (e.g., from Discord upload)."""
        self.templates[name] = json_data
        self.descriptions[name] = description or f"Workflow: {name}"

        # Save to disk
        json_path = self.workflows_dir / f"{name}.json"
        txt_path = self.workflows_dir / f"{name}.txt"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2)
        if description:
            txt_path.write_text(description, encoding="utf-8")

        # Analyze
        profile = self._analyzer.analyze(name, json_data, description)
        self.profiles[name] = profile

        logger.info(f"Added workflow: {name} (images={profile.min_images}, caps={profile.capabilities})")
        return profile

    def update_description(self, name: str, description: str):
        """Update the description of an existing workflow and save it to a .txt file."""
        if name in self.templates:
            self.descriptions[name] = description
            txt_path = self.workflows_dir / f"{name}.txt"
            txt_path.write_text(description, encoding="utf-8")
            logger.info(f"Updated description for workflow '{name}'")


    def build_workflow(self, plan: GenerationPlan) -> Tuple[dict, bool]:
        """Build a ready-to-submit API workflow from a plan.
        
        Returns (workflow_dict, success_bool).
        """
        template_name = plan.workflow_template
        
        if template_name not in self.templates:
            logger.error(f"Template '{template_name}' not found")
            # Try to find a similar one
            for name in self.templates:
                if template_name.lower() in name.lower():
                    template_name = name
                    logger.info(f"Using similar template: {name}")
                    break
            else:
                if self.templates:
                    template_name = next(iter(self.templates))
                    logger.warning(f"Falling back to first template: {template_name}")
                else:
                    return {}, False

        raw_data = copy.deepcopy(self.templates[template_name])

        # Step 1: Convert to API format if needed
        if self._is_graph_format(raw_data):
            logger.info(f"Converting {template_name} from graph-editor to API format")
            api_workflow = self._convert_graph_to_api(raw_data)
        elif self._is_api_format(raw_data):
            api_workflow = raw_data
        else:
            logger.error(f"Unknown workflow format for {template_name}")
            return {}, False

        if not api_workflow:
            logger.error(f"Conversion produced empty workflow for {template_name}")
            return {}, False

        # Step 2: Inject user parameters
        self._inject_prompt(api_workflow, plan.enhanced_prompt)
        self._inject_negative_prompt(api_workflow, plan.negative_prompt)
        self._inject_seed(api_workflow, plan.seed)
        self._inject_dimensions(api_workflow, plan.width, plan.height)

        # Step 3: Inject images if provided
        if hasattr(plan, 'images') and plan.images:
            self._inject_images(api_workflow, plan.images, template_name)

        logger.info(f"Built workflow: {template_name} ({len(api_workflow)} nodes)")
        return api_workflow, True

    # ═══════════════════════════════════════════════════════════════
    # Format Detection
    # ═══════════════════════════════════════════════════════════════

    def _is_api_format(self, data: dict) -> bool:
        if not isinstance(data, dict):
            return False
        sample = next(iter(data.values()), None) if data else None
        return isinstance(sample, dict) and "class_type" in sample

    def _is_graph_format(self, data: dict) -> bool:
        return isinstance(data, dict) and "nodes" in data and "links" in data

    # ═══════════════════════════════════════════════════════════════
    # Graph-Editor → API Format Conversion (Complete Rewrite)
    # ═══════════════════════════════════════════════════════════════

    def _convert_graph_to_api(self, data: dict) -> dict:
        """Convert ComfyUI graph-editor JSON to API prompt format.
        
        Handles:
        - Regular top-level nodes
        - Subgraph wrapper nodes (expanded recursively)
        - Bypassed/muted nodes (mode >= 2) are SKIPPED
        - Post-conversion: remap subgraph outputs and prune broken refs
        """
        api = {}
        
        # Reset per-conversion state
        self._sg_outputs = {}  # wrapper_id -> {output_slot -> (inner_node_id, inner_slot)}

        # Collect all subgraph definitions
        subgraphs = {}
        for sg in data.get("definitions", {}).get("subgraphs", []):
            subgraphs[sg["id"]] = sg

        # Collect bypassed node IDs (mode >= 2 means muted/bypassed in ComfyUI)
        bypassed_ids = set()
        for node in data.get("nodes", []):
            if node.get("mode", 0) >= 2:
                bypassed_ids.add(node["id"])

        # Parse top-level links
        top_links = self._parse_top_links(data.get("links", []))

        # Process each top-level node
        for node in data.get("nodes", []):
            node_type = node.get("type", "")
            node_id = node["id"]

            # Skip decorative, bypassed, or muted nodes
            if node_type in SKIP_TYPES or node_id in bypassed_ids:
                continue

            if node_type in subgraphs:
                # Expand subgraph wrapper into flat API nodes
                self._expand_subgraph(api, subgraphs[node_type], node, node_id, top_links, subgraphs)
            else:
                # Regular node
                api[str(node_id)] = {
                    "class_type": node_type,
                    "inputs": self._build_node_inputs(node, top_links),
                }

        # Post-process: remap all references to subgraph wrappers → their inner output nodes
        self._remap_subgraph_outputs(api)

        # Post-process: remove nodes that reference non-existent nodes (from bypassed subgraphs)
        self._prune_broken_refs(api)

        return api

    def _parse_top_links(self, links_data) -> dict:
        """Parse top-level links: [id, from_node, from_slot, to_node, to_slot, type]."""
        links = {}
        for link in links_data:
            if isinstance(link, list) and len(link) >= 6:
                link_id, from_node, from_slot, to_node, to_slot, link_type = link[:6]
                links[link_id] = {
                    "from_node": from_node,
                    "from_slot": from_slot,
                    "to_node": to_node,
                    "to_slot": to_slot,
                    "type": link_type,
                }
        return links

    def _expand_subgraph(self, api: dict, sg: dict, wrapper_node: dict,
                         wrapper_id: int, top_links: dict, subgraphs: dict):
        """Expand a subgraph's inner nodes into the flat API dict.
        
        Algorithm:
        1. Parse inner links into a connection map: {target_id: {target_slot: (origin_id, origin_slot, link_id)}}
        2. For each inner node, build its inputs from widgets + links
        3. For -10 origins (subgraph proxy inputs), resolve through the wrapper node's top-level links
        4. For -20 targets (subgraph proxy outputs), record in self._sg_outputs for post-processing
        """
        inner_nodes = sg.get("nodes", [])
        inner_links = sg.get("links", [])
        sg_inputs = sg.get("inputs", [])

        # Step 1: Parse all inner links
        # connections[target_id][target_slot] = (origin_id, origin_slot, link_id)
        connections = {}
        
        for link in inner_links:
            if isinstance(link, dict):
                l_id = link.get("id")
                o_id = link.get("origin_id")
                o_slot = link.get("origin_slot", 0)
                t_id = link.get("target_id")
                t_slot = link.get("target_slot", 0)
            elif isinstance(link, list) and len(link) >= 6:
                l_id, o_id, o_slot, t_id, t_slot = link[0], link[1], link[2], link[3], link[4]
            else:
                continue

            if t_id == -20:
                # Subgraph output → record for post-processing
                if wrapper_id not in self._sg_outputs:
                    self._sg_outputs[wrapper_id] = {}
                self._sg_outputs[wrapper_id][t_slot] = (o_id, o_slot)
            else:
                if t_id not in connections:
                    connections[t_id] = {}
                connections[t_id][t_slot] = (o_id, o_slot, l_id)

        # Step 2: Build each inner node
        for inode in inner_nodes:
            nid = str(inode["id"])
            class_type = inode.get("type", "")

            if class_type in SKIP_TYPES:
                continue

            # Handle nested subgraph wrappers
            if subgraphs and class_type in subgraphs:
                nested_sg = subgraphs[class_type]
                # Build synthetic top-links for the nested wrapper
                nested_top_links = self._build_nested_top_links(inode, connections, inner_links)
                self._expand_subgraph(api, nested_sg, inode, inode["id"], nested_top_links, subgraphs)
                continue

            # Build inputs from widget values (fallback defaults)
            inputs = {}
            widget_map = WIDGET_MAPS.get(class_type, [])
            widgets = inode.get("widgets_values", [])
            node_inputs = inode.get("inputs", [])

            # Map widget values → named inputs
            for i, wname in enumerate(widget_map):
                if i < len(widgets):
                    inputs[wname] = widgets[i]

            # Apply link connections (overwrite widget values where linked)
            if inode["id"] in connections:
                for t_slot, (o_id, o_slot, l_id) in connections[inode["id"]].items():
                    # Find input name by matching link ID to the node's input definitions
                    inp_name = self._find_input_name(node_inputs, l_id, class_type, t_slot)
                    
                    if o_id == -10:
                        # Proxied from parent — resolve through wrapper's inputs
                        resolved = self._resolve_parent_input(o_slot, sg_inputs, wrapper_node, top_links)
                        if resolved:
                            inputs[inp_name] = [str(resolved[0]), resolved[1]]
                    else:
                        inputs[inp_name] = [str(o_id), o_slot]

            api[nid] = {"class_type": class_type, "inputs": inputs}

    def _build_nested_top_links(self, inode: dict, parent_connections: dict, parent_links: list) -> dict:
        """Build synthetic top-links for a nested subgraph wrapper node."""
        nested_top_links = {}
        
        # Inbound connections from parent to this nested wrapper
        if inode["id"] in parent_connections:
            for t_slot, (o_id, o_slot, l_id) in parent_connections[inode["id"]].items():
                fake_link_id = f"nested_in_{inode['id']}_{t_slot}"
                nested_top_links[fake_link_id] = {
                    "from_node": o_id, "from_slot": o_slot,
                    "to_node": inode["id"], "to_slot": t_slot,
                }
                # Patch the wrapper's inputs so _resolve_parent_input can find them
                node_inputs = inode.get("inputs", [])
                patched = False
                for inp in node_inputs:
                    if inp.get("link") == l_id:
                        inp["link"] = fake_link_id
                        patched = True
                        break
                
                if not patched and t_slot < len(node_inputs):
                    node_inputs[t_slot]["link"] = fake_link_id

        # Outbound connections from this nested wrapper to siblings
        for link in parent_links:
            if isinstance(link, dict):
                if link.get("origin_id") == inode["id"]:
                    nested_top_links[link["id"]] = {
                        "from_node": link["origin_id"], "from_slot": link.get("origin_slot", 0),
                        "to_node": link["target_id"], "to_slot": link.get("target_slot", 0),
                    }
            elif isinstance(link, list) and len(link) >= 5:
                if link[1] == inode["id"]:
                    nested_top_links[link[0]] = {
                        "from_node": link[1], "from_slot": link[2],
                        "to_node": link[3], "to_slot": link[4],
                    }

        return nested_top_links

    def _find_input_name(self, node_inputs: list, link_id, class_type: str, slot: int) -> str:
        """Find the input name for a given link. Try link-ID match first, then slot map."""
        # Method 1: Match by link ID in the node's input definitions
        for inp in node_inputs:
            if inp.get("link") == link_id:
                return inp.get("name", f"input_{slot}")
        
        # Method 2: Use the slot-based map
        slot_map = SLOT_MAPS.get(class_type, {})
        if slot in slot_map:
            return slot_map[slot]
        
        # Method 3: Try matching by slot index in input definitions
        if slot < len(node_inputs):
            return node_inputs[slot].get("name", f"input_{slot}")
        
        return f"input_{slot}"

    def _resolve_parent_input(self, sg_input_slot: int, sg_inputs: list,
                              wrapper_node: dict, top_links: dict):
        """Resolve a -10 (parent proxy) input to the actual source node.
        
        Returns (from_node_id, from_slot) or None.
        """
        if sg_input_slot >= len(sg_inputs):
            return None
        
        sg_input = sg_inputs[sg_input_slot]
        input_name = sg_input.get("name", "")

        # Find the wrapper's input that matches this name
        for winp in wrapper_node.get("inputs", []):
            if winp.get("name") == input_name or winp.get("label") == input_name:
                link_id = winp.get("link")
                if link_id is not None and link_id in top_links:
                    link_info = top_links[link_id]
                    return (link_info["from_node"], link_info["from_slot"])
        
        return None

    def _remap_subgraph_outputs(self, api: dict):
        """Remap all references to subgraph wrapper nodes → their actual inner output nodes.
        
        When node A references [wrapper_id, slot], we replace it with [inner_id, inner_slot]
        using the _sg_outputs map. This handles chained subgraphs too.
        """
        if not self._sg_outputs:
            return

        for nid, node_data in api.items():
            for inp_name, inp_val in list(node_data.get("inputs", {}).items()):
                if not (isinstance(inp_val, list) and len(inp_val) == 2):
                    continue
                try:
                    src_id = int(inp_val[0])
                except (ValueError, TypeError):
                    continue
                
                src_slot = inp_val[1]
                changed = False
                seen = set()  # Prevent infinite loops
                
                while src_id in self._sg_outputs and src_slot in self._sg_outputs[src_id]:
                    if src_id in seen:
                        break
                    seen.add(src_id)
                    src_id, src_slot = self._sg_outputs[src_id][src_slot]
                    changed = True
                
                if changed:
                    node_data["inputs"][inp_name] = [str(src_id), src_slot]

    def _prune_broken_refs(self, api: dict):
        """Remove nodes whose inputs reference non-existent nodes.
        
        This handles cases where bypassed subgraphs leave dangling references.
        We iteratively remove until stable (a chain of deps might all need removal).
        """
        changed = True
        while changed:
            changed = False
            to_remove = set()
            for nid, node_data in api.items():
                for inp_name, inp_val in node_data.get("inputs", {}).items():
                    if isinstance(inp_val, list) and len(inp_val) == 2:
                        ref_id = str(inp_val[0])
                        if ref_id not in api and ref_id != nid:
                            to_remove.add(nid)
                            break
            for nid in to_remove:
                logger.debug(f"Pruning node {nid} (broken reference)")
                del api[nid]
                changed = True

    def _build_node_inputs(self, node: dict, top_links: dict) -> dict:
        """Build inputs dict for a regular (non-subgraph) top-level node."""
        inputs = {}
        class_type = node.get("type", "")
        widgets = node.get("widgets_values", [])
        widget_map = WIDGET_MAPS.get(class_type, [])
        node_inputs = node.get("inputs", [])

        # Widget values → named inputs
        for i, wname in enumerate(widget_map):
            if i < len(widgets):
                inputs[wname] = widgets[i]

        # Link connections (overwrite widget values)
        for inp in node_inputs:
            link_id = inp.get("link")
            inp_name = inp.get("name", "")
            if link_id is not None and link_id in top_links:
                link_info = top_links[link_id]
                inputs[inp_name] = [str(link_info["from_node"]), link_info["from_slot"]]

        return inputs

    # ═══════════════════════════════════════════════════════════════
    # Parameter Injection
    # ═══════════════════════════════════════════════════════════════

    def _inject_prompt(self, workflow: dict, prompt: str):
        """Inject prompt into CLIPTextEncode or TextEncodeQwenImageEditPlus nodes."""
        if not prompt:
            return
        
        # Find all text-encoding nodes
        text_nodes = []
        for nid, node in workflow.items():
            ct = node.get("class_type", "")
            if ct == "CLIPTextEncode":
                text_nodes.append((nid, node, "text"))
            elif ct == "TextEncodeQwenImageEditPlus":
                text_nodes.append((nid, node, "prompt"))

        if not text_nodes:
            logger.warning("No text-encoding node found to inject prompt")
            return

        # Find which one is connected to KSampler positive input
        positive_node_id = None
        for nid, node in workflow.items():
            ct = node.get("class_type", "")
            if ct in ("KSampler", "KSamplerAdvanced", "SamplerCustomAdvanced"):
                pos = node.get("inputs", {}).get("positive")
                if isinstance(pos, list) and len(pos) >= 1:
                    positive_node_id = str(pos[0])
                    break
            elif ct == "CFGGuider":
                pos = node.get("inputs", {}).get("positive")
                if isinstance(pos, list) and len(pos) >= 1:
                    positive_node_id = str(pos[0])
                    break

        if positive_node_id and positive_node_id in workflow:
            target = workflow[positive_node_id]
            ct = target.get("class_type", "")
            if ct == "CLIPTextEncode":
                target["inputs"]["text"] = prompt
                logger.info(f"Injected prompt into CLIPTextEncode node {positive_node_id}")
                return
            elif ct == "TextEncodeQwenImageEditPlus":
                target["inputs"]["prompt"] = prompt
                logger.info(f"Injected prompt into TextEncodeQwenImageEditPlus node {positive_node_id}")
                return

        # Fallback: inject into the first text node
        nid, node, key = text_nodes[0]
        node["inputs"][key] = prompt
        logger.info(f"Injected prompt into first text node {nid} (fallback)")

    def _inject_negative_prompt(self, workflow: dict, neg_prompt: str):
        """Inject negative prompt if there's a node for it."""
        if not neg_prompt:
            return
        
        for nid, node in workflow.items():
            if node.get("class_type") in ("KSampler", "KSamplerAdvanced"):
                neg = node.get("inputs", {}).get("negative")
                if isinstance(neg, list) and len(neg) >= 1:
                    neg_node_id = str(neg[0])
                    if neg_node_id in workflow:
                        neg_node = workflow[neg_node_id]
                        if neg_node.get("class_type") == "CLIPTextEncode":
                            neg_node["inputs"]["text"] = neg_prompt
                            logger.info(f"Injected negative prompt into node {neg_node_id}")
                            return

    def _inject_seed(self, workflow: dict, seed: int):
        """Inject seed into KSampler or RandomNoise nodes."""
        for nid, node in workflow.items():
            class_type = node.get("class_type", "")
            inputs = node.get("inputs", {})
            
            if class_type in ("KSampler", "KSamplerAdvanced"):
                if "seed" in inputs:
                    inputs["seed"] = seed
                elif "noise_seed" in inputs:
                    inputs["noise_seed"] = seed
                logger.info(f"Injected seed {seed} into {class_type} node {nid}")
                return
            elif class_type == "RandomNoise":
                inputs["noise_seed"] = seed
                logger.info(f"Injected seed {seed} into RandomNoise node {nid}")
                return

    def _inject_dimensions(self, workflow: dict, width: int, height: int):
        """Inject width/height into latent image nodes."""
        for nid, node in workflow.items():
            class_type = node.get("class_type", "")
            if class_type in ("EmptyLatentImage", "EmptySD3LatentImage", "EmptyFlux2LatentImage"):
                node["inputs"]["width"] = width
                node["inputs"]["height"] = height
                logger.info(f"Injected dimensions {width}x{height} into {class_type}")
                return

    def _inject_images(self, workflow: dict, images: list, template_name: str = ""):
        """Inject uploaded image filenames into LoadImage nodes.
        
        Uses the WorkflowProfile to understand which LoadImage node expects what.
        Falls back to sequential assignment if no profile metadata available.
        """
        if not images:
            return

        # Find all LoadImage nodes in the workflow
        load_image_nodes = []
        for nid, node in workflow.items():
            if node.get("class_type") == "LoadImage":
                load_image_nodes.append((nid, node))

        if not load_image_nodes:
            logger.warning("No LoadImage nodes found in workflow to inject images into")
            return

        # Get the profile for purpose-aware injection
        profile = self.profiles.get(template_name)

        if profile and profile.image_inputs:
            # Purpose-aware injection: match images to inputs by profile order
            for i, (nid, node) in enumerate(load_image_nodes):
                if i < len(images):
                    node["inputs"]["image"] = images[i]
                    purpose = "unknown"
                    if i < len(profile.image_inputs):
                        purpose = profile.image_inputs[i].purpose
                    logger.info(f"Injected image '{images[i]}' into LoadImage node {nid} (purpose: {purpose})")
        else:
            # Sequential fallback: assign images to LoadImage nodes in order
            for i, (nid, node) in enumerate(load_image_nodes):
                if i < len(images):
                    node["inputs"]["image"] = images[i]
                    logger.info(f"Injected image '{images[i]}' into LoadImage node {nid}")

