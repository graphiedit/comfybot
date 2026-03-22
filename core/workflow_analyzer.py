"""
Workflow Analyzer — auto-introspects ComfyUI workflow JSON to extract structured metadata.

This is the brain of the Smart Workflow Hub. When you drop a workflow JSON
into data/workflows/, this module automatically figures out:
  - How many image inputs does it need?
  - What kind of images? (edit source, style reference, pose, etc.)
  - What prompt nodes exist?
  - What parameters are tunable?
  - What model architecture does it use?
  - What capabilities does it have? (text2img, img2img, controlnet, upscale, etc.)

The AI agent uses this metadata to make smart decisions about which workflow
to select and what information to ask the user for.
"""
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Any, Optional, Set

logger = logging.getLogger(__name__)


# Node types that represent image inputs from the user
IMAGE_INPUT_TYPES = {"LoadImage"}

# Node types that accept text prompts
PROMPT_NODE_TYPES = {"CLIPTextEncode", "TextEncodeQwenImageEditPlus", "PrimitiveStringMultiline"}

# Node types that are samplers (KSampler family)
SAMPLER_TYPES = {"KSampler", "KSamplerAdvanced", "SamplerCustomAdvanced", "SamplerCustom"}

# Node types for latent image creation (dimensions control)
LATENT_TYPES = {"EmptyLatentImage", "EmptySD3LatentImage", "EmptyFlux2LatentImage"}

# Node types for model loading
CHECKPOINT_TYPES = {"CheckpointLoaderSimple"}
UNET_TYPES = {"UNETLoader"}
CLIP_TYPES = {"CLIPLoader", "DualCLIPLoader"}
VAE_TYPES = {"VAELoader"}

# ControlNet-related nodes
CONTROLNET_TYPES = {
    "ControlNetLoader", "ControlNetApplyAdvanced", "ControlNetApply",
    "OpenposePreprocessor", "DWPreprocessor", "Canny", "CannyEdgePreprocessor",
    "LineArtPreprocessor", "MiDaS-DepthMapPreprocessor",
    "QwenImageDiffsynthControlnet",
}

# IP-Adapter nodes
IPADAPTER_TYPES = {
    "IPAdapterModelLoader", "IPAdapterAdvanced", "IPAdapterApply",
    "CLIPVisionLoader",
}

# Upscale nodes
UPSCALE_TYPES = {
    "UpscaleModelLoader", "ImageUpscaleWithModel", "LatentUpscale",
    "ImageScale", "ImageScaleToTotalPixels", "ImageScaleToMaxDimension",
}

# Face fix / detailer nodes
FACE_FIX_TYPES = {
    "FaceDetailer", "UltralyticsDetectorProvider",
}

# Guidance / flux-specific
GUIDANCE_TYPES = {"FluxGuidance", "ModelSamplingFlux", "CFGGuider", "CFGNorm"}

# LoRA types
LORA_TYPES = {"LoraLoader", "LoraLoaderModelOnly"}

# Nodes that imply image editing (take an image and transform it)
IMAGE_EDIT_MARKERS = {
    "VAEEncode",  # encoding an input image implies img2img / edit
    "TextEncodeQwenImageEditPlus",  # explicit edit encoder
    "FluxKontextImageScale",  # Flux Kontext editing
    "ReferenceLatent",
}


@dataclass
class ImageInputInfo:
    """Describes a single image input slot in a workflow."""
    node_id: str
    node_type: str = "LoadImage"
    purpose: str = "unknown"  # edit_source, style_reference, pose_reference, controlnet_input, generic
    connected_to: List[str] = field(default_factory=list)  # What downstream nodes it feeds
    required: bool = True


@dataclass
class WorkflowProfile:
    """Complete structured metadata for a workflow template."""
    name: str = ""
    description: str = ""  # From the .txt file

    # Image inputs
    image_inputs: List[ImageInputInfo] = field(default_factory=list)
    requires_image: bool = False
    min_images: int = 0
    max_images: int = 0

    # Prompt info
    has_positive_prompt: bool = False
    has_negative_prompt: bool = False
    prompt_node_ids: List[str] = field(default_factory=list)

    # Model architecture
    architecture: str = "unknown"  # sdxl, flux, hunyuan, unknown
    model_nodes: List[str] = field(default_factory=list)

    # Capabilities (tags)
    capabilities: Set[str] = field(default_factory=set)
    # Possible: text_to_image, image_edit, style_transfer, controlnet,
    #           ipadapter, upscale, face_fix, lora, guidance

    # Tunable parameters found
    tunable_params: Dict[str, Any] = field(default_factory=dict)
    # e.g. {"steps": 30, "cfg": 7.0, "width": 1024, "height": 1024, ...}

    # Node type counts (for diagnostics)
    node_types: Dict[str, int] = field(default_factory=dict)
    total_nodes: int = 0

    def to_llm_summary(self) -> str:
        """Generate a concise summary for the LLM to reason about."""
        parts = [f"- {self.name}: {self.description}"]

        caps = ", ".join(sorted(self.capabilities)) if self.capabilities else "general"
        parts.append(f"  Capabilities: {caps}")

        if self.requires_image:
            img_desc = []
            for img in self.image_inputs:
                img_desc.append(f"{img.purpose}")
            parts.append(f"  Requires {self.min_images} image(s): {', '.join(img_desc)}")
        else:
            parts.append("  No image input needed (text-only)")

        parts.append(f"  Architecture: {self.architecture}")

        if self.tunable_params:
            tunable = ", ".join(f"{k}={v}" for k, v in list(self.tunable_params.items())[:6])
            parts.append(f"  Tunable: {tunable}")

        return "\n".join(parts)

    def to_dict(self) -> dict:
        """Serialize to dict for JSON storage."""
        return {
            "name": self.name,
            "description": self.description,
            "requires_image": self.requires_image,
            "min_images": self.min_images,
            "max_images": self.max_images,
            "image_inputs": [
                {"node_id": ii.node_id, "purpose": ii.purpose, "required": ii.required}
                for ii in self.image_inputs
            ],
            "architecture": self.architecture,
            "capabilities": sorted(self.capabilities),
            "tunable_params": self.tunable_params,
            "total_nodes": self.total_nodes,
        }


class WorkflowAnalyzer:
    """Analyzes ComfyUI workflow JSON files to extract structured metadata."""

    def analyze(self, name: str, workflow_data: dict, description: str = "") -> WorkflowProfile:
        """
        Analyze a workflow JSON and return a WorkflowProfile.

        Supports both API format (flat dict of nodes) and graph-editor format
        (nodes[] + links[]). For graph-editor format, we analyze the raw nodes
        directly without converting to API format first.
        """
        profile = WorkflowProfile(name=name, description=description)

        if self._is_api_format(workflow_data):
            self._analyze_api_format(workflow_data, profile)
        elif self._is_graph_format(workflow_data):
            self._analyze_graph_format(workflow_data, profile)
        else:
            logger.warning(f"Unknown workflow format for {name}")
            return profile

        # Post-processing: infer capabilities and image purposes
        self._infer_capabilities(profile)
        self._infer_image_purposes(profile, workflow_data)

        logger.info(
            f"Analyzed workflow '{name}': {profile.total_nodes} nodes, "
            f"caps={profile.capabilities}, images_needed={profile.min_images}"
        )
        return profile

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
    # API Format Analysis
    # ═══════════════════════════════════════════════════════════════

    def _analyze_api_format(self, data: dict, profile: WorkflowProfile):
        """Analyze a workflow already in API format."""
        profile.total_nodes = len(data)

        # Build connection map: which node feeds into which
        connections = self._build_api_connections(data)

        for node_id, node in data.items():
            class_type = node.get("class_type", "")
            inputs = node.get("inputs", {})

            # Count node types
            profile.node_types[class_type] = profile.node_types.get(class_type, 0) + 1

            self._process_node(profile, node_id, class_type, inputs, connections)

    def _build_api_connections(self, data: dict) -> Dict[str, List[str]]:
        """Build forward connection map: source_id -> [target_ids]."""
        connections = {}
        for node_id, node in data.items():
            for inp_name, inp_val in node.get("inputs", {}).items():
                if isinstance(inp_val, list) and len(inp_val) == 2:
                    src_id = str(inp_val[0])
                    if src_id not in connections:
                        connections[src_id] = []
                    connections[src_id].append(node_id)
        return connections

    # ═══════════════════════════════════════════════════════════════
    # Graph Format Analysis (raw ComfyUI export)
    # ═══════════════════════════════════════════════════════════════

    def _analyze_graph_format(self, data: dict, profile: WorkflowProfile):
        """Analyze a workflow in graph-editor format (nodes[] + links[])."""
        nodes = data.get("nodes", [])
        links = data.get("links", [])
        subgraphs = {sg["id"]: sg for sg in data.get("definitions", {}).get("subgraphs", [])}

        profile.total_nodes = len(nodes)

        # Build connection map from links
        # connections[target_node_id] = [(source_node_id, link info)]
        forward_connections = {}  # source_id -> [target_ids]
        for link in links:
            if isinstance(link, list) and len(link) >= 6:
                _link_id, from_node, _from_slot, to_node, _to_slot, _link_type = link[:6]
                if from_node not in forward_connections:
                    forward_connections[from_node] = []
                forward_connections[from_node].append(to_node)

        # Process top-level nodes
        for node in nodes:
            if node.get("mode", 0) >= 2:  # Skip bypassed/muted
                continue

            node_type = node.get("type", "")
            node_id = str(node.get("id", ""))

            # Count node types
            profile.node_types[node_type] = profile.node_types.get(node_type, 0) + 1

            # Build a pseudo-inputs dict from widgets_values
            from core.workflow_manager import WIDGET_MAPS
            inputs = {}
            widget_map = WIDGET_MAPS.get(node_type, [])
            widgets = node.get("widgets_values", [])
            for i, wname in enumerate(widget_map):
                if i < len(widgets):
                    inputs[wname] = widgets[i]

            self._process_node(profile, node_id, node_type, inputs, forward_connections)

        # Also scan subgraph inner nodes
        for sg_id, sg in subgraphs.items():
            for inode in sg.get("nodes", []):
                itype = inode.get("type", "")
                iid = str(inode.get("id", ""))
                profile.node_types[itype] = profile.node_types.get(itype, 0) + 1

                inputs = {}
                widget_map = WIDGET_MAPS.get(itype, [])
                widgets = inode.get("widgets_values", [])
                for i, wname in enumerate(widget_map):
                    if i < len(widgets):
                        inputs[wname] = widgets[i]

                # Use empty connections for subgraph nodes (simplified)
                self._process_node(profile, iid, itype, inputs, {})

    # ═══════════════════════════════════════════════════════════════
    # Shared Node Processing
    # ═══════════════════════════════════════════════════════════════

    def _process_node(self, profile: WorkflowProfile, node_id: str,
                      class_type: str, inputs: dict, connections: dict):
        """Process a single node and update the profile."""

        # Image inputs
        if class_type in IMAGE_INPUT_TYPES:
            downstream = connections.get(node_id, []) if isinstance(node_id, int) else connections.get(node_id, [])
            # Also check int version
            try:
                int_id = int(node_id)
                downstream = downstream or connections.get(int_id, [])
            except (ValueError, TypeError):
                pass

            img_info = ImageInputInfo(
                node_id=node_id,
                node_type=class_type,
                connected_to=[str(d) for d in downstream],
            )
            profile.image_inputs.append(img_info)

        # Prompt nodes
        if class_type in PROMPT_NODE_TYPES:
            profile.prompt_node_ids.append(node_id)
            profile.has_positive_prompt = True

        # Model architecture detection
        if class_type in CHECKPOINT_TYPES:
            profile.architecture = "sdxl"
            profile.model_nodes.append(node_id)
            if "ckpt_name" in inputs:
                profile.tunable_params["checkpoint"] = inputs["ckpt_name"]

        if class_type in UNET_TYPES:
            profile.model_nodes.append(node_id)
            if "unet_name" in inputs:
                profile.tunable_params["unet_name"] = inputs["unet_name"]
                # Infer architecture from model name
                unet_name = str(inputs["unet_name"]).lower()
                if "flux" in unet_name or "z_image" in unet_name:
                    profile.architecture = "flux"
                elif "hunyuan" in unet_name:
                    profile.architecture = "hunyuan"
                elif "qwen" in unet_name:
                    profile.architecture = "flux"  # Qwen models use flux pipeline
                else:
                    profile.architecture = "flux"  # UNET loader = likely flux

        if class_type in CLIP_TYPES:
            if "clip_name1" in inputs:
                profile.tunable_params["clip_name1"] = inputs["clip_name1"]
            if "clip_name2" in inputs:
                profile.tunable_params["clip_name2"] = inputs["clip_name2"]

        # Sampler parameters
        if class_type in SAMPLER_TYPES:
            for param in ["steps", "cfg", "sampler_name", "scheduler", "denoise"]:
                if param in inputs and inputs[param] is not None:
                    profile.tunable_params[param] = inputs[param]
            for seed_key in ["seed", "noise_seed"]:
                if seed_key in inputs:
                    profile.tunable_params["seed"] = inputs[seed_key]

        # Latent dimensions
        if class_type in LATENT_TYPES:
            if "width" in inputs:
                profile.tunable_params["width"] = inputs["width"]
            if "height" in inputs:
                profile.tunable_params["height"] = inputs["height"]

        # Guidance
        if class_type in GUIDANCE_TYPES:
            if "guidance" in inputs:
                profile.tunable_params["guidance"] = inputs["guidance"]

    # ═══════════════════════════════════════════════════════════════
    # Capability Inference
    # ═══════════════════════════════════════════════════════════════

    def _infer_capabilities(self, profile: WorkflowProfile):
        """Infer workflow capabilities from the node types present."""
        types_present = set(profile.node_types.keys())

        # ControlNet
        if types_present & CONTROLNET_TYPES:
            profile.capabilities.add("controlnet")

        # IP-Adapter
        if types_present & IPADAPTER_TYPES:
            profile.capabilities.add("ipadapter")
            profile.capabilities.add("style_transfer")

        # Upscale
        if types_present & UPSCALE_TYPES:
            profile.capabilities.add("upscale")

        # Face fix
        if types_present & FACE_FIX_TYPES:
            profile.capabilities.add("face_fix")

        # LoRA
        if types_present & LORA_TYPES:
            profile.capabilities.add("lora")

        # Guidance (flux-specific)
        if types_present & GUIDANCE_TYPES:
            profile.capabilities.add("guidance")

        # Image editing
        if types_present & IMAGE_EDIT_MARKERS:
            profile.capabilities.add("image_edit")

        # Image inputs
        if profile.image_inputs:
            profile.requires_image = True
            profile.min_images = len(profile.image_inputs)
            profile.max_images = len(profile.image_inputs)
            profile.capabilities.add("image_input")
        else:
            profile.capabilities.add("text_to_image")

        # Determine negative prompt
        # Simple heuristic: if there are 2+ prompt nodes, one is likely negative
        if len(profile.prompt_node_ids) >= 2:
            profile.has_negative_prompt = True

    def _infer_image_purposes(self, profile: WorkflowProfile, workflow_data: dict):
        """
        Infer the purpose of each image input based on what it connects to.

        Heuristics:
        - If connected to VAEEncode → edit_source (img2img)
        - If connected to IP-Adapter → style_reference
        - If connected to ControlNet preprocessor → pose_reference / controlnet_input
        - If connected to TextEncodeQwenImageEditPlus → edit_source
        - Otherwise → generic
        """
        if not profile.image_inputs:
            return

        # For graph format, we need to trace connections
        # For API format, we can check downstream nodes
        if self._is_api_format(workflow_data):
            self._infer_purposes_api(profile, workflow_data)
        elif self._is_graph_format(workflow_data):
            self._infer_purposes_graph(profile, workflow_data)

        # Fallback: check description text for hints
        desc_lower = profile.description.lower()
        for img_info in profile.image_inputs:
            if img_info.purpose == "unknown":
                if "edit" in desc_lower or "restyle" in desc_lower or "modify" in desc_lower:
                    img_info.purpose = "edit_source"
                elif "style" in desc_lower or "reference" in desc_lower:
                    img_info.purpose = "style_reference"
                elif "pose" in desc_lower or "controlnet" in desc_lower:
                    img_info.purpose = "pose_reference"
                else:
                    img_info.purpose = "generic"

    def _infer_purposes_api(self, profile: WorkflowProfile, data: dict):
        """Infer image purposes in API format by checking downstream connections."""
        # Build reverse map: for each LoadImage node, find what it connects to
        for img_info in profile.image_inputs:
            downstream_types = set()
            for node_id, node in data.items():
                for inp_name, inp_val in node.get("inputs", {}).items():
                    if isinstance(inp_val, list) and len(inp_val) == 2:
                        if str(inp_val[0]) == str(img_info.node_id):
                            downstream_types.add(node.get("class_type", ""))

            img_info.purpose = self._purpose_from_downstream(downstream_types)

    def _infer_purposes_graph(self, profile: WorkflowProfile, data: dict):
        """Infer image purposes in graph format by tracing links."""
        links = data.get("links", [])
        nodes_by_id = {n["id"]: n for n in data.get("nodes", [])}

        for img_info in profile.image_inputs:
            try:
                img_id = int(img_info.node_id)
            except (ValueError, TypeError):
                continue

            # Find all links originating from this LoadImage node
            downstream_types = set()
            for link in links:
                if isinstance(link, list) and len(link) >= 6:
                    from_node = link[1]
                    to_node = link[3]
                    if from_node == img_id and to_node in nodes_by_id:
                        downstream_types.add(nodes_by_id[to_node].get("type", ""))

            img_info.purpose = self._purpose_from_downstream(downstream_types)

    def _purpose_from_downstream(self, downstream_types: set) -> str:
        """Determine image purpose from the set of downstream node types."""
        if downstream_types & {"VAEEncode"}:
            return "edit_source"
        if downstream_types & {"TextEncodeQwenImageEditPlus"}:
            return "edit_source"
        if downstream_types & {"FluxKontextImageScale"}:
            return "edit_source"
        if downstream_types & IPADAPTER_TYPES:
            return "style_reference"
        if downstream_types & CONTROLNET_TYPES:
            return "pose_reference"
        if downstream_types & {"OpenposePreprocessor", "DWPreprocessor"}:
            return "pose_reference"
        if downstream_types & {"Canny", "CannyEdgePreprocessor"}:
            return "controlnet_input"
        return "unknown"
