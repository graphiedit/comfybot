"""
Error Recovery Agent — diagnoses ComfyUI errors and produces automatic fix strategies.

This module parses ComfyUI error messages, identifies the root cause,
and generates corrective actions that can be applied to the workflow
and generation plan for automatic retry.
"""
import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Tuple

logger = logging.getLogger(__name__)


class ErrorType(Enum):
    """Categories of ComfyUI errors."""
    VALUE_NOT_IN_LIST = "value_not_in_list"      # Wrong model/CLIP/VAE name
    META_TENSOR = "meta_tensor"                   # Model not fully loaded
    OUT_OF_MEMORY = "out_of_memory"               # GPU OOM
    VALIDATION_ERROR = "validation_error"         # Prompt validation failed
    EXECUTION_ERROR = "execution_error"           # Runtime execution error
    TIMEOUT = "timeout"                           # Generation timed out
    CONNECTION_ERROR = "connection_error"         # Can't reach ComfyUI
    UNKNOWN = "unknown"


@dataclass
class RecoveryStrategy:
    """A strategy for recovering from an error."""
    can_fix: bool = False
    error_type: ErrorType = ErrorType.UNKNOWN
    description: str = ""
    fixes: List[dict] = field(default_factory=list)  # List of {action, target, value}
    fallback_model: Optional[str] = None
    reduce_resolution: bool = False
    reduce_steps: bool = False
    switch_to_sdxl: bool = False
    retry_count: int = 0  # How many retries have been attempted already


class ErrorDiagnosticAgent:
    """
    Diagnoses ComfyUI errors and produces automatic fix strategies.
    
    Error handling is done in layers:
    1. Parse the error message to identify the error type
    2. Extract details (which node, what value, what's available)
    3. Generate a fix strategy based on the error type and context
    4. Apply the fix to the workflow and plan
    """

    def __init__(self, registry=None):
        self.registry = registry
        self._retry_history: dict = {}  # job_id -> list of attempted fixes

    def diagnose(self, error_msg: str, workflow: dict, plan=None, job_id: str = "") -> RecoveryStrategy:
        """
        Diagnose a ComfyUI error and return a recovery strategy.
        
        Args:
            error_msg: The error message from ComfyUI
            workflow: The workflow dict that failed
            plan: The GenerationPlan (if available)
            job_id: Job ID for tracking retry history
        """
        error_lower = error_msg.lower()
        
        # Track retries
        retries = self._retry_history.get(job_id, [])
        retry_count = len(retries)
        
        if retry_count >= 3:
            return RecoveryStrategy(
                can_fix=False,
                error_type=ErrorType.UNKNOWN,
                description="Maximum retries exceeded. Giving up.",
            )

        strategy = None

        # --- Pattern matching ---
        
        # 1. Value not in list (wrong model/CLIP/VAE name)
        if "not in" in error_lower and ("value" in error_lower or "clip_name" in error_lower or "vae_name" in error_lower or "unet_name" in error_lower):
            strategy = self._diagnose_value_not_in_list(error_msg, workflow, plan, retry_count)
        
        # 2. Meta tensor error (model loading issue)
        elif "meta tensor" in error_lower or "cannot copy out of meta" in error_lower:
            strategy = self._diagnose_meta_tensor(error_msg, workflow, plan, retry_count)
        
        # 3. Out of memory
        elif "out of memory" in error_lower or "cuda out of memory" in error_lower or "oom" in error_lower:
            strategy = self._diagnose_oom(error_msg, workflow, plan, retry_count)
        
        # 4. Prompt validation failed
        elif "prompt_outputs_failed_validation" in error_lower or "failed validation" in error_lower:
            strategy = self._diagnose_validation_error(error_msg, workflow, plan, retry_count)
        
        # 5. Execution error (generic)
        elif "execution_error" in error_lower or "exception_message" in error_lower:
            strategy = self._diagnose_execution_error(error_msg, workflow, plan, retry_count)
        
        # 6. Timeout
        elif "timed out" in error_lower or "timeout" in error_lower:
            strategy = RecoveryStrategy(
                can_fix=True,
                error_type=ErrorType.TIMEOUT,
                description="Generation timed out. Reducing steps and resolution.",
                reduce_resolution=True,
                reduce_steps=True,
                retry_count=retry_count,
            )
        
        # 7. Connection error
        elif "connection" in error_lower or "cannot connect" in error_lower or "refused" in error_lower:
            strategy = RecoveryStrategy(
                can_fix=False,
                error_type=ErrorType.CONNECTION_ERROR,
                description="Cannot connect to ComfyUI server.",
            )
        
        else:
            # Unknown error
            strategy = RecoveryStrategy(
                can_fix=retry_count < 1,  # Allow one generic retry
                error_type=ErrorType.UNKNOWN,
                description=f"Unknown error: {error_msg[:200]}",
                switch_to_sdxl=retry_count == 0,  # Try SDXL as last resort
                retry_count=retry_count,
            )
            
        # Feature 10: Strict Template Adherence
        if plan and getattr(plan, 'workflow_template', None) and strategy.switch_to_sdxl:
            logger.info(f"Disabling SDXL fallback because custom template '{plan.workflow_template}' is active.")
            strategy.switch_to_sdxl = False
            # If we were relying on SDXL to "fix" it, we can't fix it anymore
            if not strategy.fixes and not strategy.reduce_resolution and not strategy.reduce_steps and not strategy.fallback_model:
                strategy.can_fix = False
            strategy.description += " (Template mode: SDXL fallback disabled)"
            
        return strategy

    def _diagnose_value_not_in_list(self, error_msg, workflow, plan, retry_count) -> RecoveryStrategy:
        """Handle 'value X not in [Y, Z]' errors — wrong model/CLIP/VAE name."""
        strategy = RecoveryStrategy(
            can_fix=True,
            error_type=ErrorType.VALUE_NOT_IN_LIST,
            retry_count=retry_count,
        )
        
        # Try to parse the error to find what's wrong and what's available
        # Pattern: "clip_name1": "bad_name" not in [good1, good2, good3]
        # Or: value not in list: {"details": "...", "extra_info": {...}}
        try:
            # Try to extract the field name, bad value, and available values
            match = re.search(
                r'[\'"]?(\w+)[\'"]?:\s*[\'"]?([^"\'\]]+)[\'"]?\s*not\s+in\s*\[([^\]]+)\]',
                error_msg
            )
            if match:
                field_name = match.group(1)
                bad_value = match.group(2)
                available_raw = match.group(3)
                available = [v.strip().strip("'\"") for v in available_raw.split(",")]
                
                # Find the best replacement
                best = self._find_closest_match(bad_value, available)
                
                strategy.description = (
                    f"Invalid {field_name}: '{bad_value}'. "
                    f"Auto-switching to '{best}'."
                )
                strategy.fixes.append({
                    "action": "replace_value",
                    "field": field_name,
                    "old_value": bad_value,
                    "new_value": best,
                })
                return strategy
            
            # Broader pattern for JSON-style errors
            if "details" in error_msg:
                try:
                    # Try to parse embedded JSON
                    json_start = error_msg.find("{")
                    json_end = error_msg.rfind("}") + 1
                    if json_start >= 0 and json_end > json_start:
                        error_data = json.loads(error_msg[json_start:json_end])
                        node_errors = error_data.get("node_errors", {})
                        for node_id, node_err in node_errors.items():
                            for err in node_err.get("errors", []):
                                if "details" in err:
                                    details = err["details"]
                                    extra = err.get("extra_info", {})
                                    
                                    # Extract field name and available values
                                    field_name = extra.get("input_name", "unknown")
                                    available = extra.get("valid_values", [])
                                    
                                    if available:
                                        best = available[0]
                                        strategy.description = (
                                            f"Invalid value for '{field_name}'. "
                                            f"Auto-switching to '{best}'."
                                        )
                                        strategy.fixes.append({
                                            "action": "replace_node_input",
                                            "node_id": node_id,
                                            "field": field_name,
                                            "new_value": best,
                                        })
                except (json.JSONDecodeError, KeyError):
                    pass
        except Exception as e:
            logger.warning(f"Error parsing value_not_in_list: {e}")
        
        # If we couldn't parse specifics, try a broad fallback
        if not strategy.fixes:
            strategy.description = "Invalid model/CLIP/VAE value. Falling back to SDXL."
            strategy.switch_to_sdxl = True
        
        return strategy

    def _diagnose_meta_tensor(self, error_msg, workflow, plan, retry_count) -> RecoveryStrategy:
        """Handle 'Cannot copy out of meta tensor' — model loading failure."""
        strategy = RecoveryStrategy(
            can_fix=True,
            error_type=ErrorType.META_TENSOR,
            retry_count=retry_count,
        )
        
        if retry_count == 0:
            # First attempt: try fp8_e4m3fn weight dtype
            strategy.description = (
                "Model failed to load (meta tensor error). "
                "Retrying with fp8_e4m3fn weight dtype."
            )
            strategy.fixes.append({
                "action": "change_weight_dtype",
                "new_value": "fp8_e4m3fn",
            })
        elif retry_count == 1:
            # Second attempt: try a different model
            strategy.description = (
                "Model still failing. Switching to a different model."
            )
            # Find an alternative model
            if self.registry:
                alternatives = self._get_alternative_models(workflow, plan)
                if alternatives:
                    strategy.fallback_model = alternatives[0]
                    strategy.description += f" Using '{alternatives[0]}'."
                else:
                    strategy.switch_to_sdxl = True
                    strategy.description += " Falling back to SDXL."
            else:
                strategy.switch_to_sdxl = True
        else:
            # Third attempt: SDXL
            strategy.switch_to_sdxl = True
            strategy.description = "Persistent model loading failure. Falling back to SDXL."
        
        return strategy

    def _diagnose_oom(self, error_msg, workflow, plan, retry_count) -> RecoveryStrategy:
        """Handle out-of-memory errors."""
        strategy = RecoveryStrategy(
            can_fix=True,
            error_type=ErrorType.OUT_OF_MEMORY,
            retry_count=retry_count,
        )
        
        if retry_count == 0:
            strategy.description = "GPU out of memory. Reducing resolution to 768×768."
            strategy.reduce_resolution = True
        elif retry_count == 1:
            strategy.description = "Still OOM. Reducing steps and trying smaller model."
            strategy.reduce_resolution = True
            strategy.reduce_steps = True
            strategy.switch_to_sdxl = True
        else:
            strategy.can_fix = False
            strategy.description = "Persistent OOM. Cannot generate with current hardware."
        
        return strategy

    def _diagnose_validation_error(self, error_msg, workflow, plan, retry_count) -> RecoveryStrategy:
        """Handle prompt_outputs_failed_validation errors."""
        strategy = RecoveryStrategy(
            can_fix=True,
            error_type=ErrorType.VALIDATION_ERROR,
            retry_count=retry_count,
        )
        
        # Try to parse the structured error
        try:
            json_start = error_msg.find("{")
            json_end = error_msg.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                error_data = json.loads(error_msg[json_start:json_end])
                node_errors = error_data.get("node_errors", {})
                
                for node_id, node_err in node_errors.items():
                    class_type = node_err.get("class_type", "")
                    errors = node_err.get("errors", [])
                    
                    for err in errors:
                        if err.get("type") == "value_not_in_list":
                            details = err.get("details", "")
                            extra = err.get("extra_info", {})
                            input_name = extra.get("input_name", "")
                            
                            # Extract available values from the details string
                            match = re.search(r"not in \[([^\]]+)\]", details)
                            if match:
                                available = [v.strip().strip("'\"") for v in match.group(1).split(",")]
                                if available:
                                    strategy.fixes.append({
                                        "action": "replace_node_input",
                                        "node_id": node_id,
                                        "field": input_name,
                                        "new_value": available[0],
                                    })
                                    strategy.description = (
                                        f"Validation failed: invalid '{input_name}' in {class_type}. "
                                        f"Auto-switching to '{available[0]}'."
                                    )
        except Exception as e:
            logger.warning(f"Error parsing validation error: {e}")
        
        if not strategy.fixes:
            # Generic fallback: switch to SDXL
            strategy.switch_to_sdxl = True
            strategy.description = "Workflow validation failed. Falling back to SDXL."
        
        return strategy

    def _diagnose_execution_error(self, error_msg, workflow, plan, retry_count) -> RecoveryStrategy:
        """Handle generic execution errors."""
        strategy = RecoveryStrategy(
            can_fix=retry_count < 2,
            error_type=ErrorType.EXECUTION_ERROR,
            retry_count=retry_count,
        )
        
        if retry_count == 0:
            strategy.description = "Execution error. Retrying with different parameters."
            strategy.reduce_steps = True
        else:
            strategy.switch_to_sdxl = True
            strategy.description = "Persistent execution error. Falling back to SDXL."
        
        return strategy

    def apply_fix(self, strategy: RecoveryStrategy, workflow: dict, plan) -> Tuple[dict, object]:
        """
        Apply a recovery strategy to the workflow and plan.
        
        Returns: (fixed_workflow, fixed_plan)
        """
        import copy
        workflow = copy.deepcopy(workflow)
        plan = copy.deepcopy(plan) if plan else plan
        
        # Apply specific fixes
        for fix in strategy.fixes:
            action = fix.get("action", "")
            
            if action == "replace_value":
                # Search the entire workflow for the old value and replace it
                field = fix["field"]
                old_val = fix["old_value"]
                new_val = fix["new_value"]
                workflow = self._replace_in_workflow(workflow, field, old_val, new_val)
                # Also update plan if possible
                if plan and hasattr(plan, field):
                    setattr(plan, field, new_val)
                    logger.info(f"Recovery: updated plan.{field} → '{new_val}'")
                logger.info(f"Recovery: replaced {field} '{old_val}' → '{new_val}'")
            
            elif action == "replace_node_input":
                node_id = fix["node_id"]
                field = fix["field"]
                new_val = fix["new_value"]
                # Update plan if field exists
                if plan and hasattr(plan, field):
                    setattr(plan, field, new_val)
                    logger.info(f"Recovery: updated plan.{field} → '{new_val}'")
                
                if node_id in workflow and "inputs" in workflow[node_id]:
                    workflow[node_id]["inputs"][field] = new_val
                    logger.info(f"Recovery: set {node_id}.{field} = '{new_val}'")
                else:
                    # Search by string key match
                    for nid, node in workflow.items():
                        if isinstance(node, dict) and "inputs" in node:
                            if field in node["inputs"]:
                                node["inputs"][field] = new_val
                                logger.info(f"Recovery: set {nid}.{field} = '{new_val}'")
                                break
            
            elif action == "change_weight_dtype":
                new_dtype = fix["new_value"]
                if plan:
                    plan.weight_dtype = new_dtype
                for nid, node in workflow.items():
                    if isinstance(node, dict) and node.get("class_type") == "UNETLoader":
                        node["inputs"]["weight_dtype"] = new_dtype
                        logger.info(f"Recovery: UNETLoader weight_dtype → '{new_dtype}'")
        
        # Apply resolution reduction
        if strategy.reduce_resolution and plan:
            plan.width = min(plan.width, 768)
            plan.height = min(plan.height, 768)
            # Also update empty_latent in workflow
            for nid, node in workflow.items():
                if isinstance(node, dict) and node.get("class_type") in ("EmptyLatentImage", "EmptySD3LatentImage"):
                    node["inputs"]["width"] = plan.width
                    node["inputs"]["height"] = plan.height
            logger.info(f"Recovery: resolution reduced to {plan.width}×{plan.height}")
        
        # Apply step reduction
        if strategy.reduce_steps and plan:
            plan.steps = max(plan.steps // 2, 4)
            for nid, node in workflow.items():
                if isinstance(node, dict) and node.get("class_type") == "KSampler":
                    node["inputs"]["steps"] = plan.steps
            logger.info(f"Recovery: steps reduced to {plan.steps}")
        
        # Switch to fallback model
        if strategy.fallback_model:
            new_model = strategy.fallback_model
            for nid, node in workflow.items():
                if isinstance(node, dict) and node.get("class_type") == "UNETLoader":
                    node["inputs"]["unet_name"] = new_model
                    logger.info(f"Recovery: switched UNET to '{new_model}'")
        
        # Switch to SDXL as last resort
        if strategy.switch_to_sdxl and plan:
            plan.model_arch = "sdxl"
            plan.checkpoint = "DreamShaperXL_Lightning.safetensors"
            plan.cfg = 2.0
            plan.steps = min(plan.steps, 8)
            plan.sampler = "dpmpp_sde"
            plan.scheduler = "karras"
            # Workflow will be rebuilt by the caller
            workflow = None  # Signal to caller to rebuild
            logger.info("Recovery: falling back to SDXL (DreamShaperXL_Lightning)")
        
        return workflow, plan

    def record_attempt(self, job_id: str, strategy: RecoveryStrategy):
        """Record a recovery attempt for a job."""
        if job_id not in self._retry_history:
            self._retry_history[job_id] = []
        self._retry_history[job_id].append({
            "error_type": strategy.error_type.value,
            "description": strategy.description,
            "fixes": strategy.fixes,
        })

    def clear_history(self, job_id: str):
        """Clear retry history for a completed/abandoned job."""
        self._retry_history.pop(job_id, None)

    # --- Helpers ---
    
    def _find_closest_match(self, target: str, available: list) -> str:
        """Find the closest matching value from a list of available values."""
        if not available:
            return target
        
        target_lower = target.lower()
        
        # Exact match
        for v in available:
            if v.lower() == target_lower:
                return v
        
        # Substring match
        for v in available:
            if target_lower in v.lower() or v.lower() in target_lower:
                return v
        
        # Just return the first available
        return available[0]

    def _get_alternative_models(self, workflow, plan) -> list:
        """Get alternative models from the registry."""
        if not self.registry:
            return []
        
        current = ""
        if plan:
            current = plan.checkpoint
        
        alternatives = []
        catalog = getattr(self.registry, "_model_catalog", {})
        
        for dm in catalog.get("diffusion_models", []):
            if dm["filename"] != current:
                alternatives.append(dm["filename"])
        
        for ckpt in catalog.get("checkpoints", []):
            if ckpt["filename"] != current:
                alternatives.append(ckpt["filename"])
        
        return alternatives

    def _replace_in_workflow(self, workflow: dict, field: str, old_val: str, new_val: str) -> dict:
        """Recursively replace a value in the workflow dict."""
        for nid, node in workflow.items():
            if isinstance(node, dict) and "inputs" in node:
                inputs = node["inputs"]
                if field in inputs and inputs[field] == old_val:
                    inputs[field] = new_val
        return workflow
