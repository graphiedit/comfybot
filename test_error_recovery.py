"""Test script for the Error Diagnostic Agent."""
import sys

sys.path.insert(0, ".")

from core.error_recovery import ErrorDiagnosticAgent, ErrorType, RecoveryStrategy

def test_error_agent():
    print("="*50)
    print("TEST: Error Diagnostic Agent")
    print("="*50)
    
    agent = ErrorDiagnosticAgent()
    
    # 1. Test "value_not_in_list" error parsing
    error1 = """
    ComfyUI rejected workflow: {"error": {"type": "prompt_outputs_failed_validation", 
    "node_errors": {"clip_loader": {"errors": [{"type": "value_not_in_list", 
    "message": "Value not in list", "details": "'clip_name1': 't5xxl_fp8_e4m3fn.safetensors' not in [mistral_3_small_flux2_bf16.safetensors, qwen_2.5_vl_7b_fp8_scaled.safetensors, qwen_3_4b.safetensors]", 
    "extra_info": {"input_name": "clip_name1", "valid_values": []}}]}}}}
    """
    
    # Actually, the parsing in _diagnose_value_not_in_list depends on regex for "not in [...]"
    strategy1 = agent.diagnose(error1, {}, None)
    
    print("\n--- Testing VALUE_NOT_IN_LIST ---")
    print(f"Type: {strategy1.error_type}")
    print(f"Fixes: {strategy1.fixes}")
    
    # Our regex should capture the available list
    assert strategy1.error_type == ErrorType.VALUE_NOT_IN_LIST
    assert len(strategy1.fixes) > 0
    assert strategy1.fixes[0]["action"] == "replace_value"
    assert "t5xxl_fp8_e4m3fn.safetensors" in strategy1.fixes[0]["old_value"]
    # It should pick one of the available
    assert "qwen" in strategy1.fixes[0]["new_value"] or "mistral" in strategy1.fixes[0]["new_value"]
    
    # 2. Test "meta tensor" error
    error2 = "Cannot copy out of meta tensor; no data!"
    strategy2 = agent.diagnose(error2, {}, None)
    
    print("\n--- Testing META_TENSOR (Attempt 1) ---")
    print(f"Type: {strategy2.error_type}")
    print(f"Fixes: {strategy2.fixes}")
    assert strategy2.error_type == ErrorType.META_TENSOR
    assert strategy2.fixes[0]["action"] == "change_weight_dtype"
    assert strategy2.fixes[0]["new_value"] == "fp16"
    
    # 3. Test OOM error
    error3 = "RuntimeError: CUDA out of memory. Tried to allocate..."
    strategy3 = agent.diagnose(error3, {}, None)
    print("\n--- Testing OOM ---")
    print(f"Type: {strategy3.error_type}")
    print(f"Reduce resolution: {strategy3.reduce_resolution}")
    assert strategy3.error_type == ErrorType.OUT_OF_MEMORY
    assert strategy3.reduce_resolution == True
    
    print("\n✅ All Error Diagnostic tests passed.")

if __name__ == "__main__":
    test_error_agent()
