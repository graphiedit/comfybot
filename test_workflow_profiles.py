"""Test workflow profiles — verifies that each model builds a workflow with correct CLIP/VAE."""
import sys
import json

# Ensure imports work
sys.path.insert(0, ".")

from registry.model_registry import ModelRegistry, DEFAULT_MODEL_CATALOG, DEFAULT_WORKFLOW_PROFILES
from core.workflow_builder import WorkflowBuilder
from llm.base import GenerationPlan


def test_workflow_profiles():
    """Test that every catalogued diffusion model has a viable workflow profile."""
    config = {
        "comfyui": {"url": "http://127.0.0.1:8188"},
        "defaults": {"ckpt": "Juggernaut-XL_v9_RunDiffusion.safetensors"},
    }
    
    registry = ModelRegistry(config)
    # Load catalogs without querying ComfyUI
    registry._model_catalog = DEFAULT_MODEL_CATALOG
    registry._loaded = True
    
    print("=" * 60)
    print("TEST 1: Every diffusion model has a workflow profile")
    print("=" * 60)
    
    for dm in DEFAULT_MODEL_CATALOG["diffusion_models"]:
        filename = dm["filename"]
        profile = registry.get_workflow_profile(filename)
        assert profile, f"FAIL: No profile for {filename}"
        assert profile.get("clip_name1"), f"FAIL: No clip_name1 in profile for {filename}"
        assert profile.get("clip_name2"), f"FAIL: No clip_name2 in profile for {filename}"
        assert profile.get("vae"), f"FAIL: No vae in profile for {filename}"
        print(f"  ✅ {dm.get('display_name', filename)}: clip1={profile['clip_name1']}, clip2={profile['clip_name2']}, vae={profile['vae']}")
    
    print()
    
    print("=" * 60)
    print("TEST 2: Flux workflow uses profile CLIP/VAE (not hardcoded)")
    print("=" * 60)
    
    builder = WorkflowBuilder(config, registry=registry)
    
    # Test z_image_turbo (the model that was failing)
    plan = GenerationPlan(
        checkpoint="z_image_turbo_bf16.safetensors",
        model_arch="flux",
        enhanced_prompt="lord krishna holding flute",
        steps=4,
        cfg=1.0,
        sampler="euler",
        scheduler="simple",
    )
    
    workflow = builder._create_flux_workflow(plan)
    
    clip_node = workflow.get("clip_loader", {})
    clip_inputs = clip_node.get("inputs", {})
    
    assert clip_inputs.get("clip_name1") == "qwen_2.5_vl_7b_fp8_scaled.safetensors", \
        f"FAIL: clip_name1 = {clip_inputs.get('clip_name1')}, expected qwen_2.5_vl_7b_fp8_scaled.safetensors"
    assert clip_inputs.get("clip_name2") == "qwen_3_4b.safetensors", \
        f"FAIL: clip_name2 = {clip_inputs.get('clip_name2')}, expected qwen_3_4b.safetensors"
    
    vae_node = workflow.get("vae_loader", {})
    vae_inputs = vae_node.get("inputs", {})
    assert vae_inputs.get("vae_name") == "flux2-vae.safetensors", \
        f"FAIL: vae_name = {vae_inputs.get('vae_name')}, expected flux2-vae.safetensors"
    
    assert "t5xxl" not in json.dumps(workflow), "FAIL: Found hardcoded t5xxl in workflow!"
    assert "clip_l." not in json.dumps(workflow), "FAIL: Found hardcoded clip_l in workflow!"
    
    print(f"  ✅ z_image_turbo workflow CLIP: {clip_inputs['clip_name1']}, {clip_inputs['clip_name2']}")
    print(f"  ✅ z_image_turbo workflow VAE: {vae_inputs['vae_name']}")
    print(f"  ✅ No hardcoded t5xxl or clip_l references")
    print()
    
    # Test qwen_image_edit (uses different VAE)
    plan2 = GenerationPlan(
        checkpoint="qwen_image_edit_2509_fp8_e4m3fn.safetensors",
        model_arch="flux",
        enhanced_prompt="test",
        steps=20,
        cfg=1.0,
    )
    workflow2 = builder._create_flux_workflow(plan2)
    vae2 = workflow2["vae_loader"]["inputs"]["vae_name"]
    assert vae2 == "qwen_image_vae.safetensors", f"FAIL: qwen_image_edit VAE = {vae2}, expected qwen_image_vae.safetensors"
    print(f"  ✅ qwen_image_edit uses its own VAE: {vae2}")
    print()
    
    print("=" * 60)
    print("TEST 3: SDXL workflow still works (no profiles needed)")
    print("=" * 60)
    
    plan3 = GenerationPlan(
        checkpoint="Juggernaut-XL_v9_RunDiffusion.safetensors",
        model_arch="sdxl",
        enhanced_prompt="a beautiful landscape",
        steps=30,
        cfg=7.0,
    )
    workflow3 = builder._create_sdxl_workflow(plan3)
    assert "base_model" in workflow3, "FAIL: SDXL workflow missing base_model"
    assert workflow3["base_model"]["class_type"] == "CheckpointLoaderSimple"
    assert "clip_loader" not in workflow3, "FAIL: SDXL workflow should not have DualCLIPLoader"
    print(f"  ✅ SDXL workflow uses CheckpointLoaderSimple (no DualCLIPLoader)")
    print()
    
    print("=" * 60)
    print("TEST 4: Hunyuan workflow uses profiles")
    print("=" * 60)
    
    # Mark hunyuan as a diffusion model
    registry._diffusion_models = ["hunyuan_3d_v2.1.safetensors"]
    
    plan4 = GenerationPlan(
        checkpoint="hunyuan_3d_v2.1.safetensors",
        model_arch="hunyuan",
        enhanced_prompt="3d model",
        steps=30,
        cfg=1.0,
    )
    workflow4 = builder._create_hunyuan_workflow(plan4)
    clip4 = workflow4["clip_loader"]["inputs"]
    assert clip4["clip_name1"] == "qwen_2.5_vl_7b_fp8_scaled.safetensors", f"FAIL: Hunyuan clip1 = {clip4['clip_name1']}"
    assert "t5xxl" not in json.dumps(workflow4), "FAIL: Found hardcoded t5xxl in Hunyuan workflow!"
    print(f"  ✅ Hunyuan workflow CLIP: {clip4['clip_name1']}, {clip4['clip_name2']}")
    print()
    
    print("=" * 60)
    print("TEST 5: Unknown model falls back to arch defaults")
    print("=" * 60)
    
    profile_unknown = registry.get_workflow_profile("some_unknown_flux_model.safetensors")
    assert profile_unknown.get("clip_name1"), "FAIL: Unknown flux model has no fallback clip"
    print(f"  ✅ Unknown flux model falls back to: {profile_unknown.get('clip_name1')}")
    print()
    
    print("🎉 ALL TESTS PASSED!")


if __name__ == "__main__":
    test_workflow_profiles()
