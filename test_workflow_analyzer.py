"""Test the Workflow Analyzer — verifies auto-introspection of workflow JSON files."""
import sys
import json

sys.path.insert(0, ".")

from core.workflow_analyzer import WorkflowAnalyzer, WorkflowProfile
from core.workflow_manager import WorkflowManager


def test_workflow_analyzer():
    """Test the workflow analyzer against all existing workflow templates."""
    print("=" * 60)
    print("TEST 1: Workflow Analyzer — Auto-Introspection")
    print("=" * 60)

    analyzer = WorkflowAnalyzer()

    # Load all workflows through the manager (which now auto-analyzes)
    manager = WorkflowManager()
    
    assert len(manager.profiles) > 0, "FAIL: No workflow profiles loaded!"
    print(f"  Loaded {len(manager.profiles)} workflow profiles\n")

    for name, profile in manager.profiles.items():
        print(f"  📄 {name}")
        print(f"     Architecture: {profile.architecture}")
        print(f"     Capabilities: {', '.join(sorted(profile.capabilities)) or 'none'}")
        print(f"     Image inputs:  {profile.min_images} required")
        if profile.image_inputs:
            for img in profile.image_inputs:
                print(f"       - {img.purpose} (node {img.node_id})")
        print(f"     Has prompt:    {profile.has_positive_prompt}")
        print(f"     Total nodes:   {profile.total_nodes}")
        if profile.tunable_params:
            print(f"     Tunable:       {', '.join(f'{k}={v}' for k, v in list(profile.tunable_params.items())[:5])}")
        print()

    print("=" * 60)
    print("TEST 2: Text-only workflows have text_to_image capability")
    print("=" * 60)

    for name, profile in manager.profiles.items():
        if not profile.requires_image:
            assert "text_to_image" in profile.capabilities, \
                f"FAIL: {name} has no images but missing text_to_image capability"
            print(f"  ✅ {name}: text_to_image ✓")

    print()

    print("=" * 60)
    print("TEST 3: Image-requiring workflows have image_input capability")
    print("=" * 60)

    for name, profile in manager.profiles.items():
        if profile.requires_image:
            assert "image_input" in profile.capabilities, \
                f"FAIL: {name} requires images but missing image_input capability"
            assert profile.min_images > 0, \
                f"FAIL: {name} requires images but min_images=0"
            print(f"  ✅ {name}: image_input ✓ (needs {profile.min_images})")

    print()

    print("=" * 60)
    print("TEST 4: LLM Summary Generation")
    print("=" * 60)

    summaries = manager.get_workflow_summaries_for_llm()
    assert len(summaries) > 0, "FAIL: Empty LLM summaries"
    print(summaries)
    print()

    print("=" * 60)
    print("TEST 5: Rich Workflow List")
    print("=" * 60)

    rich = manager.get_workflow_list_rich()
    assert len(rich) == len(manager.templates), \
        f"FAIL: rich list has {len(rich)} items, expected {len(manager.templates)}"
    for name, data in rich.items():
        assert "capabilities" in data, f"FAIL: {name} missing capabilities"
        assert "requires_image" in data, f"FAIL: {name} missing requires_image"
        print(f"  ✅ {name}: {data['capabilities']}")

    print()

    print("=" * 60)
    print("TEST 6: Workflow Intent Filtering")
    print("=" * 60)

    # Text-only request
    text_only = manager.find_workflows_for_intent(has_image=False)
    print(f"  Text-only workflows: {list(text_only.keys())}")

    # Image edit request
    image_edit = manager.find_workflows_for_intent(has_image=True, wants_edit=True)
    print(f"  Image edit workflows: {list(image_edit.keys())}")

    print()
    print("🎉 ALL WORKFLOW ANALYZER TESTS PASSED!")


if __name__ == "__main__":
    test_workflow_analyzer()
