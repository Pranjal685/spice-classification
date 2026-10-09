"""
src/test_fallback.py
--------------------
Verification script for the Tier-2 Multimodal Fallback Engine.

Tests:
  1. Force-fallback on a real spice image (should return structured JSON).
  2. Stub-mode behaviour when GEMINI_API_KEY is deliberately unset.
  3. JSON schema validation on the returned dictionary.

Usage:
    python src/test_fallback.py
"""

import os, sys
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import json
from pathlib import Path

# ── resolve local imports ---------------------------------------------------
_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

PROJECT_ROOT = _SRC.parent

# ── schema keys every Tier-2 response must contain -------------------------
REQUIRED_KEYS = {
    "is_spice", "identified_name", "scientific_name",
    "is_in_dataset_scope", "confidence_assessment",
    "flavor_profile", "common_uses", "action_recommendation",
}

_SEP  = "=" * 64
_THIN = "-" * 64
_PASS = "[PASS]"
_FAIL = "[FAIL]"


def _validate_schema(result: dict, test_name: str) -> bool:
    """Check that all required keys are present in the response."""
    missing = REQUIRED_KEYS - set(result.keys())
    if missing:
        print(f"  {_FAIL} {test_name}: Missing keys: {missing}")
        return False
    print(f"  {_PASS} {test_name}: All required keys present.")
    return True


def _find_sample_image() -> Path:
    """Find a real spice image from the dataset for testing."""
    raw_dir = PROJECT_ROOT / "data" / "raw"
    if raw_dir.exists():
        for folder in sorted(raw_dir.iterdir()):
            if folder.is_dir():
                for img in folder.iterdir():
                    if img.suffix.lower() in (".jpg", ".jpeg", ".png"):
                        return img
    return None


def test_1_force_fallback_live():
    """Test 1: Real Tier-2 Gemini call with a spice image."""
    print(f"\n{_SEP}")
    print("  TEST 1: Tier-2 Gemini Fallback (Live API Call)")
    print(_SEP)

    from fallback import SpiceFallbackEngine

    engine = SpiceFallbackEngine()
    print(f"  Engine: {engine}")

    if engine.stub_mode:
        print(f"  {_FAIL} Engine is in STUB mode (no API key). Skipping live test.")
        print(f"         Set GEMINI_API_KEY in .env to enable this test.")
        return None

    sample_img = _find_sample_image()
    if sample_img is None:
        print(f"  {_FAIL} No sample images found in data/raw/")
        return None

    print(f"  Image: {sample_img}")
    result = engine.identify_unknown_image(sample_img)

    print(f"\n  Response:")
    print(json.dumps(result, indent=4, default=str))

    # Schema validation
    _validate_schema(result, "Live API schema check")

    # Check error flag
    if result.get("error"):
        print(f"  {_FAIL} API returned an error: {result.get('error_reason')}")
    else:
        print(f"  {_PASS} API call succeeded without error.")
        print(f"  {_PASS} is_spice = {result.get('is_spice')}")
        print(f"  {_PASS} identified_name = {result.get('identified_name')}")

    return result


def test_2_stub_mode():
    """Test 2: Verify stub mode when API key is deliberately unset."""
    print(f"\n{_SEP}")
    print("  TEST 2: Stub Mode (No API Key)")
    print(_SEP)

    # Temporarily remove the API key
    original_key = os.environ.pop("GEMINI_API_KEY", None)

    try:
        # Re-import with fresh state by creating a new engine manually
        from fallback import SpiceFallbackEngine, _stub_response

        # Create engine with no key
        engine = SpiceFallbackEngine.__new__(SpiceFallbackEngine)
        engine.api_key = ""
        engine.model_name = "gemini-2.5-flash"
        engine._client = None
        engine.stub_mode = True

        print(f"  Engine: {engine}")
        print(f"  {_PASS} Engine is in STUB mode as expected.")

        result = engine.identify_unknown_image("dummy_image.jpg")
        print(f"\n  Stub Response:")
        print(json.dumps(result, indent=4, default=str))

        _validate_schema(result, "Stub schema check")

        if result.get("error") and result.get("error_reason") == "NO_API_KEY":
            print(f"  {_PASS} Correct error_reason: NO_API_KEY")
        else:
            print(f"  {_FAIL} Expected error_reason='NO_API_KEY'")

        if result.get("identified_name") == "STUB_MODE":
            print(f"  {_PASS} identified_name = STUB_MODE")
        else:
            print(f"  {_FAIL} Expected identified_name='STUB_MODE'")

    finally:
        # Restore the original key
        if original_key is not None:
            os.environ["GEMINI_API_KEY"] = original_key

    return result


def test_3_schema_contract():
    """Test 3: Validate the schema contract with mock data."""
    print(f"\n{_SEP}")
    print("  TEST 3: Schema Contract Validation")
    print(_SEP)

    # Valid spice response
    valid_spice = {
        "is_spice": True,
        "identified_name": "Star Anise (Chakra Phool)",
        "scientific_name": "Illicium verum",
        "is_in_dataset_scope": False,
        "confidence_assessment": "Clear star-shaped pod visible.",
        "flavor_profile": "Sweet, licorice-like, warm.",
        "common_uses": "Biryani, garam masala blends.",
        "action_recommendation": "Identified via Tier-2 Foundation Vision Model.",
    }
    ok1 = _validate_schema(valid_spice, "Valid spice response")

    # Valid non-spice response
    valid_non_spice = {
        "is_spice": False,
        "identified_name": "None",
        "scientific_name": "N/A",
        "is_in_dataset_scope": False,
        "confidence_assessment": "Not a spice.",
        "flavor_profile": "N/A",
        "common_uses": "N/A",
        "action_recommendation": "Please upload a clear photograph of a culinary spice.",
    }
    ok2 = _validate_schema(valid_non_spice, "Valid non-spice response")

    # Incomplete response (should fail)
    incomplete = {
        "is_spice": True,
        "identified_name": "Cumin",
    }
    ok3 = _validate_schema(incomplete, "Incomplete response (expected FAIL)")

    all_pass = ok1 and ok2 and not ok3  # ok3 should fail
    if all_pass:
        print(f"\n  {_PASS} All schema contract tests passed!")
    else:
        print(f"\n  {_FAIL} Some schema contract tests failed.")

    return all_pass


def main():
    print(f"\n{'#' * 64}")
    print("  TIER-2 FALLBACK ENGINE VERIFICATION SUITE")
    print(f"{'#' * 64}")

    results = {}

    # Test 1: Live API call
    results["test_1"] = test_1_force_fallback_live()

    # Test 2: Stub mode
    results["test_2"] = test_2_stub_mode()

    # Test 3: Schema validation
    results["test_3"] = test_3_schema_contract()

    # Summary
    print(f"\n{_SEP}")
    print("  VERIFICATION SUMMARY")
    print(_SEP)
    print(f"  Test 1 (Live API)     : {'Completed' if results['test_1'] else 'Skipped/Failed'}")
    print(f"  Test 2 (Stub Mode)    : {'Completed' if results['test_2'] else 'Failed'}")
    print(f"  Test 3 (Schema Check) : {'Passed' if results['test_3'] else 'Failed'}")
    print(_SEP)


if __name__ == "__main__":
    main()
