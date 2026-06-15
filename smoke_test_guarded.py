"""
smoke_test_guarded.py
---------------------
Smoke test for GuardedPredictor and label_parser — v1.0.4.

Run on Colab after installing the wheel:

    !pip install plantdoc_predictor-1.0.4-py3-none-any.whl
    !python smoke_test_guarded.py

No real leaf images required — all tests use dummy PIL images.
Guard rejection / pass-through are forced via threshold tricks (0.0 / 1.0).
CLIP model downloads ~400 MB on first use (cached after that).
"""

import sys
import os
import tempfile

PASS = "✓ PASS"
FAIL = "✗ FAIL"
results = []


def check(label, fn):
    try:
        fn()
        print(f"  {PASS}  {label}")
        results.append(True)
    except Exception as e:
        print(f"  {FAIL}  {label}")
        print(f"         {type(e).__name__}: {e}")
        results.append(False)


# ---------------------------------------------------------------------------
# Dummy image (solid green — used throughout, no real leaf needed)
# ---------------------------------------------------------------------------
from PIL import Image as PILImage

_dummy_path = None

def _make_dummy():
    global _dummy_path
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    PILImage.new("RGB", (256, 256), color=(34, 139, 34)).save(tmp.name)
    _dummy_path = tmp.name

_make_dummy()


# ---------------------------------------------------------------------------
# 1. Imports
# ---------------------------------------------------------------------------
print("\n[1] Imports")

GuardedPredictor = None
parse_label = None


def _import_guarded():
    global GuardedPredictor
    from plantdoc_predictor import GuardedPredictor as GP
    GuardedPredictor = GP


def _import_label_parser():
    global parse_label
    from plantdoc_predictor.utils.label_parser import parse_label as PL
    parse_label = PL


check("from plantdoc_predictor import GuardedPredictor", _import_guarded)
check("from plantdoc_predictor.utils.label_parser import parse_label", _import_label_parser)


# ---------------------------------------------------------------------------
# 2. Label parser — pure function, no model needed
# ---------------------------------------------------------------------------
print("\n[2] Label parser (parse_label)")


def _parse_disease():
    r = parse_label("Apple___Apple_scab")
    assert r["crop"] == "Apple", f"crop={r['crop']}"
    assert r["disease"] == "Apple scab", f"disease={r['disease']}"
    assert r["is_healthy"] is False


def _parse_healthy():
    r = parse_label("Blueberry___healthy")
    assert r["crop"] == "Blueberry"
    assert r["disease"] is None
    assert r["is_healthy"] is True


def _parse_unknown():
    r = parse_label("unknown")
    assert r == {"crop": None, "disease": None, "is_healthy": None}


def _parse_none():
    r = parse_label(None)
    assert r == {"crop": None, "disease": None, "is_healthy": None}


def _parse_multi_word():
    r = parse_label("Tomato___Early_blight")
    assert r["crop"] == "Tomato"
    assert r["disease"] == "Early blight"
    assert r["is_healthy"] is False


check("Apple___Apple_scab  → crop/disease/is_healthy", _parse_disease)
check("Blueberry___healthy → is_healthy=True, disease=None", _parse_healthy)
check("'unknown'           → all None", _parse_unknown)
check("None input          → all None", _parse_none)
check("Tomato___Early_blight → multi-word disease parsed", _parse_multi_word)


# ---------------------------------------------------------------------------
# 3. GuardedPredictor — load
# (uses mobilenetv2_v1: smallest model, fastest download)
# CLIP also loads here on first predict — ~400 MB one-time download
# ---------------------------------------------------------------------------
print("\n[3] GuardedPredictor — load (mobilenetv2_v1)")

gp_default = None


def _load_guarded():
    global gp_default
    gp_default = GuardedPredictor(model_name="mobilenetv2_v1", verbose=False)
    assert gp_default.model_name == "mobilenetv2_v1"


check("GuardedPredictor(model_name='mobilenetv2_v1') initialises", _load_guarded)


# ---------------------------------------------------------------------------
# 4. Guard REJECTION path
# threshold=1.0 → no image can score >= 1.0, so everything is rejected
# This tests the non-leaf rejection branch without needing a real non-leaf photo
# ---------------------------------------------------------------------------
print("\n[4] Guard rejection (threshold=1.0 forces reject on any image)")

gp_reject = None


def _load_reject():
    global gp_reject
    gp_reject = GuardedPredictor(model_name="mobilenetv2_v1", guard_threshold=1.0, verbose=False)


def _reject_returns_non_leaf():
    result = gp_reject.predict(_dummy_path)
    assert result["is_leaf"] is False, f"expected is_leaf=False, got {result['is_leaf']}"
    assert result["label"] == "unknown", f"expected label='unknown', got {result['label']}"
    assert result["confidence"] is None, f"expected confidence=None, got {result['confidence']}"
    assert result["crop"] is None
    assert result["disease"] is None
    assert result["is_healthy"] is None
    print(f"         → guard_score={result['guard_score']}")


def _reject_has_model_name():
    result = gp_reject.predict(_dummy_path)
    assert result["model"] == "mobilenetv2_v1"


def _reject_guard_score_is_float_0_to_1():
    result = gp_reject.predict(_dummy_path)
    s = result["guard_score"]
    assert isinstance(s, float), f"guard_score type={type(s)}"
    assert 0.0 <= s <= 1.0, f"guard_score out of range: {s}"


def _reject_guard_score_rounded_4dp():
    result = gp_reject.predict(_dummy_path)
    s = result["guard_score"]
    assert s == round(s, 4), f"guard_score not rounded to 4dp: {s}"


def _reject_pil_input():
    img = PILImage.new("RGB", (224, 224), color=(255, 0, 0))
    result = gp_reject.predict(img)
    assert result["is_leaf"] is False


check("GuardedPredictor(threshold=1.0) loads",              _load_reject)
check("predict() returns is_leaf=False, label='unknown'",   _reject_returns_non_leaf)
check("rejected result contains model name",                _reject_has_model_name)
check("guard_score is float in [0, 1]",                     _reject_guard_score_is_float_0_to_1)
check("guard_score is rounded to 4 decimal places",         _reject_guard_score_rounded_4dp)
check("PIL Image input also rejected correctly",            _reject_pil_input)


# ---------------------------------------------------------------------------
# 5. Guard PASS-THROUGH path
# threshold=0.0 → every image passes (score is always >= 0.0)
# This tests the full prediction + label parsing branch
# ---------------------------------------------------------------------------
print("\n[5] Guard pass-through (threshold=0.0 forces pass on any image)")

gp_pass = None


def _load_pass():
    global gp_pass
    gp_pass = GuardedPredictor(model_name="mobilenetv2_v1", guard_threshold=0.0, verbose=False)


def _pass_result_structure():
    result = gp_pass.predict(_dummy_path)
    assert result["is_leaf"] is True, f"expected is_leaf=True, got {result['is_leaf']}"
    assert result["label"] != "unknown", "label should not be 'unknown' when guard passes"
    assert isinstance(result["confidence"], float), "confidence should be a float"
    assert 0.0 <= result["confidence"] <= 1.0, f"confidence out of range: {result['confidence']}"
    print(f"         → {result['label']}  ({result['confidence']:.2%})")


def _pass_has_parsed_fields():
    result = gp_pass.predict(_dummy_path)
    assert "crop" in result, "missing 'crop'"
    assert "disease" in result, "missing 'disease'"
    assert "is_healthy" in result, "missing 'is_healthy'"
    assert "guard_score" in result, "missing 'guard_score'"
    print(f"         → crop={result['crop']}  disease={result['disease']}  is_healthy={result['is_healthy']}")


def _pass_label_matches_parsed_crop():
    result = gp_pass.predict(_dummy_path)
    if result["crop"] is not None:
        assert result["crop"] in result["label"], \
            f"crop '{result['crop']}' not found in label '{result['label']}'"


def _pass_top_k():
    result = gp_pass.predict(_dummy_path, top_k=3)
    assert "top_k" in result, "missing 'top_k' key"
    assert len(result["top_k"]) == 3, f"expected 3, got {len(result['top_k'])}"
    confs = [r["confidence"] for r in result["top_k"]]
    assert confs == sorted(confs, reverse=True), "top_k not sorted by confidence"
    print(f"         → top-3 labels: {[r['label'].split('___')[-1] for r in result['top_k']]}")


def _pass_pil_image():
    img = PILImage.new("RGB", (256, 256), color=(34, 139, 34))
    result = gp_pass.predict(img)
    assert result["is_leaf"] is True
    assert "label" in result and result["label"] != "unknown"


check("GuardedPredictor(threshold=0.0) loads",                  _load_pass)
check("predict() returns is_leaf=True + label + confidence",     _pass_result_structure)
check("result has crop, disease, is_healthy, guard_score keys",  _pass_has_parsed_fields)
check("crop field matches label string",                         _pass_label_matches_parsed_crop)
check("predict(top_k=3) returns sorted top_k list",             _pass_top_k)
check("PIL Image input passes through correctly",               _pass_pil_image)


# ---------------------------------------------------------------------------
# 6. min_confidence parameter
# threshold=0.0 (always pass CLIP) + min_confidence=1.0 (impossible to meet)
# → disease model result should be 'unknown' even though CLIP passed
# ---------------------------------------------------------------------------
print("\n[6] min_confidence floor")


def _min_confidence_blocks():
    gp = GuardedPredictor(
        model_name="mobilenetv2_v1",
        guard_threshold=0.0,
        min_confidence=1.0,   # impossible threshold → always blocked
        verbose=False,
    )
    result = gp.predict(_dummy_path)
    assert result["is_leaf"] is True, "is_leaf should be True (passed CLIP)"
    assert result["label"] == "unknown", f"expected 'unknown', got {result['label']}"
    assert result["crop"] is None
    assert result["disease"] is None
    assert isinstance(result["confidence"], float), "confidence should still be reported"
    print(f"         → blocked at confidence={result['confidence']:.4f}  (threshold=1.0)")


def _min_confidence_zero_never_blocks():
    gp = GuardedPredictor(
        model_name="mobilenetv2_v1",
        guard_threshold=0.0,
        min_confidence=0.0,   # disabled
        verbose=False,
    )
    result = gp.predict(_dummy_path)
    assert result["label"] != "unknown", "min_confidence=0.0 should never block"


check("min_confidence=1.0 blocks result → label='unknown', is_leaf=True", _min_confidence_blocks)
check("min_confidence=0.0 (default) never blocks",                        _min_confidence_zero_never_blocks)


# ---------------------------------------------------------------------------
# 7. Pass-through utilities
# ---------------------------------------------------------------------------
print("\n[7] Pass-through utilities (delegates to inner Predictor)")


def _list_layers():
    layers = gp_default.list_layers()
    assert isinstance(layers, list) and len(layers) > 0
    print(f"         → {len(layers)} layers")


def _get_model():
    m = gp_default.get_model()
    assert m is not None


def _get_weights_info():
    info = gp_default.get_weights_info()
    assert isinstance(info, dict) and len(info) > 0


def _list_available_models():
    models = gp_default.list_available_models()
    assert isinstance(models, list) and len(models) > 0
    assert any("mobilenet" in m.lower() for m in models)
    print(f"         → {len(models)} models in registry")


check("list_layers() returns layer list",          _list_layers)
check("get_model() returns non-None",              _get_model)
check("get_weights_info() returns dict",           _get_weights_info)
check("list_available_models() works via guard",   _list_available_models)


# ---------------------------------------------------------------------------
# 8. Cleanup
# ---------------------------------------------------------------------------
if _dummy_path and os.path.exists(_dummy_path):
    os.unlink(_dummy_path)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
total  = len(results)
passed = sum(results)
failed = total - passed

print(f"\n{'='*50}")
print(f"  {passed}/{total} tests passed", end="")
if failed:
    print(f"  |  {failed} FAILED")
else:
    print("  |  GuardedPredictor looks good — safe to publish")
print(f"{'='*50}\n")

sys.exit(0 if failed == 0 else 1)
