"""
smoke_test_explain.py
---------------------
Smoke test for ExplainablePredictor (Grad-CAM) — v1.0.4.

Run after installing the wheel (e.g. on Colab):

    !pip install plantdoc_predictor-1.0.4-py3-none-any.whl
    !python smoke_test_explain.py

No real leaf images required — all tests use a dummy green PIL image.
Uses mobilenetv2_v1 (smallest model, fastest download).
"""

import sys
import os
import tempfile

# Windows consoles default to cp1252 and choke on the ✓/✗ glyphs below.
# Force UTF-8 so a failure prints its real message instead of UnicodeEncodeError.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

import numpy as np
from PIL import Image as PILImage

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
# Dummy image + temp output path
# ---------------------------------------------------------------------------
_dummy_path = None
_out_path = None


def _make_dummy():
    global _dummy_path, _out_path
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    PILImage.new("RGB", (256, 256), color=(34, 139, 34)).save(tmp.name)
    _dummy_path = tmp.name
    _out_path = os.path.join(tempfile.gettempdir(), "plantdoc_gradcam_test.jpg")


_make_dummy()


# ---------------------------------------------------------------------------
# 1. Import
# ---------------------------------------------------------------------------
print("\n[1] Import")

ExplainablePredictor = None


def _import():
    global ExplainablePredictor
    from plantdoc_predictor import ExplainablePredictor as EP
    ExplainablePredictor = EP


check("from plantdoc_predictor import ExplainablePredictor", _import)


# ---------------------------------------------------------------------------
# 2. Load (mobilenetv2_v1)
# ---------------------------------------------------------------------------
print("\n[2] ExplainablePredictor — load (mobilenetv2_v1)")

ep = None


def _load():
    global ep
    ep = ExplainablePredictor(model_name="mobilenetv2_v1", verbose=False)
    assert ep.model_name == "mobilenetv2_v1"


check("ExplainablePredictor(model_name='mobilenetv2_v1') initialises", _load)


# ---------------------------------------------------------------------------
# 3. explain() — basic structure + file output
# ---------------------------------------------------------------------------
print("\n[3] explain() — prediction + heatmap file")


def _explain_structure():
    r = ep.explain(_dummy_path, save_to=_out_path)
    for k in ("model", "label", "confidence", "crop", "disease", "is_healthy", "layer_name"):
        assert k in r, f"missing key '{k}'"
    assert r["model"] == "mobilenetv2_v1"
    assert isinstance(r["confidence"], float) and 0.0 <= r["confidence"] <= 1.0
    print(f"         → {r['label']}  ({r['confidence']:.2%})  layer={r['layer_name']}")


def _explain_writes_file():
    r = ep.explain(_dummy_path, save_to=_out_path)
    assert r["heatmap_path"] == _out_path
    assert os.path.exists(_out_path), "heatmap file was not written"
    # overlay must match the model input size
    saved = PILImage.open(_out_path)
    assert saved.size == tuple(ep.input_size), f"size={saved.size} vs {ep.input_size}"


def _explain_auto_layer_is_4d():
    r = ep.explain(_dummy_path)
    layer = ep.get_model().get_layer(r["layer_name"])
    assert len(layer.output.shape) == 4, "auto-detected layer is not a 4D feature map"


check("explain() returns full result dict",            _explain_structure)
check("explain(save_to=...) writes overlay file",      _explain_writes_file)
check("auto-detected target layer is 4D (conv)",       _explain_auto_layer_is_4d)


# ---------------------------------------------------------------------------
# 4. Raw heatmap array (return_heatmap=True)
# ---------------------------------------------------------------------------
print("\n[4] return_heatmap=True")


def _heatmap_array():
    r = ep.explain(_dummy_path, return_heatmap=True)
    hm = r["heatmap"]
    assert isinstance(hm, np.ndarray) and hm.ndim == 2, f"heatmap shape={getattr(hm,'shape',None)}"
    assert hm.min() >= 0.0 and hm.max() <= 1.0, f"heatmap range [{hm.min()},{hm.max()}]"
    print(f"         → heatmap {hm.shape}, range [{hm.min():.3f}, {hm.max():.3f}]")


def _overlay_array():
    r = ep.explain(_dummy_path, return_heatmap=True)
    ov = r["overlay"]
    assert ov.dtype == np.uint8 and ov.ndim == 3 and ov.shape[2] == 3
    assert ov.shape[:2] == tuple(reversed(ep.input_size)) or ov.shape[:2] == tuple(ep.input_size)


check("heatmap is 2D float array in [0, 1]",   _heatmap_array)
check("overlay is HxWx3 uint8 array",          _overlay_array)


# ---------------------------------------------------------------------------
# 5. Overrides — layer_name, class_index, PIL input
# ---------------------------------------------------------------------------
print("\n[5] Overrides")


def _explicit_layer():
    target = ep.explain(_dummy_path)["layer_name"]
    r = ep.explain(_dummy_path, layer_name=target)
    assert r["layer_name"] == target


def _class_index_override():
    r = ep.explain(_dummy_path, class_index=0)
    assert r["label"] == (ep.labels[0] if ep.labels else "Class_0")


def _pil_input():
    img = PILImage.new("RGB", (224, 224), color=(34, 139, 34))
    r = ep.explain(img, return_heatmap=True)
    assert "heatmap" in r and r["heatmap"].ndim == 2


def _bad_layer_raises():
    try:
        ep.explain(_dummy_path, layer_name="definitely_not_a_layer")
    except Exception:
        return
    raise AssertionError("expected an error for a non-existent layer name")


check("explicit layer_name is honoured",         _explicit_layer)
check("class_index override selects that class", _class_index_override)
check("PIL Image input works",                   _pil_input)
check("invalid layer_name raises",               _bad_layer_raises)


# ---------------------------------------------------------------------------
# 6. Pass-through utilities
# ---------------------------------------------------------------------------
print("\n[6] Pass-through utilities")


def _list_layers():
    layers = ep.list_layers()
    assert isinstance(layers, list) and len(layers) > 0
    print(f"         → {len(layers)} layers")


def _get_model():
    assert ep.get_model() is not None


check("list_layers() returns layer list", _list_layers)
check("get_model() returns non-None",     _get_model)


# ---------------------------------------------------------------------------
# 7. CLI — `plantdoc explain` and `plantdoc predict --guard`
# ---------------------------------------------------------------------------
print("\n[7] CLI (plantdoc explain / predict --guard)")

import json as _json
from click.testing import CliRunner
from plantdoc_predictor.cli import cli

_runner = CliRunner()
_cli_out = os.path.join(tempfile.gettempdir(), "plantdoc_gradcam_cli.jpg")


def _last_json(text):
    """Extract the final JSON object from CLI output (TF/keras noise precedes it)."""
    for line in reversed(text.strip().splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            return _json.loads(line)
    raise AssertionError(f"no JSON line found in output:\n{text}")


def _cli_explain_writes_file():
    res = _runner.invoke(cli, [
        "explain", _dummy_path, "--model", "mobilenetv2_v1", "--save-to", _cli_out,
    ])
    assert res.exit_code == 0, f"exit={res.exit_code}\n{res.output}"
    assert os.path.exists(_cli_out), "explain CLI did not write the heatmap file"


def _cli_explain_json():
    res = _runner.invoke(cli, [
        "explain", _dummy_path, "--model", "mobilenetv2_v1",
        "--save-to", _cli_out, "--json",
    ])
    assert res.exit_code == 0, f"exit={res.exit_code}\n{res.output}"
    data = _last_json(res.output)
    assert "heatmap_path" in data, f"missing heatmap_path: {data}"
    assert "layer_name" in data, f"missing layer_name: {data}"
    print(f"         → {data['label']}  layer={data['layer_name']}")


def _cli_explain_folder_rejected():
    res = _runner.invoke(cli, ["explain", tempfile.gettempdir(), "--model", "mobilenetv2_v1"])
    assert res.exit_code != 0, "explain on a folder should exit non-zero"


def _cli_predict_guard_reject():
    # guard-threshold 1.0 → nothing can pass → is_leaf False
    res = _runner.invoke(cli, [
        "predict", _dummy_path, "--model", "mobilenetv2_v1",
        "--guard", "--guard-threshold", "1.0", "--json",
    ])
    assert res.exit_code == 0, f"exit={res.exit_code}\n{res.output}"
    data = _last_json(res.output)
    assert data["is_leaf"] is False, f"expected is_leaf=False, got {data}"
    print(f"         → rejected, guard_score={data['guard_score']}")


def _cli_predict_guard_pass():
    # guard-threshold 0.0 → everything passes → is_leaf True + a label
    res = _runner.invoke(cli, [
        "predict", _dummy_path, "--model", "mobilenetv2_v1",
        "--guard", "--guard-threshold", "0.0", "--json",
    ])
    assert res.exit_code == 0, f"exit={res.exit_code}\n{res.output}"
    data = _last_json(res.output)
    assert data["is_leaf"] is True, f"expected is_leaf=True, got {data}"
    assert data["label"] != "unknown"


check("plantdoc explain writes heatmap file",       _cli_explain_writes_file)
check("plantdoc explain --json emits result dict",  _cli_explain_json)
check("plantdoc explain on a folder is rejected",   _cli_explain_folder_rejected)
check("plantdoc predict --guard rejects non-leaf",  _cli_predict_guard_reject)
check("plantdoc predict --guard passes leaf",       _cli_predict_guard_pass)


# ---------------------------------------------------------------------------
# 8. Cleanup
# ---------------------------------------------------------------------------
for p in (_dummy_path, _out_path, _cli_out):
    if p and os.path.exists(p):
        os.unlink(p)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
total = len(results)
passed = sum(results)
failed = total - passed

print(f"\n{'='*50}")
print(f"  {passed}/{total} tests passed", end="")
if failed:
    print(f"  |  {failed} FAILED")
else:
    print("  |  ExplainablePredictor looks good — safe to publish")
print(f"{'='*50}\n")

sys.exit(0 if failed == 0 else 1)
