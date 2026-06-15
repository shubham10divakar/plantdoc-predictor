"""
smoke_test.py
-------------
Run this from OUTSIDE the project directory after installing the wheel.

    pip install dist/plantdoc_predictor-1.0.4-py3-none-any.whl
    cd C:\\Users\\Subham\\Desktop
    python "D:\\...\\smoke_test.py"

All tests print PASS / FAIL. Exit code 1 if any test fails.
"""

import sys
import os
import json
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
# 0. Verify we are NOT running from inside the package source directory
# ---------------------------------------------------------------------------
cwd = os.getcwd()
package_src = os.path.join(cwd, "plantdoc_predictor")
if os.path.isdir(package_src):
    print(
        "\n  WARNING: You are running this from inside the project directory.\n"
        "  Python will pick up the local source instead of the installed package.\n"
        "  cd elsewhere and re-run for an accurate test.\n"
    )

# ---------------------------------------------------------------------------
# 1. Imports
# ---------------------------------------------------------------------------
print("\n[1] Imports")

Predictor = None
BatchPredictor = None
list_available_models = None


def _import_predictor():
    global Predictor
    from plantdoc_predictor import Predictor as P
    Predictor = P


def _import_batch():
    global BatchPredictor
    from plantdoc_predictor import BatchPredictor as BP
    BatchPredictor = BP


def _import_list():
    global list_available_models
    from plantdoc_predictor import list_available_models as L
    list_available_models = L


check("from plantdoc_predictor import Predictor", _import_predictor)
check("from plantdoc_predictor import BatchPredictor", _import_batch)
check("from plantdoc_predictor import list_available_models", _import_list)

# ---------------------------------------------------------------------------
# 2. Registry
# ---------------------------------------------------------------------------
print("\n[2] Model registry")

models = []


def _list_models():
    global models
    models = list_available_models()
    assert isinstance(models, list) and len(models) > 0, "empty registry"


def _registry_has_mobilenet():
    assert any("mobilenet" in m.lower() for m in models), "mobilenetv2_v1 not found"


def _registry_has_regnet_models():
    for name in ("regnetx_160_v1", "regnety_160_v1", "regnety_320_v1"):
        assert name in models, f"{name} not found in registry"


def _registry_count():
    assert len(models) >= 23, f"expected at least 23 models, got {len(models)}"
    print(f"         → {len(models)} models registered")


check("list_available_models() returns a non-empty list", _list_models)
check("Registry contains mobilenetv2_v1", _registry_has_mobilenet)
check("Registry contains all 3 RegNet models", _registry_has_regnet_models)
check("Registry has at least 23 models", _registry_count)

# ---------------------------------------------------------------------------
# 3. Create a dummy leaf image for testing (no external files needed)
# ---------------------------------------------------------------------------
print("\n[3] Dummy test image")

dummy_path = None


def _create_dummy_image():
    global dummy_path
    from PIL import Image
    img = Image.new("RGB", (256, 256), color=(34, 139, 34))   # solid green
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    img.save(tmp.name)
    dummy_path = tmp.name


check("Create 256x256 green JPG with Pillow", _create_dummy_image)

# ---------------------------------------------------------------------------
# 4. Predictor — load + predict (uses MobileNetV2, smallest/fastest model)
# ---------------------------------------------------------------------------
print("\n[4] Predictor (mobilenetv2_v1)")

predictor = None


def _load_predictor():
    global predictor
    predictor = Predictor(model_name="mobilenetv2_v1", verbose=False)


def _predict():
    result = predictor.predict(dummy_path)
    assert "label" in result, "missing 'label' key"
    assert "confidence" in result, "missing 'confidence' key"
    assert isinstance(result["confidence"], float), "confidence not a float"
    assert "top_k" not in result, "top_k should not be present for top_k=1"
    print(f"         → {result['label']}  ({result['confidence']:.2%})")


def _predict_pil():
    from PIL import Image
    pil_img = Image.new("RGB", (256, 256), color=(34, 139, 34))
    result = predictor.predict(pil_img)
    assert "label" in result, "PIL predict missing 'label'"
    print(f"         → {result['label']}  ({result['confidence']:.2%})")


def _predict_top_k():
    result = predictor.predict(dummy_path, top_k=3)
    assert "top_k" in result, "missing 'top_k' key"
    assert len(result["top_k"]) == 3, f"expected 3 results, got {len(result['top_k'])}"
    assert result["label"] == result["top_k"][0]["label"], "top-1 label mismatch"
    assert result["confidence"] == result["top_k"][0]["confidence"], "top-1 confidence mismatch"
    confs = [r["confidence"] for r in result["top_k"]]
    assert confs == sorted(confs, reverse=True), "top_k not sorted by confidence"
    print(f"         → top-3: {[r['label'].split('___')[-1] for r in result['top_k']]}")


check("Predictor(model_name='mobilenetv2_v1') loads", _load_predictor)
check("predict() returns label + confidence (top_k=1 default)", _predict)
check("predict() accepts PIL Image directly", _predict_pil)
check("predict(top_k=3) returns ranked top_k list", _predict_top_k)

# ---------------------------------------------------------------------------
# 5. Model inspection
# ---------------------------------------------------------------------------
print("\n[5] Model inspection")


def _get_model():
    m = predictor.get_model()
    assert m is not None


def _get_labels():
    labels = predictor.get_labels()
    assert isinstance(labels, list) and len(labels) == 38, f"expected 38 labels, got {len(labels)}"


def _list_layers():
    layers = predictor.list_layers()
    assert isinstance(layers, list) and len(layers) > 0


def _get_weights_info():
    info = predictor.get_weights_info()
    assert isinstance(info, dict) and len(info) > 0


def _get_weights():
    import numpy as np
    weights = predictor.get_weights()
    assert isinstance(weights, list)
    assert all(isinstance(w, np.ndarray) for w in weights)


check("get_model() returns non-None", _get_model)
check("get_labels() returns 38 classes", _get_labels)
check("list_layers() returns layer list", _list_layers)
check("get_weights_info() returns dict", _get_weights_info)
check("get_weights() returns list of numpy arrays", _get_weights)

# ---------------------------------------------------------------------------
# 6. Feature extraction
# ---------------------------------------------------------------------------
print("\n[6] Feature extraction")


def _extract_features():
    import numpy as np
    features = predictor.extract_features(dummy_path)
    assert isinstance(features, np.ndarray), "expected numpy array"
    assert features.ndim >= 1
    print(f"         → shape {features.shape}")


check("extract_features() returns numpy array", _extract_features)

# ---------------------------------------------------------------------------
# 7. BatchPredictor
# ---------------------------------------------------------------------------
print("\n[7] BatchPredictor")

batch_results = []


def _batch_run():
    global batch_results
    bp = BatchPredictor(model_name="mobilenetv2_v1")
    batch_results = bp.run([dummy_path, dummy_path])
    assert len(batch_results) == 2
    assert all("label" in r for r in batch_results)


def _batch_export_csv():
    bp = BatchPredictor(model_name="mobilenetv2_v1")
    out = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
    bp.export_csv(batch_results, out.name)
    assert os.path.getsize(out.name) > 0
    os.unlink(out.name)


def _batch_export_json():
    bp = BatchPredictor(model_name="mobilenetv2_v1")
    out = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
    bp.export_json(batch_results, out.name)
    assert os.path.getsize(out.name) > 0
    os.unlink(out.name)


check("BatchPredictor.run() on 2 images", _batch_run)
check("export_csv() writes non-empty file", _batch_export_csv)
check("export_json() writes non-empty file", _batch_export_json)

# ---------------------------------------------------------------------------
# 8. PyTorch models — RegNet (new in v1.0.4)
# ---------------------------------------------------------------------------
print("\n[8] PyTorch RegNet models")

REGNET_MODELS = ["regnetx_160_v1", "regnety_160_v1", "regnety_320_v1"]
_pt_predictor = None


def _load_regnetx():
    global _pt_predictor
    _pt_predictor = Predictor(model_name="regnetx_160_v1", verbose=False)


def _regnetx_predict():
    result = _pt_predictor.predict(dummy_path)
    assert "label" in result, "missing 'label'"
    assert "confidence" in result, "missing 'confidence'"
    assert isinstance(result["confidence"], float)
    print(f"         → {result['label']}  ({result['confidence']:.2%})")


def _regnetx_predict_top_k():
    result = _pt_predictor.predict(dummy_path, top_k=3)
    assert "top_k" in result, "missing 'top_k'"
    assert len(result["top_k"]) == 3
    confs = [r["confidence"] for r in result["top_k"]]
    assert confs == sorted(confs, reverse=True), "top_k not sorted"
    print(f"         → top-3: {[r['label'].split('___')[-1] for r in result['top_k']]}")


def _regnetx_predict_pil():
    from PIL import Image
    img = Image.new("RGB", (224, 224), color=(34, 139, 34))
    result = _pt_predictor.predict(img)
    assert "label" in result


def _load_regnety_160():
    p = Predictor(model_name="regnety_160_v1", verbose=False)
    result = p.predict(dummy_path)
    assert "label" in result and "confidence" in result
    print(f"         → {result['label']}  ({result['confidence']:.2%})")


def _load_regnety_320():
    p = Predictor(model_name="regnety_320_v1", verbose=False)
    result = p.predict(dummy_path)
    assert "label" in result and "confidence" in result
    print(f"         → {result['label']}  ({result['confidence']:.2%})")


def _all_regnet_same_label_count():
    for name in REGNET_MODELS:
        p = Predictor(model_name=name, verbose=False)
        labels = p.get_labels()
        assert len(labels) == 38, f"{name}: expected 38 labels, got {len(labels)}"
    print(f"         → all 3 RegNet models have 38 labels")


check("Predictor(regnetx_160_v1) loads",                    _load_regnetx)
check("regnetx_160_v1 predict() returns label+confidence",  _regnetx_predict)
check("regnetx_160_v1 predict(top_k=3) returns sorted list",_regnetx_predict_top_k)
check("regnetx_160_v1 accepts PIL Image",                   _regnetx_predict_pil)
check("regnety_160_v1 loads and predicts",                  _load_regnety_160)
check("regnety_320_v1 loads and predicts",                  _load_regnety_320)
check("all RegNet models have 38-class labels",             _all_regnet_same_label_count)

# ---------------------------------------------------------------------------
# 9. CLI (plantdoc command)
# ---------------------------------------------------------------------------
print("\n[9] CLI")

import subprocess
import shutil


def _run_cli(*args):
    return subprocess.run(
        ["plantdoc"] + list(args),
        capture_output=True,
        text=True
    )


def _cli_help():
    r = _run_cli("--help")
    assert r.returncode == 0, f"exit {r.returncode}"
    assert "predict" in r.stdout and "models" in r.stdout


def _cli_models():
    r = _run_cli("models")
    assert r.returncode == 0, f"exit {r.returncode}\n{r.stderr}"
    assert "mobilenetv2_v1" in r.stdout


def _cli_predict_path():
    r = _run_cli("predict", dummy_path, "--model", "mobilenetv2_v1")
    assert r.returncode == 0, f"exit {r.returncode}\n{r.stderr}"
    assert "mobilenetv2_v1" in r.stdout


def _cli_predict_top_k():
    r = _run_cli("predict", dummy_path, "--model", "mobilenetv2_v1", "--top-k", "3")
    assert r.returncode == 0, f"exit {r.returncode}\n{r.stderr}"
    assert "#1" in r.stdout and "#2" in r.stdout and "#3" in r.stdout


def _cli_predict_json():
    r = _run_cli("predict", dummy_path, "--model", "mobilenetv2_v1", "--json")
    assert r.returncode == 0, f"exit {r.returncode}\n{r.stderr}"
    data = json.loads(r.stdout.strip())
    assert "label" in data and "confidence" in data


def _cli_batch_folder():
    folder = tempfile.mkdtemp()
    try:
        shutil.copy(dummy_path, os.path.join(folder, "leaf1.jpg"))
        shutil.copy(dummy_path, os.path.join(folder, "leaf2.jpg"))
        r = _run_cli("predict", folder, "--model", "mobilenetv2_v1")
        assert r.returncode == 0, f"exit {r.returncode}\n{r.stderr}"
        assert "2 images" in r.stdout
    finally:
        shutil.rmtree(folder)


def _cli_batch_csv():
    folder = tempfile.mkdtemp()
    out = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
    out.close()
    try:
        shutil.copy(dummy_path, os.path.join(folder, "leaf1.jpg"))
        r = _run_cli("predict", folder, "--model", "mobilenetv2_v1", "--output", out.name)
        assert r.returncode == 0, f"exit {r.returncode}\n{r.stderr}"
        assert os.path.getsize(out.name) > 0, "CSV file is empty"
    finally:
        shutil.rmtree(folder)
        os.unlink(out.name)


def _cli_batch_json():
    folder = tempfile.mkdtemp()
    out = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
    out.close()
    try:
        shutil.copy(dummy_path, os.path.join(folder, "leaf1.jpg"))
        r = _run_cli("predict", folder, "--model", "mobilenetv2_v1", "--output", out.name)
        assert r.returncode == 0, f"exit {r.returncode}\n{r.stderr}"
        with open(out.name) as f:
            data = json.load(f)
        assert isinstance(data, list) and len(data) == 1
    finally:
        shutil.rmtree(folder)
        os.unlink(out.name)


check("plantdoc --help exits 0 and lists commands",    _cli_help)
check("plantdoc models lists mobilenetv2_v1",          _cli_models)
check("plantdoc predict <file> works",                 _cli_predict_path)
check("plantdoc predict --top-k 3 shows #1 #2 #3",    _cli_predict_top_k)
check("plantdoc predict --json outputs valid JSON",    _cli_predict_json)
check("plantdoc predict <folder> auto-batch works",    _cli_batch_folder)
check("plantdoc predict <folder> --output .csv works", _cli_batch_csv)
check("plantdoc predict <folder> --output .json works",_cli_batch_json)

# ---------------------------------------------------------------------------
# 9. Cleanup
# ---------------------------------------------------------------------------
if dummy_path and os.path.exists(dummy_path):
    os.unlink(dummy_path)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
total = len(results)
passed = sum(results)
failed = total - passed

print(f"\n{'='*40}")
print(f"  {passed}/{total} tests passed", end="")
if failed:
    print(f"  |  {failed} FAILED  <-- fix before pushing to PyPI")
else:
    print("  |  safe to publish")
print(f"{'='*40}\n")

sys.exit(0 if failed == 0 else 1)
