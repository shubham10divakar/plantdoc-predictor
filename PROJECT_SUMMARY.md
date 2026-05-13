# PlantDoc Predictor — Project Summary

## What It Is

**plantdoc-predictor** is a Python library for plant disease detection from leaf images. It wraps 20+ pre-trained deep learning models (Keras/TensorFlow and PyTorch) behind a unified, research-friendly API — so you can drop in a model, run inference, extract features, or benchmark architectures without rebuilding the scaffolding each time.

- **PyPI:** `pip install plantdoc-predictor`
- **Version:** 1.0.2
- **Dataset:** PlantVillage (38 disease classes across 14 crops)
- **Models hosted on:** Hugging Face (auto-downloaded and cached in `~/.plantdoc/`)

---

## Current Features

### 1. Single-Image Prediction

Accepts a **file path or a PIL Image** — works with Streamlit, FastAPI, and any in-memory pipeline.

```python
from plantdoc_predictor import Predictor
from PIL import Image

p = Predictor(model_name="convnext_small_v1")

# File path
result = p.predict("leaf.jpg")

# PIL Image (Streamlit, FastAPI, in-memory)
pil_img = Image.open(uploaded_file)
result = p.predict(pil_img)

# {"model": "convnext_small_v1", "label": "Apple___Apple_scab", "confidence": 0.98}

# Top-K — adds a ranked 'top_k' list, top-1 keys unchanged
result = p.predict("leaf.jpg", top_k=3)
# {
#   "model": "convnext_small_v1",
#   "label": "Apple___Apple_scab",
#   "confidence": 0.98,
#   "top_k": [
#     {"label": "Apple___Apple_scab",        "confidence": 0.98},
#     {"label": "Apple___Cedar_apple_rust",   "confidence": 0.01},
#     {"label": "Apple___Black_rot",          "confidence": 0.005}
#   ]
# }
```

### 2. Pre-trained Model Registry (20 Models)

| Architecture | Best Accuracy | Notes |
|---|---|---|
| Recursive Additive Attention v1 | 99.70% | Published IEEE research, custom attention |
| DenseNet169 v1 | 99.68% | |
| ConvNeXt Small v1 | 99.50% | |
| Vision Transformer (ViT Base/Large/Small/Tiny) | ~99.1% | PyTorch/timm backend |
| Swin Transformer (Base/Tiny) | ~99.1% | PyTorch/timm backend |
| ConvNeXt (Base/Tiny) | ~99.1% | |
| DenseNet (121/210) | 98.68–99.0% | |
| VGG16 / VGG19 | 96.8–98.98% | |
| InceptionV3 | 98.2% | |
| ResNet50 | 97.8% | |
| EfficientNetB50 | 97.8% | |
| MobileNetV2 | 96.8% | Edge-optimized |
| AlexNet v1 | 92.8% | Baseline |

### 3. Custom Model Support

```python
p = Predictor(model_path="my_model.h5", label_path="labels.json")
```

### 4. Batch Processing + Export

Accepts file paths, PIL Images, or a mix of both.

```python
from plantdoc_predictor import BatchPredictor

bp = BatchPredictor(model_name="densenet169_v1")
results = bp.run(["img1.jpg", "img2.jpg", "img3.jpg"])          # top-1
results = bp.run(["img1.jpg", "img2.jpg", "img3.jpg"], top_k=3) # top-3 per image

# Mix of paths and PIL Images
pil_imgs = [Image.open(f) for f in uploaded_files]
results = bp.run(pil_imgs)

bp.export_csv(results, "results.csv")
bp.export_json(results, "results.json")
```

### 5. Model Inspection

```python
p.get_model()          # Full Keras or PyTorch model object
p.get_weights()        # List of numpy arrays
p.get_weights_info()   # Dict of layer name → weight shape
p.list_layers()        # All layer names
p.summary()            # Architecture + metadata printout
```

### 6. Feature Extraction (Intermediate Layers)

```python
features = p.extract_features("leaf.jpg", layer_name="dense_2")
# Returns numpy array of activations — ready for downstream ML
```

### 7. Dual-Framework Backend

- **Keras/TensorFlow** — classical CNNs (VGG, ResNet, DenseNet, InceptionV3, EfficientNet, MobileNet, ConvNeXt, custom attention)
- **PyTorch via timm** — Vision Transformers (ViT), Swin Transformers

### 8. Per-Model Preprocessing

Automatic preprocessing selection per model:
- Standard rescale (`/ 255`)
- EfficientNet preprocessing
- ConvNeXt preprocessing
- ImageNet normalization (PyTorch models)

### 9. Supported Classes

38 disease/health classes across 14 crops: Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, Tomato.

---

## Changelog

### v1.0.2
- Fixed bare module imports (`from pytorch_backend` → `from .pytorch_backend`, `from predictor` → `from .predictor`) that caused `ModuleNotFoundError` when installed via pip
- `__init__.py` now correctly exports `Predictor`, `BatchPredictor`, and `list_available_models`
- Added `smoke_test.py` — pre-publish test that installs the wheel in a clean environment and verifies all public APIs
- Added **top-K predictions**: `predict(img_path, top_k=3)` returns ranked predictions; works on both Keras and PyTorch backends; `BatchPredictor.run()` also accepts `top_k`
- **PIL Image support**: `predict()` and `BatchPredictor.run()` now accept `PIL.Image` objects directly alongside file paths — enables Streamlit, FastAPI, and any in-memory pipeline without saving to disk first
- **CLI** (`plantdoc` command): `plantdoc models`, `plantdoc predict leaf.jpg`, `--top-k`, `--json`, batch folder mode, `--output .csv/.json`; registered via `console_scripts` entry point in `setup.py`; `click>=8.0.0` added as dependency

---

## What Can Be Built

> Priority order based on widest real-world impact.

---

### Priority 1 — Ship Together (small, same release)

**A. Confidence Threshold / "Unknown" Detection** ← *most critical for production*
Right now any non-leaf image (a dog, a road) still returns a disease label with high confidence. Add `min_confidence` to `predict()` — if no class exceeds it, return `"unknown"`. Every real app needs this before going to users.
```python
p.predict("dog.jpg", min_confidence=0.6)
# → {"label": "unknown", "confidence": 0.31}
```

**B. Label Parsing (`crop` + `disease` fields)** ← *low effort, high value*
Labels are `Apple___Apple_scab`. Every downstream app splits this manually today. Auto-parse it into the result dict — zero new dependencies.
```python
# current
{"label": "Apple___Apple_scab", "confidence": 0.98}

# after
{"label": "Apple___Apple_scab", "crop": "Apple", "disease": "Apple scab", "is_healthy": False, "confidence": 0.98}
```

---

### Priority 2

**C. ~~Top-K Predictions~~ ✓ Done (v1.0.2)**

**D. Grad-CAM Heatmaps** ← *biggest wow factor*
The single most requested feature in any inference library. Researchers need it for papers, app developers need it for user trust. `get_model()` and `list_layers()` already expose everything needed. Output: heatmap overlaid on the original image. Works for Keras models; PyTorch support via hooks.

---

### Priority 3

**E. Feature Extraction for PyTorch Backend** ← *research gap*
Currently raises `NotImplementedError` for ViT and Swin — the best-performing models in the registry. Researchers using embeddings, clustering, or SMOTE hit a wall immediately. Implement via forward hooks on named modules.

**F. Async Batch Processing** ← *production gap*
`BatchPredictor` is sequential. Anyone processing a field survey of 500+ images will notice. `ThreadPoolExecutor` wraps the existing `run()` in ~20 lines.

---

### Further Down the Road

**G. Model Ensembling**
Average softmax outputs from 2–3 complementary models (e.g., CNN + ViT). The existing `get_model()` API already exposes what's needed.

**H. ~~CLI Tool~~ ✓ Done (v1.0.2)**

**I. Benchmark Runner**
Given a labeled test folder, evaluate all 20 models and output a comparison table (accuracy, inference time, model size). Useful for paper baselines and model selection.

**J. Confusion Matrix Generator**
Per-model confusion matrix + per-class F1 scores. Critical for identifying which disease pairs are hard to distinguish.

**K. FastAPI / Flask REST Endpoint**
Wrap the predictor in an HTTP API. Input: image file upload. Output: JSON with label, confidence, top-K predictions. PIL support is already in place.

**L. Streamlit Demo App**
Upload a leaf image, pick a model, see the prediction + Grad-CAM heatmap. PIL input support is already in place — `Image.open(uploaded_file)` passes directly to `predict()`.

**M. Mobile-Optimized Inference**
Export MobileNetV2 to TFLite or ONNX for on-device inference — no internet required for farmers in low-connectivity areas.

**N. Custom Fine-Tuning Script**
A training wrapper that takes a user's image folder (organized by class), loads a registry model, and fine-tunes the final layers. Makes the library useful for new crops or custom datasets.

**O. SMOTE / Embedding-Based Augmentation**
Use `extract_features()` to get embeddings, then apply SMOTE in feature space to oversample rare disease classes.

**P. Async Progress Callbacks**
Hook system for batch jobs: `on_image_start`, `on_image_done`, `on_error` — useful for long-running jobs and UI progress bars.

---

## Architecture at a Glance

```
plantdoc_predictor/
├── predictor.py                        ← Main API (Predictor class)
├── batch_predictor.py                  ← Batch inference + CSV/JSON export
├── pytorch_backend.py                  ← PyTorch/timm inference backend
├── recursive_additive_attention_v1.py  ← Custom attention layers (IEEE paper)
├── utils/
│   ├── preprocessing.py                ← Per-model image preprocessing
│   └── postprocessing.py               ← Output formatting
└── models/
    └── model_registry.json             ← 20 models, remote URLs, metadata
```

Models auto-download from Hugging Face on first use and are cached in `~/.plantdoc/`.
