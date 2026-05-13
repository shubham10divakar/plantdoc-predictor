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

---

## What Can Be Built

### Immediate Extensions (Low Effort, High Value)

**A. Confidence Thresholding & "Unknown" Detection**
Add a minimum confidence threshold — if no class exceeds it, return `"unknown"` or trigger a fallback model. Useful for real-world deployment where out-of-distribution images are common.

**B. ~~Top-K Predictions~~ ✓ Done (v1.0.2)**

**C. Model Ensembling**
Average softmax outputs from 2–3 complementary models (e.g., CNN + ViT) to boost accuracy on hard cases. The existing `get_model()` API already exposes what's needed.

**D. Grad-CAM / Saliency Maps**
Visualize which regions of the leaf the model focused on. Works well with the existing layer access APIs. Output: heatmap overlaid on original image.

**E. CLI Tool**
`plantdoc predict leaf.jpg --model convnext_small_v1` — wraps the Python API for shell use, batch-folder processing, and CI pipelines.

---

### Research & Benchmarking Tools

**F. Benchmark Runner**
Given a labeled test folder, evaluate all 20 models and output a comparison table (accuracy, inference time, model size). Useful for paper baselines and model selection.

**G. Confusion Matrix Generator**
Per-model confusion matrix + per-class F1 scores. Critical for identifying which disease pairs are hard to distinguish.

**H. Cross-Model Feature Similarity**
Use the `extract_features()` API to compare intermediate representations across architectures — useful for representational similarity analysis (RSA/CKA).

---

### Application Layer

**I. FastAPI / Flask REST Endpoint**
Wrap the predictor in an HTTP API. Input: image file upload. Output: JSON with label, confidence, top-K predictions. Ready for integration into farm management apps.

**J. Streamlit Demo App**
Upload a leaf image, pick a model, see the prediction + Grad-CAM heatmap. Good for demos, paper supplements, and non-technical stakeholders. PIL input support is already in place — the app can pass `Image.open(uploaded_file)` directly to `predict()`.

**K. Mobile-Optimized Inference**
Export MobileNetV2 (already in registry) to TFLite or ONNX for on-device inference — no internet required for farmers in low-connectivity areas.

---

### Dataset & Training Utilities

**L. Custom Fine-Tuning Script**
A training wrapper that takes a user's image folder (organized by class), loads a registry model, and fine-tunes the final layers. Makes the library useful for new crops or custom datasets.

**M. Data Augmentation Pipeline**
Standard augmentation presets (flip, rotate, color jitter, cutout) as a preprocessing utility — useful for fine-tuning and for research ablations.

**N. SMOTE / Embedding-Based Augmentation**
Use `extract_features()` to get embeddings, then apply SMOTE in feature space to oversample rare disease classes. Already partially motivated by the feature extraction API.

---

### Developer Experience

**O. Async / Concurrent Batch Processing**
The current `BatchPredictor` runs sequentially. Add `asyncio` or `ThreadPoolExecutor` support for faster batch jobs on large image sets.

**P. Progress Callbacks & Logging**
Hook system for batch jobs: `on_image_start`, `on_image_done`, `on_error` — useful for long-running jobs and UI progress bars.

**Q. Model Card Generator**
Auto-generate a standardized model card (accuracy, dataset, architecture, limitations) for any registry model — useful for ML transparency requirements.

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
