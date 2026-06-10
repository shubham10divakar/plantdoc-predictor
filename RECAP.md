# PlantDoc Predictor — Full Build Recap

## What It Is

`plantdoc-predictor` is a pip-installable Python library for plant disease detection from leaf images.  
It wraps 20+ pre-trained deep learning models behind a unified, research-friendly API.

```bash
pip install plantdoc-predictor
```

- **PyPI:** https://pypi.org/project/plantdoc-predictor/
- **Author:** Subham Divakar (shubham.divakar@gmail.com)
- **Dataset:** PlantVillage — 38 disease/health classes across 14 crops
- **Model hosting:** Hugging Face — auto-downloaded and cached in `~/.plantdoc/`

---

## Repository Structure

```
plantdoc_predictor/
├── predictor.py                        ← Main API (Predictor class, Keras/TF)
├── batch_predictor.py                  ← Batch inference + CSV/JSON export
├── pytorch_backend.py                  ← PyTorch/timm backend (ViT, Swin)
├── guarded_predictor.py                ← CLIP-gated guard mode (v1.0.3)
├── recursive_additive_attention_v1.py  ← Custom attention layers (IEEE paper)
├── cli.py                              ← CLI via click
├── utils/
│   ├── preprocessing.py                ← Per-model image preprocessing
│   ├── postprocessing.py               ← Output formatting
│   └── label_parser.py                 ← Parses 'Crop___Disease' labels (v1.0.3)
└── models/
    └── model_registry.json             ← 20 models with remote URLs and metadata
```

---

## Model Zoo — 20 Pre-trained Models

| Rank | Model | Accuracy | Framework | Input |
|------|-------|----------|-----------|-------|
| 1 | Recursive Additive Attention v1 | **99.70%** | Keras | 224×224 |
| 2 | DenseNet169 v1 | **99.68%** | Keras | 224×224 |
| 3 | ConvNeXt Small v1 | **99.50%** | Keras | 224×224 |
| 4 | ConvNeXt Tiny v1 | 99.23% | Keras | 224×224 |
| 5–11 | ViT Base/Large/Small/Tiny, Swin Base/Tiny, ConvNeXt Base | ~99.1% | PyTorch/timm | 224×224 |
| 12 | VGG19 v1 | 98.98% | Keras | 224×224 |
| 13 | DenseNet121 v1 | 98.68% | Keras | 224×224 |
| 14 | InceptionV3 v1 | 98.20% | Keras | 299×299 |
| 15 | ResNet50 v1 | 97.80% | Keras | 224×224 |
| 16 | EfficientNetB50 v1 | 97.80% | Keras | 224×224 |
| 17 | DenseNet210 v1 | 97.00% | Keras | 224×224 |
| 18 | MobileNetV2 v1 | 96.80% | Keras | 224×224 |
| 19 | VGG16 v1 | 96.80% | Keras | 224×224 |
| 20 | AlexNet v1 | 92.80% | Keras | 224×224 |

**Recursive Additive Attention v1** is from a peer-reviewed IEEE paper:  
Subham Divakar, Rojalina Priyadarshini — https://ieeexplore.ieee.org/abstract/document/10958327

---

## Version History

### v1.0.0 — Initial Release
- `Predictor` class with unified API for Keras and PyTorch models
- 20-model registry downloaded from Hugging Face
- Custom model support (`model_path` + `label_path`)
- Model inspection: `get_model()`, `get_weights()`, `get_weights_info()`, `list_layers()`, `summary()`
- Feature extraction from intermediate layers (`extract_features()`) — Keras only
- Per-model preprocessing (EfficientNet, ConvNeXt, ImageNet normalization, rescale)
- 38-class PlantVillage label support

### v1.0.1
- Library import fix

### v1.0.2
- Fixed bare module imports (`from pytorch_backend` → `from .pytorch_backend`) causing `ModuleNotFoundError` on pip install
- `__init__.py` correctly exports `Predictor`, `BatchPredictor`, `list_available_models`
- **Top-K predictions:** `predict(img, top_k=3)` returns ranked list; works on both Keras and PyTorch backends
- **PIL Image input:** `predict()` and `BatchPredictor.run()` now accept `PIL.Image` objects directly — enables Streamlit, FastAPI, in-memory pipelines without saving to disk
- **BatchPredictor:** batch inference over lists of paths or PIL Images, with `export_csv()` and `export_json()`
- **CLI** (`plantdoc` command): `plantdoc models`, `plantdoc predict leaf.jpg`, `--top-k`, `--json`, batch folder mode, `--output .csv/.json`; registered via `console_scripts` in `setup.py`; `click>=8.0.0` added as dependency
- `smoke_test.py` — pre-publish test that installs the wheel in a clean environment

### v1.0.3
- **`GuardedPredictor`** — new class, zero changes to existing `Predictor`/`BatchPredictor`
  - Layer 1: CLIP (`openai/clip-vit-base-patch32`) text-image similarity guard
  - Layer 2: optional `min_confidence` floor on disease model output
  - Result always includes `is_leaf`, `guard_score`, `crop`, `disease`, `is_healthy`
- **Label parsing** (`utils/label_parser.py`): auto-splits `Apple___Apple_scab` → structured fields
- `transformers>=4.30.0` added to `install_requires`

---

## Feature Reference

### 1. Single-Image Prediction

```python
from plantdoc_predictor import Predictor

p = Predictor(model_name="densenet169_v1")

# File path
result = p.predict("leaf.jpg")
# {"model": "densenet169_v1", "label": "Apple___Apple_scab", "confidence": 0.98}

# PIL Image (Streamlit, FastAPI, in-memory)
from PIL import Image
result = p.predict(Image.open("leaf.jpg"))

# Top-K
result = p.predict("leaf.jpg", top_k=3)
# {"model": ..., "label": ..., "confidence": ...,
#  "top_k": [{"label": ..., "confidence": ...}, ...]}

# verbose=True prints formatted result to console
p = Predictor(model_name="densenet169_v1", verbose=True)
```

### 2. Custom Model

```python
p = Predictor(model_path="my_model.h5", label_path="labels.json")
```

### 3. Batch Processing

```python
from plantdoc_predictor import BatchPredictor

bp = BatchPredictor(model_name="densenet169_v1")
results = bp.run(["img1.jpg", "img2.jpg"])
results = bp.run(pil_images, top_k=3)

bp.export_csv(results, "results.csv")
bp.export_json(results, "results.json")
bp.summary(results)
```

### 4. Model Inspection

```python
p.get_model()          # Full Keras or PyTorch model object
p.get_weights()        # List of numpy arrays (Keras) / state_dict (PyTorch)
p.get_weights_info()   # Dict of layer name → weight shapes
p.list_layers()        # All layer names
p.summary()            # Architecture + metadata printout
```

### 5. Feature Extraction (Keras models only)

```python
features = p.extract_features("leaf.jpg", layer_name="dense_2")
# Returns numpy array — ready for SMOTE, clustering, embedding analysis
```

### 6. List Models

```python
from plantdoc_predictor import list_available_models
list_available_models()
```

### 7. CLI

```bash
plantdoc models
plantdoc predict leaf.jpg
plantdoc predict leaf.jpg --model densenet169_v1 --top-k 3 --json
plantdoc predict ./images_folder/ --output results.csv
```

### 8. GuardedPredictor (v1.0.3) — CLIP Leaf Guard

The problem: the disease model is a closed-world 38-class classifier. Feed it a dog photo and it still returns a confident disease label. `GuardedPredictor` adds a two-layer rejection system.

```python
from plantdoc_predictor import GuardedPredictor

gp = GuardedPredictor(model_name="densenet169_v1")

# Non-leaf image → rejected
gp.predict("dog.jpg")
# {
#   "model": "densenet169_v1",
#   "is_leaf": False,
#   "guard_score": 0.13,       ← CLIP leaf probability
#   "label": "unknown",
#   "confidence": None,
#   "crop": None,
#   "disease": None,
#   "is_healthy": None
# }

# Leaf image → passes guard, gets parsed result
gp.predict("apple_scab.jpg")
# {
#   "model": "densenet169_v1",
#   "is_leaf": True,
#   "guard_score": 0.88,
#   "label": "Apple___Apple_scab",
#   "confidence": 0.98,
#   "crop": "Apple",
#   "disease": "Apple scab",
#   "is_healthy": False
# }

# Healthy leaf
gp.predict("blueberry_healthy.jpg")
# {"crop": "Blueberry", "disease": None, "is_healthy": True, ...}
```

**Parameters:**
- `guard_threshold` (default `0.5`) — CLIP leaf-score cutoff. Below = rejected.
- `min_confidence` (default `0.0` = disabled) — disease model confidence floor. Below = `"unknown"` even if CLIP passes.
- `verbose` — prints guard score and parsed output.

**How the CLIP guard works:**
- Uses `openai/clip-vit-base-patch32` via HuggingFace Transformers
- 4 leaf prompts: `"a photo of a plant leaf"`, `"a close-up of a green leaf"`, `"a diseased plant leaf"`, `"a healthy crop leaf"`
- 6 non-leaf prompts: animal, person, food, vehicle, landscape, random object
- Softmax over all 10 prompts → sums leaf prompt probabilities → `guard_score`
- CLIP lazy-loads on first `predict()` call — importing `GuardedPredictor` is instant
- ~400MB one-time download, cached in `~/.cache/huggingface/`

**Label parsing (always included in GuardedPredictor output):**
- `Apple___Apple_scab` → `crop="Apple"`, `disease="Apple scab"`, `is_healthy=False`
- `Blueberry___healthy` → `crop="Blueberry"`, `disease=None`, `is_healthy=True`

---

## Dependencies

```
numpy>=1.21.0
Pillow>=9.0.0
requests>=2.25.0
tqdm>=4.60.0
click>=8.0.0
torch>=2.0.0
torchvision>=0.15.0
timm>=0.9.0
transformers>=4.30.0    ← added v1.0.3 for CLIP guard
tensorflow>=2.10.0
```

---

## Roadmap — What's Next

### High Priority
- **Grad-CAM heatmaps** — overlay gradient-weighted class activation maps on the input image; critical for research papers and user trust; `get_model()` and `list_layers()` already expose everything needed
- **Feature extraction for PyTorch backend** — currently raises `NotImplementedError` for ViT and Swin; implement via forward hooks on named modules
- **Async batch processing** — `BatchPredictor` is sequential today; `ThreadPoolExecutor` wraps existing `run()` in ~20 lines

### Medium Priority
- **Benchmark runner** — given a labeled test folder, evaluate all 20 models and output accuracy + inference time comparison table
- **Confusion matrix generator** — per-model confusion matrix + per-class F1 scores

### Further Out
- **Model ensembling** — average softmax from 2–3 complementary models (CNN + ViT)
- **FastAPI / Flask REST endpoint** — image upload → JSON prediction
- **Streamlit demo app** — upload leaf, pick model, see prediction + Grad-CAM
- **TFLite / ONNX export** — on-device inference for MobileNetV2
- **Custom fine-tuning script** — load registry model, fine-tune final layers on user's folder
- **SMOTE in feature space** — use `extract_features()` embeddings to oversample rare disease classes
