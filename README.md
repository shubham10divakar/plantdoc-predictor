\# 🌿 PlantDoc Predictor

# 🌿 PlantDoc-Predictor

[![PyPI version](https://badge.fury.io/py/plantdoc-predictor.svg)](https://pypi.org/project/plantdoc-predictor/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python Versions](https://img.shields.io/pypi/pyversions/plantdoc-predictor.svg)](https://pypi.org/project/plantdoc-predictor/)
[![Downloads](https://pepy.tech/badge/plantdoc-predictor)](https://pepy.tech/project/plantdoc-predictor)

---

A **Python library for predicting plant diseases** from leaf images using pre-trained or custom deep learning models.

---

## 🚀 Features

- ✅ **Pretrained models included** — Ready-to-use architectures like **InceptionV3**, **ResNet50**, and **MobileNetV2**, trained on the 38-class PlantVillage dataset.  
- 🧠 **Unified API** — One interface for both built-in and custom `.h5` models.  
- 🧩 **Custom model support** — Load your own model and label mapping JSON.  
- 🌱 **Extensible** — Easily add new crops, datasets, or models via `model_registry.json`.  
- 🧰 **Visualization support** — Displays prediction confidence and leaf images.  
- ⚙️ **Cross-platform** — Works seamlessly on Windows, macOS, and Linux.

---

## 🧠 Why Use PlantDoc-Predictor

PlantDoc-Predictor was created to **reduce repetitive work in plant-disease research**.  
Researchers and agritech developers often train from scratch — this tool eliminates that friction by providing:

- Pretrained baselines for benchmarking new models.  
- Standardized label sets and metadata for reproducibility.  
- Plug-and-play inference for agricultural image datasets.  
- A unified interface for rapid experimentation and deployment.

Whether you’re a researcher, startup, or precision-agriculture developer, PlantDoc-Predictor simplifies your workflow and lets you focus on innovation — not setup.

---

## 📦 Installation

Install directly via pip:

```bash
pip install plantdoc-predictor

## 🧩 Quick Start

```python
from plantdoc_predictor import Predictor, list_available_models

# List built-in models
list_available_models()

# Load predictor using InceptionV3
predictor = Predictor(model_name="inceptionv3", verbose=True)

# Predict a sample image
result = predictor.predict("path/to/leaf_image.jpg")
print(result)

🔄 Loading InceptionV3 model...
✅ Model loaded successfully!

================= Prediction Result =================
📂 Image Path     : path/to/leaf_image.jpg
✅ Predicted Class: Apple___Apple_scab
🔢 Confidence     : 98.42%
🏆 Top-3 Predictions:
   • Apple___Apple_scab                        → 98.42%
   • Apple___Black_rot                         → 0.97%
   • Apple___Cedar_apple_rust                  → 0.43%
=====================================================

{'model': 'inceptionv3', 'label': 'Apple___Apple_scab', 'confidence': 0.9842}

```

# Using a Custom Model
```python
You can use your own trained TensorFlow/Keras model instead of the built-in ones.
predictor = Predictor(
    model_path="models/my_custom_model.h5",
    label_path="models/my_labels.json",
    verbose=True
)

result = predictor.predict("path/to/custom_leaf.jpg", show_plot=True)
print(result)
```


✅ The library automatically detects input shape and normalizes image data.
✅ You can store your label mapping in a JSON file with the format:
```python
{
  "labels": [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy"
  ]
}
```
## 🧬 Supported Models

| Model Name | Dataset | Input Size | Accuracy | Description |
|-------------|----------|-------------|-----------|--------------|
| **InceptionV3** | PlantVillage | 299×299 | 92.8% | High-accuracy model fine-tuned for general plant disease classification. |
| **ResNet50** | PlantVillage | 224×224 | 90.3% | Residual CNN architecture with deep feature extraction. |
| **MobileNetV2** | PlantVillage | 224×224 | 89.4% | Lightweight model optimized for mobile and edge deployment. |
| **EfficientNetB0** | PlantVillage | 224×224 | 91.1% | Modern CNN architecture balancing accuracy and efficiency. |

🧩 You can easily extend the library by adding new entries to `model_registry.json`:

```json
{
  "models": [
    {
      "name": "my_custom_model",
      "path": "models/my_custom_model.h5",
      "labels": "models/my_labels.json",
      "input_size": [224, 224],
      "accuracy": 0.93,
      "description": "Custom fine-tuned model for tomato leaf disease detection"
    }
  ]
}
```

## 📝 License

This project is licensed under the **MIT License**.

You are free to:
- ✅ Use the library for both commercial and academic purposes  
- 🔧 Modify, distribute, or integrate it into your own software  
- 🌍 Reference and extend it in research or production projects  

Just ensure you include the original copyright notice and this license file.

See the full text in the [LICENSE](LICENSE) file.

---

## 🌍 Contributing

Contributions are warmly welcomed! 🌱  

We value community participation to make **PlantDoc-Predictor** more robust, accurate, and user-friendly.

You can contribute by:
- 🧠 Adding new pretrained or fine-tuned models  
- ⚙️ Improving preprocessing or postprocessing modules  
- 🧩 Extending model registry or dataset support  
- 🧪 Writing new test cases for reproducibility  
- 🐛 Fixing bugs and optimizing performance  
- 📚 Improving documentation and adding usage examples  

### 🔧 Steps to Contribute

1. **Fork** this repository  
2. **Create** your feature branch  
   ```bash
   git checkout -b feature/your-feature
   ```

## 📫 Contact

**Author:** Subham Divakar  
**Email:** subhamdivakar@gmail.com  
**GitHub:** [shubham10divakar](https://github.com/shubham10divakar)  
**PyPI:** [https://pypi.org/project/plantdoc-predictor/](https://pypi.org/project/plantdoc-predictor/)  
**LinkedIn:** [linkedin.com/in/subhamdivakar](https://linkedin.com/in/subhamdivakar)  
**Project Website:** [My Site](https://shubham10divakar.github.io/showcasehub/)

If you have any questions, collaboration ideas, or model suggestions — feel free to reach out!  
You can also open an issue or submit a pull request in the [GitHub repository](https://github.com/shubham10divakar/plantdoc-predictor).

---

## ❤️ Acknowledgements

A heartfelt thank you to all the open-source contributors and researchers who made this project possible:

- 🌿 **PlantVillage Dataset** — for providing an invaluable resource for agricultural disease detection.  
- 🤖 **TensorFlow/Keras** — for powering deep learning model training and inference.  
- 🧩 **Python Open Source Community** — for libraries that make AI tools easier to build.  
- 🧪 **Researchers & Reviewers** — for advancing the field of AI in agriculture.  
- 💻 **Contributors** — for helping test, improve, and document PlantDoc-Predictor.

Your support continues to make plant disease prediction accessible and impactful across the world.

---

> “Empowering agriculture with AI — one leaf at a time.” 🌾  
> — *Subham Divakar, Creator of PlantDoc-Predictor*




