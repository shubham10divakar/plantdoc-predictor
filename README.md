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

## 🧬 Supported Models

PlantDoc-Predictor includes a diverse **model zoo of pretrained CNN architectures** fine-tuned on the **PlantVillage 38-class plant disease dataset** in 0.2.1 version onwards.

These models are automatically downloaded from the remote registry when first used, making them ready for **plug-and-play inference**.

| Model Name | Input Size | Accuracy | Preprocessing | Description |
|-------------|-------------|-----------|---------------|--------------|
| **InceptionV3 v1** | 299×299 | **98.2%** | inception | InceptionV3 model fine-tuned on the PlantVillage 38-class dataset |
| **ResNet50 v1** | 224×224 | 97.8% | resnet50 | Deep residual network enabling powerful feature extraction |
| **EfficientNetB50 v1** | 224×224 | 97.8% | efficientnet | EfficientNet architecture balancing accuracy and computational efficiency |
| **MobileNetV2 v1** | 224×224 | 96.8% | mobilenetv2 | Lightweight model designed for mobile and edge deployment |
| **DenseNet121 v1** | 224×224 | 98.68% | densenet121_v1 | Dense connectivity architecture improving gradient flow and feature reuse |
| **DenseNet169 v1** | 224×224 | **99.68%** | densenet169_v1 | High-performance DenseNet variant achieving the best accuracy in the model zoo |
| **DenseNet210 v1** | 224×224 | 97.0% | densenet210_v1 | Very deep DenseNet architecture for advanced feature extraction |
| **VGG16 v1** | 224×224 | 96.8% | vgg16_v1 | Classic deep CNN architecture useful for benchmarking experiments |
| **VGG19 v1** | 224×224 | 98.98% | vgg19_v1 | Deeper VGG architecture providing strong classification performance |
| **AlexNet v1** | 224×224 | 92.8% | alexnet_v1 | Early CNN architecture useful as a historical baseline model |

---

### 🚀 Upcoming Models

PlantDoc-Predictor is actively expanding its **Model Zoo**.  
Future releases will include **modern vision architectures** and **transformer-based models** to further improve performance and research capabilities.

Some of the upcoming models planned for integration include:

- **Vision Transformers (ViT)** — e.g., `vit_base_patch16_224`, `vit_large_patch16_224`
- **ConvNeXt architectures** — e.g., `convnext_tiny`, `convnext_small`, `convnext_base`
- **Hybrid CNN–Transformer models**
- **Swin Transformers** — e.g., `swin_tiny_patch4_window7_224`
- **EfficientNetV2 family**
- **Multimodal models for plant disease detection that I presented via many conferences.**

Planned release very month and many more features to get added as well. 

These additions will allow researchers and developers to experiment with **state-of-the-art deep learning architectures for plant disease classification**.

Stay tuned for future releases as the **PlantDoc Model Zoo continues to grow. 🌿**

---

## 🌍 Community & Contact

PlantDoc-Predictor is growing into a **widely used open-source machine learning library for plant disease classification**.  
Community participation is essential to help expand the **PlantDoc Model Zoo** and improve the ecosystem.

### 📈 Current Usage

The library is actively used by researchers, developers, and agritech enthusiasts worldwide.

![Downloads](https://pepy.tech/badge/plantdoc-predictor)

These downloads reflect the growing interest in **AI-powered plant disease detection** and encourage further development of the library.

---

### 🤝 Contribute to the Project

We warmly welcome contributions from the community to help improve and expand **PlantDoc-Predictor**.

You can contribute by:

- 🧠 Adding new pretrained models to the **PlantDoc Model Zoo**
- ⚙️ Improving preprocessing or inference pipelines
- 📊 Benchmarking new architectures
- 🧪 Adding test cases and improving reproducibility
- 📚 Improving documentation and tutorials
- 🐛 Fixing bugs and optimizing performance

If you are interested in contributing to the development of this **widely used ML library**, feel free to open a pull request or start a discussion.

---

### 📫 Contact the Author

If you have ideas, questions, research collaboration proposals, or model suggestions, feel free to reach out.

**Author:** Subham Divakar  
📧 **Email:** shubham.divakar@gmail.com  
🐙 **GitHub:** https://github.com/shubham10divakar

We welcome **researchers, ML engineers, and contributors** who want to help make **PlantDoc-Predictor the largest open-source model zoo for plant disease detection. 🌿**

---

<!-- ### 🌿 Available Architectures -->

<!-- --- -->

<!-- ## 🧬 Supported Models

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
``` -->

---

# 📊 Model Zoo Comparison

The following table compares the pretrained models included in **PlantDoc-Predictor** .

| Model | Parameters | Input Size | Accuracy | Best Use Case |
|------|-------------|-------------|-----------|---------------|
| InceptionV3 v1 | ~23M | 299×299 | **98.2%** | High accuracy classification |
| ResNet50 v1 | ~25M | 224×224 | 97.8% | Deep feature extraction |
| EfficientNetB50 v1 | ~30M | 224×224 | 97.8% | Balanced accuracy & efficiency |
| MobileNetV2 v1 | ~3.5M | 224×224 | 96.8% | Edge / mobile deployment |
| DenseNet121 v1 | ~8M | 224×224 | 98.68% | Efficient deep CNN |
| DenseNet169 v1 | ~14M | 224×224 | **99.68%** | Highest accuracy |
| DenseNet210 v1 | ~20M | 224×224 | 97.0% | Deep dense feature learning |
| VGG16 v1 | ~138M | 224×224 | 96.8% | Baseline CNN benchmark |
| VGG19 v1 | ~144M | 224×224 | 98.98% | Deeper VGG architecture |
| AlexNet v1 | ~60M | 224×224 | 92.8% | Historical CNN baseline |

---

# 🏆 Model Performance Leaderboard

Top performing models on the **PlantVillage 38-class dataset**.

| Rank | Model | Accuracy |
|-----|------|-----------|
| 🥇 | DenseNet169 v1 | **99.68%** |
| 🥈 | VGG19 v1 | **98.98%** |
| 🥉 | DenseNet121 v1 | **98.68%** |
| 4 | InceptionV3 v1 | 98.2% |
| 5 | ResNet50 v1 | 97.8% |
| 6 | EfficientNetB50 v1 | 97.8% |
| 7 | DenseNet210 v1 | 97.0% |
| 8 | MobileNetV2 v1 | 96.8% |
| 9 | VGG16 v1 | 96.8% |
| 10 | AlexNet v1 | 92.8% |


---

## 📦 Installation

Install directly via pip:

```bash
pip install plantdoc-predictor
```

## 🚀 How to Use `plantdoc_predictor`

`plantdoc_predictor` provides an easy-to-use interface for plant disease prediction using multiple pretrained deep learning models trained on the **PlantVillage 38-class dataset**.

---

## 1️⃣ List Available Models

You can view all available pretrained models using:

```python
from plantdoc_predictor import predictor, Predictor

predictor.list_available_models()

Output:-

Available Models:
-----------------
- inceptionv3_v1      | Input: [299, 299] | Acc: 98.20% | InceptionV3 model fine-tuned on PlantVillage 38-class dataset
- resnet50_v1         | Input: [224, 224] | Acc: 97.80% | ResNet50 fine-tuned on PlantVillage 38-class dataset
- efficientnetb50_v1  | Input: [224, 224] | Acc: 97.80% | EfficientNetB50 fine-tuned on PlantVillage 38-class dataset
- mobilenetv2_v1      | Input: [224, 224] | Acc: 96.80% | Lightweight model for edge/mobile deployment
- densenet121_v1      | Input: [224, 224] | Acc: 98.68% | DenseNet121 fine-tuned on PlantVillage 38-class dataset
- densenet169_v1      | Input: [224, 224] | Acc: 99.68% | DenseNet169 fine-tuned on PlantVillage 38-class dataset
- densenet210_v1      | Input: [224, 224] | Acc: 97.00% | DenseNet210 fine-tuned on PlantVillage 38-class dataset
- vgg16_v1            | Input: [224, 224] | Acc: 96.80% | VGG16 fine-tuned on PlantVillage 38-class dataset
- vgg19_v1            | Input: [224, 224] | Acc: 98.98% | VGG19 fine-tuned on PlantVillage 38-class dataset
- alexnet_v1          | Input: [224, 224] | Acc: 92.80% | AlexNet fine-tuned on PlantVillage 38-class dataset
```
## 2. Choose a model from the available ones and use it below.

```
from plantdoc_predictor import predictor, Predictor
#predictor.list_available_models()

predictor = Predictor(model_name="efficientnetb50_v1", verbose=False)
result = predictor.predict("D:/D/my docs/my docs/projects/plant disease detection on streamlit cloud/streamlit plant disease detection/data/plantvillagedataset/train/color/Blueberry___healthy/0a3f8b2f-9bb1-4da9-85a1-fb5a52c059e2___RS_HL 2478.JPG")
print(result)

Output:-
{
 'model': 'efficientnetb50_v1',
 'label': 'Blueberry___healthy',
 'confidence': 0.9999573230743408
}

```

when verbose=True in Predictor(model_name="efficientnetb50_v1", verbose=True)

```
from plantdoc_predictor import predictor, Predictor
#predictor.list_available_models()

predictor = Predictor(model_name="efficientnetb50_v1", verbose=True)
result = predictor.predict("D:/D/my docs/my docs/projects/plant disease detection on streamlit cloud/streamlit plant disease detection/data/plantvillagedataset/train/color/Blueberry___healthy/0a3f8b2f-9bb1-4da9-85a1-fb5a52c059e2___RS_HL 2478.JPG")
print(result)

OUTPUT:-

✔ Using cached file: efficientnetb50_v1.h5
✔ Using cached file: efficientnetb50_v1_labels.json
WARNING:absl:Compiled the loaded model, but the compiled metrics have yet to be built. `model.compile_metrics` will be empty until you train or evaluate the model.
1/1 ━━━━━━━━━━━━━━━━━━━━ 2s 2s/step

================= Prediction Result =================
📂 Image Path     : D:/D/my docs/my docs/projects/plant disease detection on streamlit cloud/streamlit plant disease detection/data/plantvillagedataset/train/color/Blueberry___healthy/0a3f8b2f-9bb1-4da9-85a1-fb5a52c059e2___RS_HL 2478.JPG
🧩 Model Used     : efficientnetb50_v1
✅ Predicted Class: Blueberry___healthy
🔢 Confidence     : 100.00%
{'model': 'efficientnetb50_v1', 'label': 'Blueberry___healthy', 'confidence': 0.9999573230743408}
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
**Email:** shubham.divakar@gmail.com  
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




