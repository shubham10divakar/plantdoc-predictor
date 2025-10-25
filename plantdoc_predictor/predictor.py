"""
PlantDoc Predictor
------------------
A unified API for predicting plant diseases from leaf images
using pre-trained or custom deep learning models.

Author: Subham Divakar
Version: 1.0.0
"""

import os
import json
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


# ---------------------------------------------------------------------
# Helper: Load Model Registry
# ---------------------------------------------------------------------
def load_model_registry():
    """Loads the model registry JSON file (with paths, label files, etc.)."""
    registry_path = os.path.join(os.path.dirname(__file__), "models", "model_registry.json")
    if not os.path.exists(registry_path):
        raise FileNotFoundError("Model registry file not found in models/ directory.")
    
    with open(registry_path, "r") as f:
        registry = json.load(f)
    
    return registry.get("models", [])


# ---------------------------------------------------------------------
# Predictor Class
# ---------------------------------------------------------------------
class Predictor:
    """
    PlantDoc Predictor — Predicts plant leaf diseases using built-in or custom models.
    
    Example:
    --------
    >>> from plantdoc_predictor import Predictor
    >>> predictor = Predictor(model_name="inceptionv3")
    >>> result = predictor.predict("leaf.jpg")
    >>> print(result)
    {'model': 'inceptionv3', 'label': 'Tomato___Late_blight', 'confidence': 0.984}
    """

    def __init__(self, model_name=None, model_path=None, label_path=None, verbose=False):
        """
        Initialize the Predictor.

        Parameters
        ----------
        model_name : str, optional
            Name of a built-in model (must exist in model_registry.json).
        model_path : str, optional
            Path to a custom-trained model (.h5 file).
        label_path : str, optional
            Path to custom labels JSON (overrides default labels).
        """
        self.registry = load_model_registry()
        self.verbose = verbose

        # Case 1 — custom model provided
        if model_path:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Custom model not found: {model_path}")
            self.model = load_model(model_path)
            self.model_name = os.path.basename(model_path)
            self.labels = []
            self.input_size = (224, 224)

            # Load custom label file (if provided)
            if label_path and os.path.exists(label_path):
                with open(label_path, "r") as f:
                    self.labels = json.load(f).get("labels", [])

        # Case 2 — load built-in model
        elif model_name:
            model_info = next((m for m in self.registry if m["name"] == model_name), None)
            if not model_info:
                raise ValueError(f"Model '{model_name}' not found in registry.")
            
            package_dir = os.path.dirname(__file__)
            model_path = os.path.join(package_dir, "models", model_info["path"])
            label_path = os.path.join(package_dir, "models", model_info["labels"])
            
            #print(model_path)

            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file missing: {model_path}")
            if not os.path.exists(label_path):
                raise FileNotFoundError(f"Label file missing: {label_path}")

            self.model = load_model(model_path)
            self.model_name = model_name
            self.input_size = tuple(model_info.get("input_size", [224, 224]))
            self.accuracy = model_info.get("accuracy", None)
            self.description = model_info.get("description", "")

            with open(label_path, "r") as f:
                self.labels = json.load(f).get("labels", [])

        else:
            raise ValueError("Please provide either model_name or model_path.")

    # -----------------------------------------------------------------
    # Image Preprocessing
    # -----------------------------------------------------------------
    def preprocess(self, img_path):
        """
        Load and preprocess an image for model inference.
        """
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")
        img = image.load_img(img_path, target_size=self.input_size)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = x / 255.0
        return x

    # -----------------------------------------------------------------
    # Prediction
    # -----------------------------------------------------------------
    def predict(self, img_path):
        """
        Predict the disease label from a given leaf image.

        Returns
        -------
        dict
            Dictionary containing:
            - model: str
            - label: str
            - confidence: float
        """
        x = self.preprocess(img_path)
        preds = self.model.predict(x)
        pred_idx = np.argmax(preds, axis=1)[0]
        label = self.labels[pred_idx] if self.labels else f"Class_{pred_idx}"
        
        if self.verbose:
            #predictions = model.predict(img_array)
            #predicted_idx = np.argmax(predictions[0])
            #predicted_class = class_names[predicted_idx]
            #confidence = float(np.max(predictions[0]) * 100)

            print("\n================= Prediction Result =================")
            print(f"📂 Image Path     : {img_path}")
            print(f"🧩 Model Used     : {self.model_name}")
            print(f"✅ Predicted Class: {label}")
            print(f"🔢 Confidence     : {float(np.max(preds))}%")
            #print("\n🏆 Top-3 Predictions:")
            
            # Top-3 predictions
            #top3_indices = predictions[0].argsort()[-3:][::-1]
            #top3 = [(class_names[i], float(predictions[0][i] * 100)) for i in top3_indices]
            
            #for label, conf in top3:
             #   print(f"   • {label:40} → {conf:.2f}%")
            #print("=========================================================\n")
            
        
        return {
            "model": self.model_name,
            "label": label,
            "confidence": float(np.max(preds))
        }

    # -----------------------------------------------------------------
    # Info and Utility Methods
    # -----------------------------------------------------------------
    def get_labels(self):
        """Return the list of class labels for the current model."""
        return self.labels

    def summary(self):
        """Print the summary of the loaded model."""
        print(f"\nModel: {self.model_name}")
        print(f"Input size: {self.input_size}")
        if hasattr(self, "accuracy"):
            print(f"Reported accuracy: {self.accuracy * 100:.2f}%")
        if hasattr(self, "description"):
            print(f"Description: {self.description}\n")
        self.model.summary()


# ---------------------------------------------------------------------
# Utility: List Available Models
# ---------------------------------------------------------------------
def list_available_models():
    """Lists all available models from the registry."""
    models = load_model_registry()
    print("\nAvailable Models:\n-----------------")
    for m in models:
        acc = f"{m.get('accuracy', 0) * 100:.2f}%" if "accuracy" in m else "N/A"
        print(f"- {m['name']:15} | Input: {m['input_size']} | Acc: {acc} | {m.get('description', '')}")
    print()
    return [m["name"] for m in models]
