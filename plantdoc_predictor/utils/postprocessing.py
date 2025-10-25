import os
import json

def format_prediction(label, confidence):
    """Formats prediction output nicely."""
    return f"Prediction: {label} (Confidence: {confidence * 100:.2f}%)"

def list_available_models():
    """Lists all available models and their label count."""
    models_dir = os.path.join(os.path.dirname(__file__), "../models")
    models = []
    for file in os.listdir(models_dir):
        if file.endswith(".h5"):
            name = file.replace("_default.h5", "")
            label_file = os.path.join(models_dir, f"{name}_labels.json")
            num_labels = 0
            if os.path.exists(label_file):
                with open(label_file) as f:
                    num_labels = len(json.load(f)["labels"])
            models.append({"model": name, "num_labels": num_labels})
    return models
