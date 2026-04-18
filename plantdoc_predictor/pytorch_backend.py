# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 14:50:33 2026

@author: Subham Divakar
"""

import os
import json
import torch
import numpy as np
from PIL import Image
import timm
import torch.nn as nn
from torchvision import transforms


class PyTorchBackend:
    def __init__(
    self,
    model_path,
    model_name,
    label_path=None,
    input_size=(224, 224),
    preprocessing_type="vitbase16"):
        self.framework = "pytorch"
        self.model_name = model_name
        self.input_size = input_size
        self.preprocessing_type = preprocessing_type

        bundle = torch.load(
        model_path,
        map_location="cpu",
        weights_only=False)
        
        # =========================================
        # CASE 1: Bundle format (correct format)
        # =========================================
        if isinstance(bundle, dict) and "model_name" in bundle:
        
            self.model = timm.create_model(
                bundle["model_name"],
                pretrained=False
            )
        
            in_features = self.model.head.in_features
        
            self.model.head = nn.Sequential(
                nn.Linear(in_features, 512),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(512, bundle["num_classes"])
            )
        
            self.model.load_state_dict(bundle["model_state_dict"])
        
        # =========================================
        # CASE 2: Full model (DataParallel or raw)
        # =========================================
        else:
            print("⚠ Detected full model file")
        
            if isinstance(bundle, torch.nn.DataParallel):
                self.model = bundle.module
            else:
                self.model = bundle
        
        # Final step
        self.model.eval()
    
        #self.model = timm.create_model(
         #   bundle["model_name"],
          #  pretrained=False
        #)

        #in_features = self.model.head.in_features
        
        #self.model.head = nn.Sequential(
         #   nn.Linear(in_features, 512),
          #  nn.GELU(),
           # nn.Dropout(0.3),
            #nn.Linear(512, bundle["num_classes"])
        #)
        
        #self.model.load_state_dict(bundle["model_state_dict"])
        #self.model.eval()

        self.labels = []

        if label_path and os.path.exists(label_path):
            with open(label_path, "r") as f:
                data = json.load(f)

            if isinstance(data, dict) and "labels" in data:
                self.labels = data["labels"]

            elif isinstance(data, dict):
                idx_to_class = {v: k for k, v in data.items()}
                self.labels = [idx_to_class[i] for i in range(len(idx_to_class))]

    def preprocess(self, img_path):
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")

        img = Image.open(img_path).convert("RGB")

        transform = transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        return transform(img).unsqueeze(0)

    def predict(self, img_path):
        x = self.preprocess(img_path)

        with torch.no_grad():
            preds = self.model(x)
            probs = torch.softmax(preds, dim=1)

        pred_idx = torch.argmax(probs, dim=1).item()

        label = self.labels[pred_idx] if self.labels else f"Class_{pred_idx}"

        return {
            "model": self.model_name,
            "label": label,
            "confidence": probs[0][pred_idx].item()
        }

    def get_model(self):
        return self.model

    def get_weights(self):
        return self.model.state_dict()

    def get_weights_info(self):
        return {
            name: tuple(param.shape)
            for name, param in self.model.state_dict().items()
        }

    def list_layers(self):
        return [name for name, _ in self.model.named_modules()]

    def summary(self):
        print(f"\nModel: {self.model_name}")
        print(f"Framework: PyTorch")
        print(f"Input size: {self.input_size}")
        print(self.model)

    def extract_features(self, img_path, layer_name=None):
        raise NotImplementedError(
            "Feature extraction for PyTorch backend not implemented yet."
        )
        
    def preprocess(self, img_path):
        """
        Preprocess image for PyTorch model inference.
        """
    
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")
    
        img = Image.open(img_path).convert("RGB")
    
        # -----------------------------------
        # ImageNet Standard Preprocessing
        # -----------------------------------
        if self.preprocessing_type == "vit":
            transform = transforms.Compose([
                transforms.Resize(self.input_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
    
        # -----------------------------------
        # Simple Rescale Only
        # -----------------------------------
        elif self.preprocessing_type == "rescale":
            transform = transforms.Compose([
                transforms.Resize(self.input_size),
                transforms.ToTensor()
            ])
    
        # -----------------------------------
        # Add Future Custom Pipelines Here
        # -----------------------------------
        else:
            raise ValueError(
                f"Unsupported PyTorch preprocessing type: {self.preprocessing_type}"
            )
    
        return transform(img).unsqueeze(0)