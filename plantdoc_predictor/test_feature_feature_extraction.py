# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 12:16:51 2026

@author: Subham Divakar
"""

# -*- coding: utf-8 -*-
"""
Feature Extraction Test File

Tests:
- extract_features()
- list_layers()
- Works across ALL models

Author: Subham Divakar
"""

import sys
import os
import random

# Fix import path (adjust if needed)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from predictor import Predictor, list_available_models


# ---------------------------------------------------------
# Utility: Get ONE random image from dataset
# ---------------------------------------------------------
def get_one_random_image(dataset_dir):

    all_images = []

    for class_name in os.listdir(dataset_dir):

        class_path = os.path.join(dataset_dir, class_name)

        if not os.path.isdir(class_path):
            continue

        images = [
            os.path.join(class_path, img)
            for img in os.listdir(class_path)
            if img.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        all_images.extend(images)

    if len(all_images) == 0:
        raise ValueError("No images found in dataset")

    return random.choice(all_images)


# ---------------------------------------------------------
# Test Function
# ---------------------------------------------------------
def test_feature_extraction():

    print("\n===== FEATURE EXTRACTION TEST =====\n")

    dataset_path = "D:/D/my docs/my docs/projects/plantdoc-predictor/plantdoc-predictor/Test/test_images/"

    sample_image = get_one_random_image(dataset_path)

    print("📂 Using image:", sample_image)

    model_names = list_available_models()

    success = 0
    failed = 0

    for model_name in model_names:

        print(f"\n🔹 Testing Model: {model_name}")
        print("--------------------------------------")

        try:
            predictor = Predictor(model_name=model_name, verbose=False)

            # ✅ Test default feature extraction
            features = predictor.extract_features(sample_image)

            assert features is not None
            print("✅ Default feature extraction works")
            print("📊 Shape:", features.shape)

            # ✅ Test layer listing
            layers = predictor.list_layers()
            assert isinstance(layers, list)
            print(f"✅ Total layers: {len(layers)}")

            # ✅ Test extraction from specific layer
            if len(layers) > 3:
                test_layer = layers[-3]

                features_layer = predictor.extract_features(
                    sample_image,
                    layer_name=test_layer
                )

                print(f"✅ Layer '{test_layer}' extraction works")
                print("📊 Shape:", features_layer.shape)

            success += 1

        except Exception as e:
            print(f"❌ FAILED: {model_name}")
            print("Error:", str(e))
            failed += 1

    print("\n===== SUMMARY =====")
    print("Total Models:", len(model_names))
    print("Success:", success)
    print("Failed:", failed)

    if failed == 0:
        print("\n🎉 ALL FEATURE EXTRACTION TESTS PASSED")
    else:
        print("\n⚠️ SOME TESTS FAILED")


# ---------------------------------------------------------
# Run Test
# ---------------------------------------------------------
if __name__ == "__main__":
    test_feature_extraction()