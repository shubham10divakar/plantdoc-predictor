# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 12:10:38 2026

@author: Subham Divakar
"""

# -*- coding: utf-8 -*-
"""
Test file for:
- get_model()
- get_weights()
- get_weights_info()

Author: Subham Divakar
"""

from predictor import list_available_models
from predictor import Predictor
import os
import random

def get_one_random_image(dataset_dir):
    """
    Return one random image from dataset (any class).
    """

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

def test_model_and_weights():
    print("\n===== TEST: MODEL + WEIGHTS =====\n")

    model_names = list_available_models()

    success = 0
    failed = 0

    for model_name in model_names:
        print(f"\n🔹 Testing model: {model_name}")
        print("--------------------------------------")

        try:
            predictor = Predictor(model_name=model_name, verbose=False)

            # ✅ Test get_model
            model = predictor.get_model()
            assert model is not None
            print("✅ get_model() works")

            # ✅ Test get_weights
            weights = predictor.get_weights()
            assert isinstance(weights, list)
            assert len(weights) > 0
            print(f"✅ get_weights() works | Tensors: {len(weights)}")

            # ✅ Test get_weights_info
            weights_info = predictor.get_weights_info()
            assert isinstance(weights_info, dict)
            assert len(weights_info) > 0
            print(f"✅ get_weights_info() works | Layers: {len(weights_info)}")

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
        print("\n🎉 ALL TESTS PASSED")
    else:
        print("\n⚠️ SOME TESTS FAILED")


if __name__ == "__main__":
    test_model_and_weights()