# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 11:37:39 2026

@author: Subham Divakar
"""

# -*- coding: utf-8 -*-
"""
Test file for validating model loading and get_model() feature

Author: Subham Divakar
"""
from predictor import Predictor
from predictor import list_available_models


def test_model_loading_and_return():
    """
    Test:
    1. Model loads correctly
    2. get_model() returns valid model object
    """

    print("\n===== MODEL LOADING TEST STARTED =====\n")

    model_names = list_available_models()

    success = 0
    failed = 0

    for model_name in model_names:
        print(f"\n🔹 Testing model: {model_name}")
        print("--------------------------------------")

        try:
            # Load predictor
            predictor = Predictor(model_name=model_name, verbose=False)

            # Get model
            model = predictor.get_model()

            # Basic validation
            if model is None:
                raise ValueError("Returned model is None")

            # Optional: check layers
            num_layers = len(model.layers)

            print(f"✅ Model loaded successfully")
            print(f"📊 Number of layers: {num_layers}")

            success += 1

        except Exception as e:
            print(f"❌ Failed to load model: {model_name}")
            print(f"Error: {str(e)}")
            failed += 1

    print("\n===== TEST SUMMARY =====")
    print(f"Total Models : {len(model_names)}")
    print(f"Successful   : {success}")
    print(f"Failed       : {failed}")

    if failed == 0:
        print("\n🎉 All models loaded successfully!")
    else:
        print("\n⚠️ Some models failed to load.")


if __name__ == "__main__":
    test_model_loading_and_return()