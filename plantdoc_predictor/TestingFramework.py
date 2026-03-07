# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 14:32:33 2026

@author: Subham Divakar
"""
from predictor import Predictor
from predictor import list_available_models
import os
import json

test_images = [
"D:/D/my docs/my docs/projects/plantdoc-predictor/plantdoc-predictor/Test/test_images/Apple___Apple_scab/0a5e9323-dbad-432d-ac58-d291718345d9___FREC_Scab 3417.JPG",
"D:/D/my docs/my docs/projects/plantdoc-predictor/plantdoc-predictor/Test/test_images/Corn_(maize)___healthy/0a68ef5a-027c-41ae-b227-159dae77d3dd___R.S_HL 7969 copy.jpg",
"D:/D/my docs/my docs/projects/plantdoc-predictor/plantdoc-predictor/Test/test_images/Orange___Haunglongbing_(Citrus_greening)/0a0e1e0f-e0d2-4a9b-9265-ec636592d0b2___CREC_HLB 7565.JPG",
"D:/D/my docs/my docs/projects/plantdoc-predictor/plantdoc-predictor/Test/test_images/Tomato___Tomato_Yellow_Leaf_Curl_Virus/0a3befdb-c654-435b-b834-d3451436afd3___YLCV_NREC 2313.JPG",
"D:/D/my docs/my docs/projects/plantdoc-predictor/plantdoc-predictor/Test/test_images/Tomato___Bacterial_spot/0a22f50a-5f25-4cf6-816b-76cae94b7f30___GCREC_Bact.Sp 6103.JPG"
]


def test_model(model_name):

    predictor = Predictor(model_name=model_name, verbose=True)

    correct = 0
    wrong = 0

    print("\n================================")
    print("Testing Model:", model_name)
    print("================================")

    for path in test_images:

        result = predictor.predict(path)

        true_label = os.path.basename(os.path.dirname(path))
        predicted_label = result["label"] if isinstance(result, dict) else result

        if predicted_label == true_label:
            correct += 1
            status = "CORRECT"
        else:
            wrong += 1
            status = "WRONG"

        print(result)
        print("True:", true_label)
        print("Pred:", predicted_label)
        print("Result:", status)
        print("--------------------------------")

    total = correct + wrong
    accuracy = (correct/total) * 100

    print("\nTEST RESULT")
    print("Model:", model_name)
    print("Total:", total)
    print("Correct:", correct)
    print("Incorrect:", wrong)
    print("Accuracy:", round(accuracy,2), "%")

    return accuracy


def get_model_list(json_path):

    with open(json_path, "r") as f:
        data = json.load(f)

    models = [model["name"] for model in data["models"]]

    return models

models = get_model_list("models/model_registry.json")

print(models)

results = {}

for model in models:
    acc = test_model(model)
    results[model] = acc
    
print("\n\nMODEL BENCHMARK")
print("-----------------------------------")

for model, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
    print(f"{model:<20} {acc:.2f}%")