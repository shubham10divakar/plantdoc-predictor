# -*- coding: utf-8 -*-
"""
Created on Sat Feb 28 14:59:29 2026

@author: Subham Divakar
"""

from predictor import Predictor
from predictor import list_available_models
import os

test_images = [
"D:/D/my docs/my docs/projects/plantdoc-predictor/plantdoc-predictor/Test/test_images/Apple___Apple_scab/0a5e9323-dbad-432d-ac58-d291718345d9___FREC_Scab 3417.JPG",
"D:/D/my docs/my docs/projects/plantdoc-predictor/plantdoc-predictor/Test/test_images/Corn_(maize)___healthy/0a68ef5a-027c-41ae-b227-159dae77d3dd___R.S_HL 7969 copy.jpg",
"D:/D/my docs/my docs/projects/plantdoc-predictor/plantdoc-predictor/Test/test_images/Orange___Haunglongbing_(Citrus_greening)/0a0e1e0f-e0d2-4a9b-9265-ec636592d0b2___CREC_HLB 7565.JPG",
"D:/D/my docs/my docs/projects/plantdoc-predictor/plantdoc-predictor/Test/test_images/Tomato___Tomato_Yellow_Leaf_Curl_Virus/0a3befdb-c654-435b-b834-d3451436afd3___YLCV_NREC 2313.JPG",
"D:/D/my docs/my docs/projects/plantdoc-predictor/plantdoc-predictor/Test/test_images/Tomato___Bacterial_spot/0a22f50a-5f25-4cf6-816b-76cae94b7f30___GCREC_Bact.Sp 6103.JPG"
]

import os

def efficientnetb50_v1_test():

    predictor = Predictor(model_name="efficientnetb50_v1", verbose=True)

    correct = 0
    wrong = 0

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
        print("----------------------------------")

    total = correct + wrong

    print("\nTEST RESULT")
    print("Model: efficientnetb50_v1")
    print("Total:", total)
    print("Correct:", correct)
    print("Incorrect:", wrong)
    print("Accuracy:", (correct/total)*100, "%")

def inceptionv3_v1_test():

    test_images = [
    "D:/D/my docs/my docs/projects/plantdoc-predictor/plantdoc-predictor/Test/test_images/Apple___Apple_scab/0a5e9323-dbad-432d-ac58-d291718345d9___FREC_Scab 3417.JPG",
    "D:/D/my docs/my docs/projects/plantdoc-predictor/plantdoc-predictor/Test/test_images/Corn_(maize)___healthy/0a68ef5a-027c-41ae-b227-159dae77d3dd___R.S_HL 7969 copy.jpg",
    "D:/D/my docs/my docs/projects/plantdoc-predictor/plantdoc-predictor/Test/test_images/Apple___Apple_scab/0a5e9323-dbad-432d-ac58-d291718345d9___FREC_Scab 3417.JPG",
    "D:/D/my docs/my docs/projects/plantdoc-predictor/plantdoc-predictor/Test/test_images/Apple___Apple_scab/0a5e9323-dbad-432d-ac58-d291718345d9___FREC_Scab 3417.JPG"
    ]

    predictor = Predictor(model_name="inceptionv3_v1", verbose=True)

    correct = 0
    wrong = 0

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
        print("----------------------------------")

    total = correct + wrong

    print("\nTEST RESULT")
    print("Model: inceptionv3_v1")
    print("Total:", total)
    print("Correct:", correct)
    print("Incorrect:", wrong)
    print("Accuracy:", (correct/total)*100, "%")
    
def resnet50_v1_test():

    test_images = [
    "D:/D/my docs/my docs/projects/plantdoc-predictor/plantdoc-predictor/Test/test_images/Apple___Apple_scab/0a5e9323-dbad-432d-ac58-d291718345d9___FREC_Scab 3417.JPG",
    "D:/D/my docs/my docs/projects/plantdoc-predictor/plantdoc-predictor/Test/test_images/Corn_(maize)___healthy/0a68ef5a-027c-41ae-b227-159dae77d3dd___R.S_HL 7969 copy.jpg",
    "D:/D/my docs/my docs/projects/plantdoc-predictor/plantdoc-predictor/Test/test_images/Apple___Apple_scab/0a5e9323-dbad-432d-ac58-d291718345d9___FREC_Scab 3417.JPG",
    "D:/D/my docs/my docs/projects/plantdoc-predictor/plantdoc-predictor/Test/test_images/Tomato___Bacterial_spot/0a22f50a-5f25-4cf6-816b-76cae94b7f30___GCREC_Bact.Sp 6103.JPG"
    ]

    predictor = Predictor(model_name="resnet50_v1", verbose=True)

    correct = 0
    wrong = 0

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
        print("----------------------------------")

    total = correct + wrong

    print("\nTEST RESULT")
    print("Model: resnet50_v1")
    print("Total:", total)
    print("Correct:", correct)
    print("Incorrect:", wrong)
    print("Accuracy:", (correct/total)*100, "%")
    
def mobilenetv2_v1_test():

    test_images = [
    "D:/D/my docs/my docs/projects/plantdoc-predictor/plantdoc-predictor/Test/test_images/Apple___Apple_scab/0a5e9323-dbad-432d-ac58-d291718345d9___FREC_Scab 3417.JPG",
    "D:/D/my docs/my docs/projects/plantdoc-predictor/plantdoc-predictor/Test/test_images/Corn_(maize)___healthy/0a68ef5a-027c-41ae-b227-159dae77d3dd___R.S_HL 7969 copy.jpg",
    "D:/D/my docs/my docs/projects/plantdoc-predictor/plantdoc-predictor/Test/test_images/Orange___Haunglongbing_(Citrus_greening)/0a0e1e0f-e0d2-4a9b-9265-ec636592d0b2___CREC_HLB 7565.JPG",
    "D:/D/my docs/my docs/projects/plantdoc-predictor/plantdoc-predictor/Test/test_images/Tomato___Tomato_Yellow_Leaf_Curl_Virus/0a3befdb-c654-435b-b834-d3451436afd3___YLCV_NREC 2313.JPG",
    "D:/D/my docs/my docs/projects/plantdoc-predictor/plantdoc-predictor/Test/test_images/Tomato___Bacterial_spot/0a22f50a-5f25-4cf6-816b-76cae94b7f30___GCREC_Bact.Sp 6103.JPG"
    ]

    predictor = Predictor(model_name="mobilenetv2_v1", verbose=True)

    correct = 0
    wrong = 0

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
        print("----------------------------------")

    total = correct + wrong

    print("\nTEST RESULT")
    print("Model: mobilenetv2_v1")
    print("Total:", total)
    print("Correct:", correct)
    print("Incorrect:", wrong)
    print("Accuracy:", (correct/total)*100, "%")

def densenet121_v1_test():

    test_images = [
        "D:/D/my docs/my docs/projects/plantdoc-predictor/plantdoc-predictor/Test/test_images/Apple___Apple_scab/0a5e9323-dbad-432d-ac58-d291718345d9___FREC_Scab 3417.JPG",
        "D:/D/my docs/my docs/projects/plantdoc-predictor/plantdoc-predictor/Test/test_images/Corn_(maize)___healthy/0a68ef5a-027c-41ae-b227-159dae77d3dd___R.S_HL 7969 copy.jpg",
        "D:/D/my docs/my docs/projects/plantdoc-predictor/plantdoc-predictor/Test/test_images/Orange___Haunglongbing_(Citrus_greening)/0a0e1e0f-e0d2-4a9b-9265-ec636592d0b2___CREC_HLB 7565.JPG",
        "D:/D/my docs/my docs/projects/plantdoc-predictor/plantdoc-predictor/Test/test_images/Tomato___Tomato_Yellow_Leaf_Curl_Virus/0a3befdb-c654-435b-b834-d3451436afd3___YLCV_NREC 2313.JPG",
        "D:/D/my docs/my docs/projects/plantdoc-predictor/plantdoc-predictor/Test/test_images/Tomato___Bacterial_spot/0a22f50a-5f25-4cf6-816b-76cae94b7f30___GCREC_Bact.Sp 6103.JPG"
    ]

    predictor = Predictor(model_name="densenet121_v1", verbose=True)

    correct_count = 0
    wrong_count = 0

    print("\nRunning DenseNet121 Test\n")
    print("--------------------------------------------------")

    for path in test_images:

        result = predictor.predict(path)

        true_label = os.path.basename(os.path.dirname(path))
        predicted_label = result["label"] if isinstance(result, dict) else result

        is_correct = predicted_label == true_label

        if is_correct:
            correct_count += 1
        else:
            wrong_count += 1

        print("Image:", path)
        print("True Label:", true_label)
        print("Predicted:", predicted_label)
        print("Correct:", is_correct)
        print("--------------------------------------------------")

    total = correct_count + wrong_count
    accuracy = (correct_count / total) * 100 if total > 0 else 0

    print("\n========== TEST RESULT ==========")
    print("Model           : densenet121_v1")
    print("Total Tests     :", total)
    print("Correct         :", correct_count)
    print("Incorrect       :", wrong_count)
    print("Accuracy        : {:.2f}%".format(accuracy))
    print("=================================\n")
    

def densenet169_v1_test():

    predictor = Predictor(model_name="densenet169_v1", verbose=True)

    correct = 0
    wrong = 0

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

        print(path)
        print("True:", true_label)
        print("Pred:", predicted_label)
        print("Result:", status)
        print("------------------------------------------------")

    total = correct + wrong

    print("\nTEST RESULT")
    print("Model: densenet169_v1")
    print("Total:", total)
    print("Correct:", correct)
    print("Incorrect:", wrong)
    print("Accuracy:", (correct/total)*100, "%")
    
def densenet210_v1_test():

    predictor = Predictor(model_name="densenet210_v1", verbose=True)

    correct = 0
    wrong = 0

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

        print(path)
        print("True:", true_label)
        print("Pred:", predicted_label)
        print("Result:", status)
        print("------------------------------------------------")

    total = correct + wrong

    print("\nTEST RESULT")
    print("Model: densenet210_v1")
    print("Total:", total)
    print("Correct:", correct)
    print("Incorrect:", wrong)
    print("Accuracy:", (correct/total)*100, "%")
    

def vgg16_v1_test():

    predictor = Predictor(model_name="vgg16_v1", verbose=True)

    correct = 0
    wrong = 0

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

        print(path)
        print("True:", true_label)
        print("Pred:", predicted_label)
        print("Result:", status)
        print("------------------------------------------------")

    total = correct + wrong

    print("\nTEST RESULT")
    print("Model: vgg16_v1")
    print("Total:", total)
    print("Correct:", correct)
    print("Incorrect:", wrong)
    print("Accuracy:", (correct/total)*100, "%")
    
    
def vgg19_v1_test():

    predictor = Predictor(model_name="vgg19_v1", verbose=True)

    correct = 0
    wrong = 0

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

        print(path)
        print("True:", true_label)
        print("Pred:", predicted_label)
        print("Result:", status)
        print("------------------------------------------------")

    total = correct + wrong

    print("\nTEST RESULT")
    print("Model: vgg19_v1")
    print("Total:", total)
    print("Correct:", correct)
    print("Incorrect:", wrong)
    print("Accuracy:", (correct/total)*100, "%")
    
    
def alexnet_v1_test():

    predictor = Predictor(model_name="alexnet_v1", verbose=True)

    correct = 0
    wrong = 0

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

        print(path)
        print("True:", true_label)
        print("Pred:", predicted_label)
        print("Result:", status)
        print("------------------------------------------------")

    total = correct + wrong

    print("\nTEST RESULT")
    print("Model: alexnet_v1")
    print("Total:", total)
    print("Correct:", correct)
    print("Incorrect:", wrong)
    print("Accuracy:", (correct/total)*100, "%")
    
    

#list_available_models()
#inceptionv3_v1_test()
#efficientnetb50_v1_test()
#resnet50_v1_test()
#mobilenetv2_v1_test()
#densenet121_v1_test()
#densenet169_v1_test()
#densenet210_v1_test()
#vgg16_v1_test()
#vgg19_v1_test()
#alexnet_v1_test()