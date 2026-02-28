# -*- coding: utf-8 -*-
"""
Created on Sat Feb 28 14:59:29 2026

@author: Subham Divakar
"""

from predictor import Predictor
from predictor import list_available_models

def efficientnetb50_v1_test():
    ## efficientnetb50_v1 Test 1.1
    predictor = Predictor(model_name="efficientnetb50_v1", verbose=True)
    result = predictor.predict("D:/D/my docs/my docs/projects/plantdoc-predictor/plantdoc-predictor/Test/test_images/Apple___Apple_scab/0a5e9323-dbad-432d-ac58-d291718345d9___FREC_Scab 3417.JPG")
    print(result)

    ## efficientnetb50_v1 Test 1.2
    predictor = Predictor(model_name="efficientnetb50_v1")
    result = predictor.predict("D:/D/my docs/my docs/projects/plantdoc-predictor/plantdoc-predictor/Test/test_images/Corn_(maize)___healthy/0a68ef5a-027c-41ae-b227-159dae77d3dd___R.S_HL 7969 copy.jpg")
    print(result)

    ## efficientnetb50_v1 Test 1.3
    predictor = Predictor(model_name="efficientnetb50_v1")
    result = predictor.predict("D:/D/my docs/my docs/projects/plantdoc-predictor/plantdoc-predictor/Test/test_images/Apple___Apple_scab/0a5e9323-dbad-432d-ac58-d291718345d9___FREC_Scab 3417.JPG")
    print(result)

    ## efficientnetb50_v1 Test 1.4
    predictor = Predictor(model_name="efficientnetb50_v1")
    result = predictor.predict("D:/D/my docs/my docs/projects/plantdoc-predictor/plantdoc-predictor/Test/test_images/Apple___Apple_scab/0a5e9323-dbad-432d-ac58-d291718345d9___FREC_Scab 3417.JPG")
    print(result)

def inceptionv3_v1_test():
    ## Inceptionv3_v1 Test 1.1
    predictor = Predictor(model_name="inceptionv3_v1", verbose=True)
    result = predictor.predict("D:/D/my docs/my docs/projects/plantdoc-predictor/plantdoc-predictor/Test/test_images/Apple___Apple_scab/0a5e9323-dbad-432d-ac58-d291718345d9___FREC_Scab 3417.JPG")
    print(result)

    ## Inceptionv3_v1 Test 1.2
    predictor = Predictor(model_name="inceptionv3_v1")
    result = predictor.predict("D:/D/my docs/my docs/projects/plantdoc-predictor/plantdoc-predictor/Test/test_images/Corn_(maize)___healthy/0a68ef5a-027c-41ae-b227-159dae77d3dd___R.S_HL 7969 copy.jpg")
    print(result)

    ## Inceptionv3_v1 Test 1.3
    predictor = Predictor(model_name="inceptionv3_v1")
    result = predictor.predict("D:/D/my docs/my docs/projects/plantdoc-predictor/plantdoc-predictor/Test/test_images/Apple___Apple_scab/0a5e9323-dbad-432d-ac58-d291718345d9___FREC_Scab 3417.JPG")
    print(result)

    ## Inceptionv3_v1 Test 1.4
    predictor = Predictor(model_name="inceptionv3_v1")
    result = predictor.predict("D:/D/my docs/my docs/projects/plantdoc-predictor/plantdoc-predictor/Test/test_images/Apple___Apple_scab/0a5e9323-dbad-432d-ac58-d291718345d9___FREC_Scab 3417.JPG")
    print(result)
    
def resnet50_v1_test():
    ## Inceptionv3_v1 Test 1.1
    predictor = Predictor(model_name="resnet50_v1", verbose=True)
    result = predictor.predict("D:/D/my docs/my docs/projects/plantdoc-predictor/plantdoc-predictor/Test/test_images/Apple___Apple_scab/0a5e9323-dbad-432d-ac58-d291718345d9___FREC_Scab 3417.JPG")
    print(result)

    ## Inceptionv3_v1 Test 1.2
    predictor = Predictor(model_name="resnet50_v1")
    result = predictor.predict("D:/D/my docs/my docs/projects/plantdoc-predictor/plantdoc-predictor/Test/test_images/Corn_(maize)___healthy/0a68ef5a-027c-41ae-b227-159dae77d3dd___R.S_HL 7969 copy.jpg")
    print(result)

    ## Inceptionv3_v1 Test 1.3
    predictor = Predictor(model_name="resnet50_v1")
    result = predictor.predict("D:/D/my docs/my docs/projects/plantdoc-predictor/plantdoc-predictor/Test/test_images/Apple___Apple_scab/0a5e9323-dbad-432d-ac58-d291718345d9___FREC_Scab 3417.JPG")
    print(result)

    ## Inceptionv3_v1 Test 1.4
    predictor = Predictor(model_name="resnet50_v1")
    result = predictor.predict("D:/D/my docs/my docs/projects/plantdoc-predictor/plantdoc-predictor/Test/test_images/Tomato___Bacterial_spot/0a22f50a-5f25-4cf6-816b-76cae94b7f30___GCREC_Bact.Sp 6103.JPG")
    print(result)
    
def mobilenetv2_v1_test():
    ## mobilenetv2_v1 Test 1.1
    predictor = Predictor(model_name="mobilenetv2_v1", verbose=True)
    result = predictor.predict("D:/D/my docs/my docs/projects/plantdoc-predictor/plantdoc-predictor/Test/test_images/Apple___Apple_scab/0a5e9323-dbad-432d-ac58-d291718345d9___FREC_Scab 3417.JPG")
    print(result)

    ## mobilenetv2_v1 Test 1.2
    predictor = Predictor(model_name="mobilenetv2_v1")
    result = predictor.predict("D:/D/my docs/my docs/projects/plantdoc-predictor/plantdoc-predictor/Test/test_images/Corn_(maize)___healthy/0a68ef5a-027c-41ae-b227-159dae77d3dd___R.S_HL 7969 copy.jpg")
    print(result)

    ## mobilenetv2_v1 Test 1.3
    predictor = Predictor(model_name="mobilenetv2_v1")
    result = predictor.predict("D:/D/my docs/my docs/projects/plantdoc-predictor/plantdoc-predictor/Test/test_images/Orange___Haunglongbing_(Citrus_greening)/0a0e1e0f-e0d2-4a9b-9265-ec636592d0b2___CREC_HLB 7565.JPG")
    print(result)

    ## mobilenetv2_v1 Test 1.4
    predictor = Predictor(model_name="mobilenetv2_v1")
    result = predictor.predict("D:/D/my docs/my docs/projects/plantdoc-predictor/plantdoc-predictor/Test/test_images/Tomato___Tomato_Yellow_Leaf_Curl_Virus/0a3befdb-c654-435b-b834-d3451436afd3___YLCV_NREC 2313.JPG")
    print(result)
    
    ## mobilenetv2_v1 Test 1.4
    predictor = Predictor(model_name="mobilenetv2_v1")
    result = predictor.predict("D:/D/my docs/my docs/projects/plantdoc-predictor/plantdoc-predictor/Test/test_images/Tomato___Bacterial_spot/0a22f50a-5f25-4cf6-816b-76cae94b7f30___GCREC_Bact.Sp 6103.JPG")
    print(result)
    
    
#list_available_models()
#inceptionv3_v1_test()
#efficientnetb50_v1_test()
#resnet50_v1_test()
mobilenetv2_v1_test()