# -*- coding: utf-8 -*-
"""
Created on Sat Oct 25 18:55:47 2025

@author: Subham Divakar
"""

from plantdoc_predictor import Predictor
from plantdoc_predictor.predictor import list_available_models

##Test 1
list_available_models()


## Inceptionv3_v1 Test 1.1
predictor = Predictor(model_name="inceptionv3", verbose=True)
result = predictor.predict("D:/D/my docs/my docs/projects/plantdoc-predictor/plantdoc-predictor/Test/test_images/Apple___Apple_scab/0a5e9323-dbad-432d-ac58-d291718345d9___FREC_Scab 3417.JPG")
print(result)

## Inceptionv3_v1 Test 1.2
predictor = Predictor(model_name="inceptionv3")
result = predictor.predict("D:/D/my docs/my docs/projects/plantdoc-predictor/plantdoc-predictor/Test/test_images/Corn_(maize)___healthy/0a68ef5a-027c-41ae-b227-159dae77d3dd___R.S_HL 7969 copy.jpg")
print(result)

## Inceptionv3_v1 Test 1.3
predictor = Predictor(model_name="inceptionv3")
result = predictor.predict("D:/D/my docs/my docs/projects/plantdoc-predictor/plantdoc-predictor/Test/test_images/Apple___Apple_scab/0a5e9323-dbad-432d-ac58-d291718345d9___FREC_Scab 3417.JPG")
print(result)

## Inceptionv3_v1 Test 1.4
predictor = Predictor(model_name="inceptionv3")
result = predictor.predict("D:/D/my docs/my docs/projects/plantdoc-predictor/plantdoc-predictor/Test/test_images/Apple___Apple_scab/0a5e9323-dbad-432d-ac58-d291718345d9___FREC_Scab 3417.JPG")
print(result)

