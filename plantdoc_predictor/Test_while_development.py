# -*- coding: utf-8 -*-
"""
@author: Subham Divakar

Development test runner — tests all registered models against local test images.

SKIP_MODELS: add any model name here to skip it during a run.
RUN_ONLY:    if non-empty, only these models are run (overrides SKIP_MODELS).
"""

from predictor import Predictor, list_available_models
import os
import shutil
import random

# ---------------------------------------------------------------------------
# Configuration — edit these to control what runs
# ---------------------------------------------------------------------------

TEST_IMAGES_DIR = "D:/D/my docs/my docs/projects/plantdoc-predictor/plantdoc-predictor/Test/test_images/"
IMAGES_PER_CLASS = 2

# Add model names here to skip them
SKIP_MODELS = {
    # "alexnet_v1",
    # "mobilenetv2_v1",
}

# If non-empty, only these models are run (SKIP_MODELS is ignored)
RUN_ONLY = [
    "regnetx_160_v1",
    "regnety_160_v1",
    "regnety_320_v1",
]

ALL_MODELS = [
    "rec_add_attention_v1",
    "convnext_base_v1",
    "convnext_small_v1",
    "convnext_tiny_v1",
    "densenet169_v1",
    "densenet121_v1",
    "densenet210_v1",
    "vgg19_v1",
    "vgg16_v1",
    "inceptionv3_v1",
    "resnet50_v1",
    "efficientnetb50_v1",
    "mobilenetv2_v1",
    "alexnet_v1",
    "vit_base_16_v1",
    "vit_large_16_v1",
    "vit_small_16_v1",
    "vit_tiny_16_v1",
    "swin_base_patch4_window7",
    "swin_tiny_patch4_window7",
    "regnetx_160_v1",
    "regnety_160_v1",
    "regnety_320_v1",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_random_test_images(dataset_dir, images_per_class=2):
    test_images = []
    for class_name in os.listdir(dataset_dir):
        class_path = os.path.join(dataset_dir, class_name)
        if not os.path.isdir(class_path):
            continue
        images = [
            os.path.join(class_path, img)
            for img in os.listdir(class_path)
            if img.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        if images:
            test_images.extend(random.sample(images, min(images_per_class, len(images))))
    return test_images


def clear_cache():
    cache_dir = os.path.join(os.path.expanduser("~"), ".plantdoc")
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
        print("Cache cleared:", cache_dir)
    else:
        print("No cache found.")


def run_model_test(model_name, test_images):
    print("\n" + "=" * 50)
    print(f"Testing: {model_name}")
    print("=" * 50)

    try:
        predictor = Predictor(model_name=model_name, verbose=True)
    except Exception as e:
        print(f"FAILED to load model: {e}")
        return None

    correct = 0
    wrong = 0

    for path in test_images:
        try:
            result = predictor.predict(path)
        except Exception as e:
            print(f"  ERROR predicting {path}: {e}")
            wrong += 1
            continue

        true_label = os.path.basename(os.path.dirname(path))
        predicted_label = result["label"] if isinstance(result, dict) else result
        is_correct = predicted_label == true_label

        if is_correct:
            correct += 1
        else:
            wrong += 1

        status = "CORRECT" if is_correct else "WRONG"
        print(f"  [{status}] True: {true_label} | Pred: {predicted_label}")
        print(f"  Image: {path}")
        print("  " + "-" * 46)

    total = correct + wrong
    accuracy = (correct / total * 100) if total > 0 else 0.0

    print(f"\n  Model     : {model_name}")
    print(f"  Total     : {total}")
    print(f"  Correct   : {correct}")
    print(f"  Incorrect : {wrong}")
    print(f"  Accuracy  : {accuracy:.2f}%")

    return {"model": model_name, "total": total, "correct": correct, "wrong": wrong, "accuracy": accuracy}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

test_images = get_random_test_images(TEST_IMAGES_DIR, IMAGES_PER_CLASS)
print(f"Images loaded for testing: {len(test_images)}")

models_to_run = RUN_ONLY if RUN_ONLY else [m for m in ALL_MODELS if m not in SKIP_MODELS]

skipped = [m for m in ALL_MODELS if m not in models_to_run]
if skipped:
    print(f"Skipping: {skipped}")

summary = []
for model_name in models_to_run:
    result = run_model_test(model_name, test_images)
    if result:
        summary.append(result)

# Final summary table
if summary:
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  {'Model':<35} {'Acc':>8}  {'Correct':>7}  {'Total':>5}")
    print("  " + "-" * 56)
    for r in summary:
        print(f"  {r['model']:<35} {r['accuracy']:>7.2f}%  {r['correct']:>7}  {r['total']:>5}")
    print("=" * 60)
