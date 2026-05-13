# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 22:29:55 2026

@author: Subham Divakar
"""

# -*- coding: utf-8 -*-
"""
PlantDoc Batch Predictor
-------------------------
Standalone utility for running predictions on multiple images at once,
with optional CSV export of results.

Author: Subham Divakar
Version: 1.1.0

Usage
-----
From code:
    from batch_predictor import BatchPredictor

    bp = BatchPredictor(model_name="inceptionv3_v1")
    results = bp.run(["leaf1.jpg", "leaf2.jpg", "leaf3.jpg"])
    bp.export_csv(results, "results.csv")

From Streamlit:
    uploaded = st.file_uploader(..., accept_multiple_files=True)
    pil_images = [Image.open(f) for f in uploaded]
    results = bp.run(pil_images)
"""

import os
import csv
import json
from datetime import datetime
from PIL import Image as PILImage
from .predictor import Predictor


class BatchPredictor:
    """
    Wrapper around Predictor for batch inference.

    Parameters
    ----------
    model_name : str, optional
        Name of a built-in model from the registry.
    model_path : str, optional
        Path to a custom model file.
    label_path : str, optional
        Path to a custom labels JSON.
    verbose : bool, optional
        Print per-image results as they are processed. Default False.

    Example
    -------
    >>> bp = BatchPredictor(model_name="convnext_small_v1")
    >>> results = bp.run(["img1.jpg", "img2.jpg"])
    >>> bp.export_csv(results, "batch_results.csv")
    """

    def __init__(self, model_name=None, model_path=None, label_path=None, verbose=False):
        self.verbose = verbose
        self.predictor = Predictor(
            model_name=model_name,
            model_path=model_path,
            label_path=label_path,
            verbose=False   # suppress per-call verbosity; we handle it here
        )
        self.model_name = self.predictor.model_name

    def run(self, img_inputs, top_k=1, stop_on_error=False):
        """
        Run predictions on a list of images.

        Parameters
        ----------
        img_inputs : list of str or PIL.Image.Image
            File paths, PIL Images, or a mix of both.
        top_k : int, optional
            Number of top predictions per image (default 1).
            When top_k > 1, each result includes a 'top_k' list.
        stop_on_error : bool, optional
            If True, raises on the first failed image.
            If False (default), records the error and continues.

        Returns
        -------
        list of dict
            Each dict is one result with these keys:
                input       : str  — file path or 'PIL Image #N'
                model       : str
                label       : str
                confidence  : float
                top_k       : list (only when top_k > 1)
                error       : str or None
        """
        if not isinstance(img_inputs, (list, tuple)):
            raise TypeError("img_inputs must be a list or tuple.")

        results = []
        total = len(img_inputs)

        for idx, img_input in enumerate(img_inputs):
            label = img_input if isinstance(img_input, str) else f"PIL Image #{idx}"

            if self.verbose:
                print(f"[{idx + 1}/{total}] Processing: {label}")

            try:
                result = self.predictor.predict(img_input, top_k=top_k)
                result["input"] = label
                result["error"] = None
                results.append(result)

                if self.verbose:
                    print(
                        f"         ✅ {result['label']} "
                        f"({result['confidence'] * 100:.1f}%)"
                    )

            except Exception as e:
                if stop_on_error:
                    raise

                error_result = {
                    "input": label,
                    "model": self.model_name,
                    "label": None,
                    "crop": None,
                    "disease": None,
                    "is_healthy": None,
                    "confidence": None,
                    "top3": [],
                    "error": str(e)
                }
                results.append(error_result)

                if self.verbose:
                    print(f"         ❌ Error: {e}")

        if self.verbose:
            success = sum(1 for r in results if r["error"] is None)
            print(f"\n✔ Batch complete: {success}/{total} succeeded.")

        return results

    def export_csv(self, results, output_path=None):
        """
        Export batch results to a CSV file.

        Parameters
        ----------
        results : list of dict
            Output from run().
        output_path : str, optional
            Path for the CSV file.
            Defaults to 'plantdoc_batch_<timestamp>.csv' in the current directory.

        Returns
        -------
        str
            Absolute path to the written CSV file.
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"plantdoc_batch_{timestamp}.csv"

        fieldnames = [
            "input", "model", "label", "crop", "disease",
            "is_healthy", "confidence", "error",
            "top2_label", "top2_confidence",
            "top3_label", "top3_confidence"
        ]

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for r in results:
                top3 = r.get("top3", [])
                row = {
                    "input":          r.get("input"),
                    "model":          r.get("model"),
                    "label":          r.get("label"),
                    "crop":           r.get("crop"),
                    "disease":        r.get("disease"),
                    "is_healthy":     r.get("is_healthy"),
                    "confidence":     f"{r['confidence']:.4f}" if r.get("confidence") is not None else "",
                    "error":          r.get("error") or "",
                    "top2_label":     top3[1]["label"] if len(top3) > 1 else "",
                    "top2_confidence": f"{top3[1]['confidence']:.4f}" if len(top3) > 1 else "",
                    "top3_label":     top3[2]["label"] if len(top3) > 2 else "",
                    "top3_confidence": f"{top3[2]['confidence']:.4f}" if len(top3) > 2 else "",
                }
                writer.writerow(row)

        print(f"📄 Results saved to: {os.path.abspath(output_path)}")
        return os.path.abspath(output_path)

    def export_json(self, results, output_path=None):
        """
        Export batch results to a JSON file.

        Parameters
        ----------
        results : list of dict
            Output from run().
        output_path : str, optional
            Defaults to 'plantdoc_batch_<timestamp>.json'.

        Returns
        -------
        str
            Absolute path to the written JSON file.
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"plantdoc_batch_{timestamp}.json"

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)

        print(f"📄 Results saved to: {os.path.abspath(output_path)}")
        return os.path.abspath(output_path)

    def summary(self, results):
        """
        Print a quick summary of a batch run.

        Parameters
        ----------
        results : list of dict
            Output from run().
        """
        total = len(results)
        success = [r for r in results if r["error"] is None]
        failed = [r for r in results if r["error"] is not None]
        healthy = [r for r in success if r["is_healthy"]]

        print("\n========== Batch Summary ==========")
        print(f"Total images   : {total}")
        print(f"Succeeded      : {len(success)}")
        print(f"Failed         : {len(failed)}")
        if success:
            avg_conf = sum(r["confidence"] for r in success) / len(success)
            print(f"Avg confidence : {avg_conf * 100:.1f}%")
            print(f"Healthy plants : {len(healthy)} / {len(success)}")
        if failed:
            print("\nFailed inputs:")
            for r in failed:
                print(f"  ✗ {r['input']} → {r['error']}")
        print("===================================\n")