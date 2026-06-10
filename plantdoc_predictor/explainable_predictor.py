"""
ExplainablePredictor — Grad-CAM heatmaps for plant disease predictions.

Wraps the existing Predictor and produces a Grad-CAM (Gradient-weighted Class
Activation Mapping) heatmap that highlights which regions of a leaf drove the
prediction. Useful for research papers, debugging, and user trust.

Existing Predictor / BatchPredictor / GuardedPredictor are completely untouched.

Currently supports Keras models only. PyTorch (ViT / Swin) models raise
NotImplementedError until forward-hook support lands in pytorch_backend.

Author: Subham Divakar
"""

import os

import numpy as np
from PIL import Image as PILImage

from .predictor import Predictor
from .utils.label_parser import parse_label


# ---------------------------------------------------------------------------
# Colormap (numpy-only jet, so we add no matplotlib dependency)
# ---------------------------------------------------------------------------

def _jet_colormap(gray):
    """
    Map a HxW float array in [0, 1] to an HxWx3 uint8 RGB image using a
    jet-like colormap. Kept dependency-free on purpose.
    """
    g = np.clip(gray, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4 * g - 3), 0, 1)
    gr = np.clip(1.5 - np.abs(4 * g - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * g - 1), 0, 1)
    rgb = np.stack([r, gr, b], axis=-1) * 255.0
    return rgb.astype(np.uint8)


# ---------------------------------------------------------------------------
# ExplainablePredictor
# ---------------------------------------------------------------------------

class ExplainablePredictor:
    """
    A Grad-CAM wrapper around Predictor (Keras backend).

    Parameters
    ----------
    model_name : str, optional
        Name of a built-in model from the registry.
    model_path : str, optional
        Path to a custom model (.h5 file).
    label_path : str, optional
        Path to a custom labels JSON.
    verbose : bool, optional
        Print prediction and Grad-CAM details. Default False.

    Example
    -------
    >>> from plantdoc_predictor import ExplainablePredictor
    >>> ep = ExplainablePredictor(model_name="densenet169_v1")
    >>> result = ep.explain("apple_leaf.jpg", save_to="heatmap.jpg")
    >>> result["label"], result["heatmap_path"]
    ('Apple___Apple_scab', 'heatmap.jpg')
    """

    def __init__(
        self,
        model_name=None,
        model_path=None,
        label_path=None,
        verbose=False,
    ):
        self.verbose = verbose
        self._predictor = Predictor(
            model_name=model_name,
            model_path=model_path,
            label_path=label_path,
            verbose=verbose,
        )

        if getattr(self._predictor, "framework", "keras") == "pytorch":
            raise NotImplementedError(
                "Grad-CAM is currently supported for Keras models only. "
                "PyTorch (ViT / Swin) support via forward hooks is on the roadmap."
            )

        self.model = self._predictor.model
        self.model_name = self._predictor.model_name
        self.input_size = self._predictor.input_size
        self.labels = self._predictor.labels

    # ------------------------------------------------------------------
    # Layer discovery
    # ------------------------------------------------------------------
    def _find_last_conv_layer(self):
        """
        Return the name of the last layer whose output is a 4D feature map
        (batch, H, W, channels) — the standard Grad-CAM target.
        """
        for layer in reversed(self.model.layers):
            try:
                shape = layer.output.shape
            except (AttributeError, RuntimeError):
                continue
            if len(shape) == 4:
                return layer.name
        raise ValueError(
            "Could not auto-detect a convolutional (4D) layer for Grad-CAM. "
            "Pass layer_name explicitly — use list_layers() to inspect options."
        )

    # ------------------------------------------------------------------
    # Heatmap computation
    # ------------------------------------------------------------------
    def _compute_heatmap(self, x, layer_name, class_index):
        """
        Core Grad-CAM. Returns (heatmap[0..1] HxW float32, class_index used).
        """
        import tensorflow as tf
        from tensorflow.keras.models import Model

        conv_layer = self.model.get_layer(layer_name)
        grad_model = Model(
            inputs=self.model.inputs,
            outputs=[conv_layer.output, self.model.output],
        )

        x_tensor = tf.convert_to_tensor(x)
        with tf.GradientTape() as tape:
            conv_out, preds = grad_model(x_tensor)
            # Flatten to a 1-D score vector (batch size is always 1 here). This is
            # robust to models whose output carries extra/singleton dims, which would
            # otherwise make tf.argmax return a non-scalar and break int().
            scores = tf.reshape(preds, [-1])
            if class_index is None:
                class_index = int(tf.argmax(scores))
            class_channel = scores[class_index]

        grads = tape.gradient(class_channel, conv_out)
        if grads is None:
            raise ValueError(
                f"No gradients flow to layer '{layer_name}'. "
                "Pick a layer that lies on the path to the model output."
            )

        # Global-average-pool the gradients → per-channel importance weights.
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        conv_out = conv_out[0]
        heatmap = conv_out @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # ReLU + normalize to [0, 1].
        heatmap = tf.maximum(heatmap, 0)
        max_val = tf.reduce_max(heatmap)
        if max_val > 0:
            heatmap = heatmap / max_val

        return heatmap.numpy().astype(np.float32), class_index

    # ------------------------------------------------------------------
    # Original image (for overlay), no model preprocessing applied
    # ------------------------------------------------------------------
    def _load_display_image(self, img_input):
        if isinstance(img_input, PILImage.Image):
            img = img_input.convert("RGB").resize(self.input_size)
        else:
            if not os.path.exists(img_input):
                raise FileNotFoundError(f"Image not found: {img_input}")
            img = PILImage.open(img_input).convert("RGB").resize(self.input_size)
        return np.array(img).astype(np.uint8)

    def _overlay(self, base_rgb, heatmap, alpha):
        """Resize heatmap to base image size, colorize, alpha-blend."""
        h, w = base_rgb.shape[:2]
        hm_img = PILImage.fromarray((heatmap * 255).astype(np.uint8)).resize(
            (w, h), resample=PILImage.BILINEAR
        )
        hm_resized = np.array(hm_img).astype(np.float32) / 255.0
        colored = _jet_colormap(hm_resized).astype(np.float32)
        blended = (1 - alpha) * base_rgb.astype(np.float32) + alpha * colored
        return np.clip(blended, 0, 255).astype(np.uint8)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def explain(
        self,
        img_input,
        save_to=None,
        layer_name=None,
        class_index=None,
        alpha=0.4,
        return_heatmap=False,
    ):
        """
        Predict and produce a Grad-CAM explanation for a leaf image.

        Parameters
        ----------
        img_input : str or PIL.Image.Image
            File path or an already-loaded PIL image.
        save_to : str, optional
            Path to write the overlaid heatmap image (PNG/JPG). If None, no
            file is written (use return_heatmap=True to get the array instead).
        layer_name : str, optional
            Target conv layer. Defaults to the last 4D feature-map layer.
        class_index : int, optional
            Class to explain. Defaults to the model's predicted class.
        alpha : float, optional
            Heatmap blend strength in [0, 1]. Default 0.4.
        return_heatmap : bool, optional
            If True, include the raw [0..1] heatmap and the overlay array in
            the result dict. Default False.

        Returns
        -------
        dict
            model, label, confidence, crop, disease, is_healthy, layer_name,
            heatmap_path (if save_to given). When return_heatmap=True also
            includes 'heatmap' (HxW float) and 'overlay' (HxWx3 uint8).
        """
        if layer_name is None:
            layer_name = self._find_last_conv_layer()

        x = self._predictor.preprocess(img_input)
        heatmap, used_index = self._compute_heatmap(x, layer_name, class_index)

        label = self.labels[used_index] if self.labels else f"Class_{used_index}"
        preds_full = np.asarray(self.model.predict(x, verbose=0)).reshape(-1)
        confidence = float(preds_full[used_index])
        parsed = parse_label(label)

        result = {
            "model": self.model_name,
            "label": label,
            "confidence": confidence,
            "crop": parsed["crop"],
            "disease": parsed["disease"],
            "is_healthy": parsed["is_healthy"],
            "layer_name": layer_name,
        }

        base_rgb = self._load_display_image(img_input)
        overlay = self._overlay(base_rgb, heatmap, alpha)

        if save_to:
            PILImage.fromarray(overlay).save(save_to)
            result["heatmap_path"] = save_to

        if return_heatmap:
            result["heatmap"] = heatmap
            result["overlay"] = overlay

        if self.verbose:
            print("\n================= Grad-CAM Explanation =================")
            print(f"🧩 Model        : {self.model_name}")
            print(f"🎯 Target layer : {layer_name}")
            print(f"✅ Class        : {label}  ({confidence * 100:.2f}%)")
            if save_to:
                print(f"🖼  Saved heatmap: {save_to}")
            print("========================================================\n")

        return result

    # ------------------------------------------------------------------
    # Pass-through utilities (delegate to inner Predictor)
    # ------------------------------------------------------------------
    def get_model(self):
        return self._predictor.get_model()

    def list_layers(self):
        return self._predictor.list_layers()

    def summary(self):
        return self._predictor.summary()
