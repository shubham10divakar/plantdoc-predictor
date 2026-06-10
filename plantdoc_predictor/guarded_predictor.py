"""
GuardedPredictor — CLIP-gated plant disease predictor.

Adds a two-layer guard on top of the existing Predictor:
  Layer 1 — CLIP text-image similarity: rejects images that don't look like plant leaves.
  Layer 2 — Disease model confidence threshold: rejects uncertain predictions.

Existing Predictor and BatchPredictor are completely untouched.

Author: Subham Divakar
"""

import torch
from PIL import Image as PILImage

from .predictor import Predictor
from .utils.label_parser import parse_label


# ---------------------------------------------------------------------------
# CLIP Leaf Guard
# ---------------------------------------------------------------------------

class _CLIPLeafGuard:
    """
    Scores an image 0–1 for how likely it is a plant leaf using CLIP.
    Lazy-loads the model on first call so import is instant.
    """

    _LEAF_PROMPTS = [
        "a photo of a plant leaf",
        "a close-up of a green leaf",
        "a diseased plant leaf",
        "a healthy crop leaf",
    ]

    _NON_LEAF_PROMPTS = [
        "a photo of an animal",
        "a photo of a person",
        "a photo of food",
        "a photo of a vehicle",
        "a photo of a landscape without plants",
        "a random object that is not a leaf",
    ]

    def __init__(self, verbose=False):
        self.verbose = verbose
        self._model = None
        self._processor = None

    def _load(self):
        if self._model is not None:
            return
        if self.verbose:
            print("Loading CLIP guard model (first use — one-time download ~400 MB)...")
        from transformers import CLIPModel, CLIPProcessor
        self._model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self._processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self._model.eval()
        if self.verbose:
            print("CLIP guard model ready.")

    def score(self, img_input):
        """
        Returns a float in [0, 1] — the probability mass assigned to leaf prompts.
        Higher = more likely to be a plant leaf.
        """
        self._load()

        if isinstance(img_input, PILImage.Image):
            image = img_input.convert("RGB")
        else:
            image = PILImage.open(img_input).convert("RGB")

        all_prompts = self._LEAF_PROMPTS + self._NON_LEAF_PROMPTS
        n_leaf = len(self._LEAF_PROMPTS)

        inputs = self._processor(
            text=all_prompts,
            images=image,
            return_tensors="pt",
            padding=True
        )

        with torch.no_grad():
            outputs = self._model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1)[0]

        return probs[:n_leaf].sum().item()


# ---------------------------------------------------------------------------
# GuardedPredictor
# ---------------------------------------------------------------------------

class GuardedPredictor:
    """
    A CLIP-gated wrapper around Predictor that rejects non-leaf images.

    Parameters
    ----------
    model_name : str, optional
        Name of a built-in model from the registry.
    model_path : str, optional
        Path to a custom model (.h5 file).
    label_path : str, optional
        Path to a custom labels JSON.
    guard_threshold : float, optional
        CLIP leaf-score threshold in [0, 1]. Images scoring below this are
        rejected as non-leaves. Default 0.5.
    min_confidence : float, optional
        Minimum disease-model confidence required. If the top-1 confidence
        falls below this after passing the CLIP guard, the result is returned
        as 'unknown' (but is_leaf stays True). Default 0.0 (disabled).
    verbose : bool, optional
        Print guard scores and prediction details. Default False.

    Example
    -------
    >>> from plantdoc_predictor import GuardedPredictor
    >>> gp = GuardedPredictor(model_name="densenet169_v1", guard_threshold=0.5)
    >>> gp.predict("dog.jpg")
    {'model': 'densenet169_v1', 'is_leaf': False, 'guard_score': 0.14, 'label': 'unknown', ...}
    >>> gp.predict("apple_leaf.jpg")
    {'model': 'densenet169_v1', 'is_leaf': True, 'guard_score': 0.87, 'label': 'Apple___Apple_scab',
     'confidence': 0.98, 'crop': 'Apple', 'disease': 'Apple scab', 'is_healthy': False}
    """

    def __init__(
        self,
        model_name=None,
        model_path=None,
        label_path=None,
        guard_threshold=0.5,
        min_confidence=0.0,
        verbose=False,
    ):
        self.guard_threshold = guard_threshold
        self.min_confidence = min_confidence
        self.verbose = verbose

        self._predictor = Predictor(
            model_name=model_name,
            model_path=model_path,
            label_path=label_path,
            verbose=verbose,
        )
        self.model_name = self._predictor.model_name
        self._guard = _CLIPLeafGuard(verbose=verbose)

    def predict(self, img_input, top_k=1):
        """
        Predict plant disease with CLIP-based leaf guard.

        Parameters
        ----------
        img_input : str or PIL.Image.Image
            File path or an already-loaded PIL image.
        top_k : int, optional
            Number of top disease predictions (default 1).

        Returns
        -------
        dict
            Always contains:
                model, is_leaf, guard_score, label, confidence, crop, disease, is_healthy
            When top_k > 1 and is_leaf is True:
                top_k list of {label, confidence}
        """
        guard_score = round(self._guard.score(img_input), 4)

        if self.verbose:
            print(f"Guard score: {guard_score:.4f}  (threshold: {self.guard_threshold})")

        # --- Layer 1: CLIP guard ---
        if guard_score < self.guard_threshold:
            return {
                "model": self.model_name,
                "is_leaf": False,
                "guard_score": guard_score,
                "label": "unknown",
                "confidence": None,
                "crop": None,
                "disease": None,
                "is_healthy": None,
            }

        # --- Disease model ---
        result = self._predictor.predict(img_input, top_k=top_k)

        # --- Layer 2: confidence floor ---
        if self.min_confidence > 0 and result["confidence"] < self.min_confidence:
            return {
                "model": self.model_name,
                "is_leaf": True,
                "guard_score": guard_score,
                "label": "unknown",
                "confidence": result["confidence"],
                "crop": None,
                "disease": None,
                "is_healthy": None,
            }

        # --- Label parsing ---
        parsed = parse_label(result["label"])

        result.update({
            "is_leaf": True,
            "guard_score": guard_score,
            "crop": parsed["crop"],
            "disease": parsed["disease"],
            "is_healthy": parsed["is_healthy"],
        })

        if self.verbose:
            status = "healthy" if parsed["is_healthy"] else f"disease: {parsed['disease']}"
            print(f"Crop: {parsed['crop']}  |  {status}  |  confidence: {result['confidence']:.4f}")

        return result

    # ------------------------------------------------------------------
    # Pass-through utilities (delegates to inner Predictor)
    # ------------------------------------------------------------------

    def list_available_models(self):
        from .predictor import list_available_models
        return list_available_models()

    def get_model(self):
        return self._predictor.get_model()

    def get_weights(self):
        return self._predictor.get_weights()

    def get_weights_info(self):
        return self._predictor.get_weights_info()

    def list_layers(self):
        return self._predictor.list_layers()

    def summary(self):
        return self._predictor.summary()
