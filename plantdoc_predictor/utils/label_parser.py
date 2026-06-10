def parse_label(label):
    """
    Parse a PlantVillage label string into structured fields.

    Format: 'Crop___Disease_name' or 'Crop___healthy'

    Returns
    -------
    dict with keys: crop (str), disease (str or None), is_healthy (bool)
    """
    if not label or label == "unknown":
        return {"crop": None, "disease": None, "is_healthy": None}

    if "___" not in label:
        return {"crop": label.replace("_", " "), "disease": None, "is_healthy": False}

    crop_raw, disease_raw = label.split("___", 1)
    crop = crop_raw.replace("_", " ")

    if disease_raw.lower() == "healthy":
        return {"crop": crop, "disease": None, "is_healthy": True}

    disease = disease_raw.replace("_", " ")
    return {"crop": crop, "disease": disease, "is_healthy": False}
