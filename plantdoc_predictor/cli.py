"""
PlantDoc CLI — Command-line interface for plantdoc-predictor.

Usage:
    plantdoc models
    plantdoc predict leaf.jpg
    plantdoc predict leaf.jpg --model densenet169_v1 --top-k 3
    plantdoc predict leaf.jpg --json
    plantdoc predict ./images/ --output results.csv
"""

import csv
import json
import os
import sys
import click


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bar(conf, width=8):
    return "█" * round(conf * width)


def _color(conf):
    if conf >= 0.80:
        return "green"
    elif conf >= 0.50:
        return "yellow"
    return "red"


def _divider():
    click.echo("  " + "─" * 57)


def _export(results, output_path):
    ext = os.path.splitext(output_path)[1].lower()

    if ext == ".json":
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

    elif ext == ".csv":
        if not results:
            return
        fieldnames = [k for k in results[0].keys() if k != "top_k"]
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(results)

    else:
        click.echo(click.style(f"  Unknown format '{ext}'. Use .csv or .json.", fg="yellow"))


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------

@click.group()
@click.version_option(package_name="plantdoc-predictor")
def cli():
    """PlantDoc Predictor — plant disease detection from leaf images."""
    pass


# ---------------------------------------------------------------------------
# plantdoc models
# ---------------------------------------------------------------------------

@cli.command("models")
def models_cmd():
    """List all available pre-trained models."""
    from plantdoc_predictor.predictor import load_model_registry

    registry = load_model_registry()

    click.echo()
    click.echo(f"  Available Models ({len(registry)})")
    _divider()
    click.echo(f"  {'NAME':<28} {'FRAMEWORK':<10} {'INPUT':<10} ACCURACY")
    _divider()

    for m in registry:
        acc = f"{m.get('accuracy', 0) * 100:.2f}%" if "accuracy" in m else "N/A"
        size = f"{m['input_size'][0]}x{m['input_size'][1]}"
        fw = m.get("framework", "keras")
        click.echo(f"  {m['name']:<28} {fw:<10} {size:<10} {acc}")

    _divider()
    click.echo()


# ---------------------------------------------------------------------------
# plantdoc predict
# ---------------------------------------------------------------------------

@cli.command("predict")
@click.argument("input_path")
@click.option("--model",  "-m", default="mobilenetv2_v1", show_default=True, help="Model name from registry.")
@click.option("--top-k", "-k", default=1, show_default=True, type=int,       help="Number of top predictions.")
@click.option("--output", "-o", default=None,                                 help="Save batch results (.csv or .json).")
@click.option("--json",  "as_json", is_flag=True,                             help="Print result as JSON (single image).")
def predict_cmd(input_path, model, top_k, output, as_json):
    """Predict disease from IMAGE path or FOLDER (auto-detects batch mode)."""
    if os.path.isdir(input_path):
        _batch(input_path, model, top_k, output)
    else:
        _single(input_path, model, top_k, as_json)


# ---------------------------------------------------------------------------
# Single-image prediction
# ---------------------------------------------------------------------------

def _single(input_path, model_name, top_k, as_json):
    from plantdoc_predictor import Predictor

    try:
        p = Predictor(model_name=model_name, verbose=False)
        result = p.predict(input_path, top_k=max(top_k, 1))
    except Exception as e:
        click.echo(click.style(f"  Error: {e}", fg="red"), err=True)
        sys.exit(1)

    if as_json:
        click.echo(json.dumps(result))
        return

    preds = result.get(
        "top_k",
        [{"label": result["label"], "confidence": result["confidence"]}]
    )

    click.echo()
    click.echo(f"  Model  : {result['model']}")
    click.echo(f"  Image  : {input_path}")
    _divider()

    for i, pred in enumerate(preds, 1):
        conf = pred["confidence"]
        click.echo(
            f"  #{i}  {pred['label']:<45} "
            + click.style(f"{conf * 100:5.1f}%", fg=_color(conf))
            + f"  {_bar(conf)}"
        )

    _divider()
    click.echo()


# ---------------------------------------------------------------------------
# Batch prediction (folder)
# ---------------------------------------------------------------------------

def _batch(input_path, model_name, top_k, output):
    from plantdoc_predictor import Predictor

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    images = sorted([
        os.path.join(input_path, f)
        for f in os.listdir(input_path)
        if os.path.splitext(f)[1].lower() in exts
    ])

    if not images:
        click.echo(click.style("  No images found in folder.", fg="yellow"))
        return

    try:
        p = Predictor(model_name=model_name, verbose=False)
    except Exception as e:
        click.echo(click.style(f"  Error loading model: {e}", fg="red"), err=True)
        sys.exit(1)

    click.echo()
    click.echo(f"  Model  : {model_name}")
    click.echo(f"  Images : {len(images)} found")
    _divider()

    results = []
    for idx, img_path in enumerate(images, 1):
        fname = os.path.basename(img_path)
        try:
            result = p.predict(img_path, top_k=top_k)
            result["input"] = img_path
            result["error"] = None
            results.append(result)
            conf = result["confidence"]
            click.echo(
                f"  [{idx}/{len(images)}]  {fname:<28} {result['label']:<42} "
                + click.style(f"{conf * 100:5.1f}%", fg=_color(conf))
            )
        except Exception as e:
            results.append({"input": img_path, "label": None, "confidence": None, "error": str(e)})
            click.echo(
                f"  [{idx}/{len(images)}]  {fname:<28} "
                + click.style(f"ERROR: {e}", fg="red")
            )

    _divider()

    if output:
        _export(results, output)
        click.echo(click.style(f"  ✓ Done — {len(images)} images  |  Saved to {output}", fg="green"))
    else:
        click.echo(click.style(f"  ✓ Done — {len(images)} images", fg="green"))

    click.echo()
