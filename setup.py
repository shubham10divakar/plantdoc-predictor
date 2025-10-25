from setuptools import setup, find_packages
import io

# --- Read README.md safely (UTF-8 encoding fixes emoji/Unicode errors) ---
with io.open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="plantdoc-predictor",
    version="0.1.1",  # ⬅️ bump version (PyPI doesn’t allow re-upload of same version)
    author="Subham Divakar",
    author_email="shubham.divakar@gmail.com",
    description="A Python library for predicting plant diseases from leaf images using trained deep learning models.",
    long_description=long_description,
    long_description_content_type="text/markdown",  # ⬅️ ensures PyPI renders README properly
    url="https://github.com/shubham10divakar/plantdoc-predictor",  # ⬅️ optional but good practice
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "tensorflow>=2.10.0",
        "numpy",
        "Pillow"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
)
