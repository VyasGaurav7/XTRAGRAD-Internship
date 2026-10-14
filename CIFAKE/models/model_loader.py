"""
CIFAKE Model Loader & Caching Service
Provides cached model instantiation for Streamlit.
"""

import os
import sys
import streamlit as st

try:
    import tensorflow as tf
except ImportError as err:
    st.error(
        f"**TensorFlow Error**: No module named 'tensorflow' in Python environment `{sys.executable}`.\n\n"
        "**Resolution**:\n"
        "Run Streamlit using the Python interpreter where TensorFlow is installed:\n"
        "```bash\n"
        "python -m streamlit run app.py\n"
        "```\n"
        "Or install dependencies in your current environment:\n"
        "```bash\n"
        "pip install -r requirements.txt\n"
        "```"
    )
    raise ImportError(
        f"TensorFlow is not installed in the active environment ({sys.executable}). "
        "Please launch with 'python -m streamlit run app.py' or install via 'pip install tensorflow'."
    ) from err
from models.model_architectures import (
    build_baseline_cnn,
    build_modified_cnn,
    build_efficientnet,
    MODEL_BENCHMARKS
)


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
EFF_WEIGHTS_PATH = os.path.join(CURRENT_DIR, "efficientnet_weights.weights.h5")
BASE_WEIGHTS_PATH = os.path.join(CURRENT_DIR, "baseline_cnn2d_weights.weights.h5")
MOD_WEIGHTS_PATH = os.path.join(CURRENT_DIR, "modified_cnn2d_weights.weights.h5")


@st.cache_resource(show_spinner="Loading Deep Learning Model...")
def load_cifake_model(model_key="EfficientNetB0"):
    """
    Loads and caches neural network models and returns (model, base_model, model_type_str).
    """
    if model_key == "EfficientNetB0":
        model, base_model = build_efficientnet(input_shape=(96, 96, 3))
        if os.path.exists(EFF_WEIGHTS_PATH):
            model.load_weights(EFF_WEIGHTS_PATH)
        return model, base_model, "EfficientNetB0"

    elif model_key == "Modified_CNN2D":
        model = build_modified_cnn(input_shape=(32, 32, 3))
        if os.path.exists(MOD_WEIGHTS_PATH):
            model.load_weights(MOD_WEIGHTS_PATH)
        return model, None, "Modified_CNN2D"

    elif model_key == "CNN2D_Baseline":
        model = build_baseline_cnn(input_shape=(32, 32, 3))
        if os.path.exists(BASE_WEIGHTS_PATH):
            model.load_weights(BASE_WEIGHTS_PATH)
        return model, None, "CNN2D_Baseline"

    else:
        raise ValueError(f"Unknown model_key: {model_key}")


def get_model_specs(model_key):
    """
    Returns benchmark and architecture specifications for the given model key.
    """
    return MODEL_BENCHMARKS.get(model_key, MODEL_BENCHMARKS["Modified_CNN2D"])
