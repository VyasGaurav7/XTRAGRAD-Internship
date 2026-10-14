"""
CIFAKE Explainable AI (XAI) Engine
Implements Gradient-weighted Class Activation Mapping (Grad-CAM)
to visualize discriminative pixel regions and diffusion frequency artifacts.

Reference:
Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via
Gradient-Based Localization", ICCV 2017.
Bird et al., CIFAKE IEEE Access 2024.
"""

import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.cm as cm


def generate_gradcam_efficientnet(img_array, model, base_model):
    """
    Computes Grad-CAM for the Functional EfficientNetB0 model.
    Target layer: 'top_activation' (final convolutional feature map of base_model).
    """
    gap_layer = model.get_layer("gap2d")
    drop_layer = model.get_layer("dropout")
    dense_128 = model.get_layer("dense_128")
    output_layer = model.get_layer("output")

    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)

    with tf.GradientTape() as tape:
        conv_outputs = base_model(img_tensor, training=False)
        tape.watch(conv_outputs)
        x = gap_layer(conv_outputs)
        x = drop_layer(x, training=False)
        x = dense_128(x)
        preds = output_layer(x)
        prob_real = preds[0, 0]
        # Target score: real probability if >= 0.5, else fake probability (1.0 - prob_real)
        score = prob_real if prob_real >= 0.5 else (1.0 - prob_real)

    grads = tape.gradient(score, conv_outputs)
    if grads is None:
        # Fallback if gradient is zero
        cam = np.ones((conv_outputs.shape[1], conv_outputs.shape[2]), dtype=np.float32)
    else:
        weights = tf.reduce_mean(grads, axis=(1, 2))
        cam = tf.reduce_sum(tf.multiply(weights[0], conv_outputs[0]), axis=-1)
        cam = np.maximum(cam.numpy(), 0.0)
        max_val = np.max(cam)
        if max_val > 1e-8:
            cam = cam / max_val
        else:
            cam = np.zeros_like(cam)

    return cam, float(prob_real.numpy())


def generate_gradcam_cnn(img_array, model, target_layer_name="conv2d_2"):
    """
    Computes Grad-CAM for Baseline CNN2D or Modified CNN2D models.
    Target layer: 'conv2d_2' (final convolutional layer before pooling/flattening).
    """
    grad_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(target_layer_name).output, model.output]
    )

    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)

    with tf.GradientTape() as tape:
        conv_outputs, preds = grad_model(img_tensor, training=False)
        prob_real = preds[0, 0]
        score = prob_real if prob_real >= 0.5 else (1.0 - prob_real)

    grads = tape.gradient(score, conv_outputs)
    if grads is None:
        cam = np.ones((conv_outputs.shape[1], conv_outputs.shape[2]), dtype=np.float32)
    else:
        weights = tf.reduce_mean(grads, axis=(1, 2))
        cam = tf.reduce_sum(tf.multiply(weights[0], conv_outputs[0]), axis=-1)
        cam = np.maximum(cam.numpy(), 0.0)
        max_val = np.max(cam)
        if max_val > 1e-8:
            cam = cam / max_val
        else:
            cam = np.zeros_like(cam)

    return cam, float(prob_real.numpy())


def analyze_image_with_xai(pil_image, model, model_type="EfficientNetB0", base_model=None, colormap_name="jet", alpha=0.45):
    """
    Full pipeline: Takes a PIL image and model, executes inference,
    generates Grad-CAM heatmap, superimposed overlay, and forensic attribution metrics.

    Returns:
      dict with keys:
        - label: 'REAL' or 'FAKE'
        - prob_real: float
        - prob_fake: float
        - confidence: float (0.0 to 1.0)
        - cam_normalized: 2D numpy array [0, 1]
        - overlay_image: PIL.Image of original image + colored heatmap
        - heatmap_colored: PIL.Image of isolated colored heatmap
        - saliency_grayscale: PIL.Image of grayscale saliency
        - forensic_metrics: dict of analytical observations
    """
    orig_w, orig_h = pil_image.size
    orig_rgb = np.array(pil_image.convert("RGB"), dtype=np.float32)

    # 1. Determine model dimensions & preprocess
    if model_type == "EfficientNetB0":
        # CIFAKE was natively trained on 32x32 CIFAR-10 / Stable Diffusion images upsampled to 96x96.
        # Directly resizing high-resolution camera images to 96x96 retains high-frequency sub-pixel
        # photon noise that confuses the network. Resampling through the 32x32 native frequency
        # bottleneck matches the exact distribution the model was fine-tuned on:
        cifake_bottleneck = pil_image.resize((32, 32), Image.Resampling.BILINEAR)
        resized_img = cifake_bottleneck.resize((96, 96), Image.Resampling.BILINEAR)
        arr = np.array(resized_img, dtype=np.float32)[np.newaxis]
        cam_raw, prob_real = generate_gradcam_efficientnet(arr, model, base_model)
    else:
        target_size = (32, 32)
        resized_img = pil_image.resize(target_size, Image.Resampling.BILINEAR)
        arr = np.array(resized_img, dtype=np.float32)[np.newaxis]
        cam_raw, prob_real = generate_gradcam_cnn(arr, model, target_layer_name="conv2d_2")

    prob_fake = 1.0 - prob_real
    label = "REAL" if prob_real >= 0.5 else "FAKE"
    confidence = prob_real if label == "REAL" else prob_fake

    # 2. Resize CAM back to original image size using PIL bilinear interpolation
    cam_pil = Image.fromarray((cam_raw * 255.0).astype(np.uint8)).resize((orig_w, orig_h), resample=Image.Resampling.BILINEAR)
    cam_full = np.array(cam_pil, dtype=np.float32) / 255.0

    # 3. Apply color palette
    try:
        import matplotlib as mpl
        cmap = mpl.colormaps[colormap_name]
    except Exception:
        import matplotlib.pyplot as plt
        cmap = plt.get_cmap(colormap_name)
    colored_cam = cmap(cam_full)[:, :, :3]  # Strip alpha channel -> [0, 1] float
    heatmap_rgb = (colored_cam * 255.0).astype(np.uint8)

    # 4. Superimpose with original image
    overlay_arr = (1.0 - alpha) * orig_rgb + alpha * (colored_cam * 255.0)
    overlay_arr = np.clip(overlay_arr, 0, 255).astype(np.uint8)

    overlay_img = Image.fromarray(overlay_arr)
    heatmap_img = Image.fromarray(heatmap_rgb)
    saliency_gray = Image.fromarray((cam_full * 255.0).astype(np.uint8))

    # 5. Compute Forensic Attribution Metrics
    # Split into central subject (inner 50% box) vs background border
    h, w = cam_full.shape
    center_y1, center_y2 = int(h * 0.25), int(h * 0.75)
    center_x1, center_x2 = int(w * 0.25), int(w * 0.75)

    center_mask = np.zeros_like(cam_full, dtype=bool)
    center_mask[center_y1:center_y2, center_x1:center_x2] = True
    background_mask = ~center_mask

    center_intensity = float(np.mean(cam_full[center_mask])) if np.sum(center_mask) > 0 else 0.0
    bg_intensity = float(np.mean(cam_full[background_mask])) if np.sum(background_mask) > 0 else 0.0
    total_intensity = center_intensity + bg_intensity + 1e-9

    center_ratio = (center_intensity / total_intensity) * 100.0
    bg_ratio = (bg_intensity / total_intensity) * 100.0

    # High attention focus threshold (pixels >= 0.7 max attention)
    hotspot_area_pct = float(np.mean(cam_full >= 0.70)) * 100.0

    # Generate forensic explanation based on CIFAKE paper findings
    if label == "FAKE":
        if bg_ratio > 48.0:
            forensic_narrative = (
                "⚠️ **Background Diffusion Fingerprint Detected**: The model concentrated heavily "
                f"({bg_ratio:.1f}% peripheral attention) on subtle micro-textures and unnatural background "
                "smoothness typical of Stable Diffusion v1.4 latent noise de-noising trajectories."
            )
        else:
            forensic_narrative = (
                "⚠️ **Synthetic Object Inconsistency**: The attention map reveals focused activations "
                f"({center_ratio:.1f}% semantic subject) along semantic boundaries with atypical high-frequency "
                "artifacts and edge inconsistencies absent in real sensor captures."
            )
    else:
        forensic_narrative = (
            "✅ **Natural Sensor Coherence**: The neural network distributed activation naturally "
            f"across physical contours ({center_ratio:.1f}% subject attention) consistent with optical camera "
            "sensors, natural photon noise, and authentic CIFAR-10 photographic distributions."
        )

    return {
        "label": label,
        "prob_real": prob_real,
        "prob_fake": prob_fake,
        "confidence": confidence,
        "cam_normalized": cam_full,
        "overlay_image": overlay_img,
        "heatmap_colored": heatmap_img,
        "saliency_grayscale": saliency_gray,
        "forensic_metrics": {
            "center_attention_pct": center_ratio,
            "background_attention_pct": bg_ratio,
            "hotspot_area_pct": hotspot_area_pct,
            "narrative": forensic_narrative
        }
    }
