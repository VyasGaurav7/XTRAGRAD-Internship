"""
CIFAKE Deep Learning Training Pipeline
Trains neural network architectures directly on the 120,000 CIFAKE dataset:
  - 100,000 Training Images (50,000 REAL / 50,000 FAKE)
  - 20,000 Testing Images   (10,000 REAL / 10,000 FAKE)

Supports:
  1. Modified CNN2D (Paper Proposed Model - GAP + Dropout 0.3)
  2. Baseline CNN2D (Standard Conv2D + Spatial Flatten)

Computes full classification report, confusion matrix, and runs
Explainable AI (Grad-CAM) visual diagnostics on real and synthetic images.
"""

import os
import sys
import time
import argparse
import numpy as np
from PIL import Image
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models.model_architectures import build_modified_cnn, build_baseline_cnn
from models.xai_engine import analyze_image_with_xai

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")


def get_datasets(batch_size=256, img_size=(32, 32)):
    print(f"\n📂 Loading Training Dataset from: {TRAIN_DIR}")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        labels="inferred",
        label_mode="binary",
        image_size=img_size,
        batch_size=batch_size,
        shuffle=True,
        seed=42
    )

    print(f"📂 Loading Testing Dataset from:  {TEST_DIR}")
    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        labels="inferred",
        label_mode="binary",
        image_size=img_size,
        batch_size=batch_size,
        shuffle=False
    )

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=autotune)
    test_ds = test_ds.prefetch(buffer_size=autotune)

    print(f"✅ Loaded: {train_ds.cardinality().numpy()*batch_size} train, {test_ds.cardinality().numpy()*batch_size} test (approx)")
    return train_ds, test_ds


def train_model(model_name="modified", epochs=3, batch_size=256, warm_start=True):
    train_ds, test_ds = get_datasets(batch_size=batch_size)

    if model_name == "modified":
        print("\n" + "="*70)
        print("⭐ Building Modified CNN2D (Paper Proposed Model - GAP + Dropout)")
        print("="*70)
        model = build_modified_cnn(input_shape=(32, 32, 3))
        weights_file = "modified_cnn2d_weights.weights.h5"
        m_type = "Modified_CNN2D"
    else:
        print("\n" + "="*70)
        print("🏗️ Building Baseline CNN2D (Standard Spatial Flatten)")
        print("="*70)
        model = build_baseline_cnn(input_shape=(32, 32, 3))
        weights_file = "baseline_cnn2d_weights.weights.h5"
        m_type = "CNN2D_Baseline"

    weights_path = os.path.join(MODELS_DIR, weights_file)

    if warm_start and os.path.exists(weights_path):
        print(f"🔄 Initializing from existing weights: {weights_path}")
        try:
            model.load_weights(weights_path)
            print("   Warm-start weights loaded successfully.")
        except Exception as e:
            print(f"   Could not load existing weights ({e}), training from scratch.")

    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=2,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=weights_path,
            monitor="val_accuracy",
            save_best_only=True,
            save_weights_only=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=1,
            verbose=1,
            min_lr=1e-5
        )
    ]

    print(f"\n🚀 Starting training for {epochs} epochs on 100,000 images...")
    t0 = time.time()
    history = model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=epochs,
        callbacks=callbacks
    )
    elapsed = time.time() - t0
    print(f"\n✅ Training completed in {elapsed/60:.2f} minutes ({elapsed:.1f}s).")

    # Evaluate on full 20,000 test set
    evaluate_model(model, test_ds, model_name=m_type)

    # Run check on test and sample images with XAI
    check_sample_images(model, model_type=m_type)

    return model


def evaluate_model(model, test_ds, model_name="Modified_CNN2D"):
    print("\n" + "="*70)
    print(f"📊 Quantitative Evaluation on 20,000 Test Images ({model_name})")
    print("="*70)

    eval_results = model.evaluate(test_ds, verbose=1)
    metric_names = model.metrics_names
    print("\n📈 Test Metrics:")
    for name, val in zip(metric_names, eval_results):
        print(f"   • {name:15s}: {val:.4f}")

    print("\n🔮 Generating predictions for confusion matrix and report...")
    y_preds_prob = model.predict(test_ds, verbose=1)
    y_pred = (y_preds_prob >= 0.5).astype(int).flatten()

    y_true = []
    for _, labels in test_ds:
        y_true.extend(labels.numpy().flatten())
    y_true = np.array(y_true).astype(int)

    report = classification_report(y_true, y_pred, target_names=["FAKE (AI)", "REAL (Photo)"], digits=4)
    print("\n📑 Classification Report:")
    print(report)

    cm = confusion_matrix(y_true, y_pred)
    total = np.sum(cm)
    acc = (cm[0, 0] + cm[1, 1]) / total * 100.0
    print("🎯 Confusion Matrix:")
    print(f"   True Negative  (Fake correctly identified): {cm[0, 0]:5d} ({cm[0, 0]/10000*100:.2f}%)")
    print(f"   False Positive (Fake misclassified as Real):{cm[0, 1]:5d} ({cm[0, 1]/10000*100:.2f}%)")
    print(f"   False Negative (Real misclassified as Fake):{cm[1, 0]:5d} ({cm[1, 0]/10000*100:.2f}%)")
    print(f"   True Positive  (Real correctly identified): {cm[1, 1]:5d} ({cm[1, 1]/10000*100:.2f}%)")
    print(f"   Overall Accuracy: {acc:.2f}%\n")


def check_sample_images(model, model_type="Modified_CNN2D"):
    print("="*70)
    print("🔍 Testing Inference & Grad-CAM XAI on Real Photos vs Synthetic Images")
    print("="*70)

    sample_tests = [
        ("Real Camera Photo (Cat)", "samples/real/cat_real.jpg"),
        ("Real Camera Photo (Airplane)", "samples/real/airplane_real.jpg"),
        ("Real Camera Photo (Horse)", "samples/real/horse_real.jpg"),
        ("Test Set REAL Photo #1", "data/test/REAL/0000 (2).jpg"),
        ("Test Set REAL Photo #2", "data/test/REAL/0001 (2).jpg"),
        ("AI Synthetic Image (Airplane)", "samples/fake/airplane_fake.png"),
        ("AI Synthetic Image (Cat)", "samples/fake/cat_fake.png"),
        ("AI Synthetic Image (Landscape)", "samples/fake/ai_synth_landscape.jpg"),
        ("Test Set FAKE Image #1", "data/test/FAKE/0 (2).jpg"),
        ("Test Set FAKE Image #2", "data/test/FAKE/1 (2).jpg"),
    ]

    base_dir = os.path.dirname(os.path.abspath(__file__))

    print(f"\n{'Image Description':32s} | {'Prediction':14s} | {'Confidence':10s} | {'Real Prob':10s} | {'Central Focus':13s} | {'Peripheral':10s}")
    print("-" * 102)

    for desc, rel_path in sample_tests:
        full_path = os.path.join(base_dir, rel_path)
        if not os.path.exists(full_path):
            print(f"File not found: {rel_path}")
            continue

        img = Image.open(full_path).convert("RGB")
        res = analyze_image_with_xai(
            pil_image=img,
            model=model,
            model_type=model_type,
            colormap_name="jet",
            alpha=0.45
        )

        label = res["label"]
        conf = res["confidence"] * 100.0
        prob_real = res["prob_real"] * 100.0
        c_attn = res["forensic_metrics"]["center_attention_pct"]
        b_attn = res["forensic_metrics"]["background_attention_pct"]

        pred_str = f"{label} ({'Photo' if label=='REAL' else 'AI'})"
        print(f"{desc:32s} | {pred_str:14s} | {conf:9.1f}% | {prob_real:9.1f}% | {c_attn:11.1f}% | {b_attn:8.1f}%")
        print(f"  └─ XAI Diagnostic: {res['forensic_metrics']['narrative'][:85]}...\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CIFAKE Model Training & Evaluation")
    parser.add_argument("--model", choices=["modified", "baseline", "both"], default="modified",
                        help="Which architecture to train: 'modified' (GAP), 'baseline', or 'both'")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs to train")
    parser.add_argument("--batch-size", type=int, default=256, help="Training batch size")
    parser.add_argument("--scratch", action="store_true", help="Train from scratch (no warm-start)")

    args = parser.parse_args()

    models_to_train = ["modified"] if args.model == "modified" else (["baseline"] if args.model == "baseline" else ["modified", "baseline"])

    for m in models_to_train:
        train_model(
            model_name=m,
            epochs=args.epochs,
            batch_size=args.batch_size,
            warm_start=not args.scratch
        )
