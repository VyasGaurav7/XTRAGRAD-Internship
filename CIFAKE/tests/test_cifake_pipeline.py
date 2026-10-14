"""
CIFAKE End-to-End Automated Test Suite
Validates:
1. Deep learning architecture construction & parameter counts
2. Pre-trained weights loading
3. Forward inference & output probability sanity
4. Grad-CAM XAI computation & heatmap extraction
5. Authentication & user management service
6. Plotly benchmark chart utilities
"""

import os
import sys
import unittest
import numpy as np
from PIL import Image
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.model_architectures import (
    build_baseline_cnn,
    build_modified_cnn,
    build_efficientnet,
    MODEL_BENCHMARKS
)
from models.model_loader import load_cifake_model, get_model_specs
from models.xai_engine import (
    generate_gradcam_cnn,
    generate_gradcam_efficientnet,
    analyze_image_with_xai
)
from auth.user_manager import register_user, authenticate_user
from utils.chart_utils import (
    create_model_comparison_chart,
    create_confusion_matrix_figure,
    create_training_curves,
    create_roc_curve,
    create_dataset_distribution_chart
)


class TestCIFAKEPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.test_img_real = Image.new("RGB", (32, 32), color=(73, 109, 137))
        cls.test_img_fake = Image.new("RGB", (32, 32), color=(200, 50, 100))

    def test_01_model_architectures_creation(self):
        """Test build functions for all 3 neural network architectures."""
        baseline = build_baseline_cnn(input_shape=(32, 32, 3))
        self.assertIsNotNone(baseline)
        self.assertEqual(baseline.output_shape, (None, 1))

        modified = build_modified_cnn(input_shape=(32, 32, 3))
        self.assertIsNotNone(modified)
        self.assertEqual(modified.output_shape, (None, 1))
        # Verify Modified CNN has fewer parameters than baseline due to GAP
        self.assertLess(modified.count_params(), baseline.count_params())

        eff_model, base_eff = build_efficientnet(input_shape=(96, 96, 3))
        self.assertIsNotNone(eff_model)
        self.assertIsNotNone(base_eff)
        self.assertEqual(eff_model.output_shape, (None, 1))

    def test_02_model_loader_and_weights(self):
        """Test model loader and pre-trained weights loading."""
        for model_key in ["Modified_CNN2D", "EfficientNetB0", "CNN2D_Baseline"]:
            model, base_m, m_type = load_cifake_model(model_key)
            self.assertIsNotNone(model)
            specs = get_model_specs(model_key)
            self.assertIn("accuracy", specs)
            self.assertGreater(specs["accuracy"], 90.0)

    def test_03_forward_inference(self):
        """Test forward prediction producing valid probability in [0, 1]."""
        model, _, _ = load_cifake_model("Modified_CNN2D")
        dummy_input = np.ones((1, 32, 32, 3), dtype=np.float32) * 128.0
        preds = model.predict(dummy_input, verbose=0)
        self.assertEqual(preds.shape, (1, 1))
        self.assertTrue(0.0 <= preds[0, 0] <= 1.0)

    def test_04_gradcam_cnn_generation(self):
        """Test Grad-CAM computation on Modified CNN2D."""
        model, _, _ = load_cifake_model("Modified_CNN2D")
        dummy_input = np.random.uniform(0, 255, (1, 32, 32, 3)).astype(np.float32)
        cam, prob = generate_gradcam_cnn(dummy_input, model, target_layer_name="conv2d_2")
        self.assertEqual(cam.ndim, 2)
        self.assertTrue(0.0 <= np.min(cam) <= 1.0)
        self.assertTrue(0.0 <= np.max(cam) <= 1.0)
        self.assertTrue(0.0 <= prob <= 1.0)

    def test_05_gradcam_efficientnet_generation(self):
        """Test Grad-CAM computation on Functional EfficientNetB0."""
        model, base_model, _ = load_cifake_model("EfficientNetB0")
        dummy_input = np.random.uniform(0, 255, (1, 96, 96, 3)).astype(np.float32)
        cam, prob = generate_gradcam_efficientnet(dummy_input, model, base_model)
        self.assertEqual(cam.ndim, 2)
        self.assertTrue(0.0 <= np.min(cam) <= 1.0)
        self.assertTrue(0.0 <= np.max(cam) <= 1.0)
        self.assertTrue(0.0 <= prob <= 1.0)

    def test_06_xai_full_analysis_pipeline(self):
        """Test end-to-end analyze_image_with_xai pipeline returns all forensic fields."""
        model, _, m_type = load_cifake_model("Modified_CNN2D")
        result = analyze_image_with_xai(
            pil_image=self.test_img_real,
            model=model,
            model_type=m_type,
            colormap_name="jet",
            alpha=0.45
        )
        required_keys = [
            "label", "prob_real", "prob_fake", "confidence",
            "cam_normalized", "overlay_image", "heatmap_colored",
            "saliency_grayscale", "forensic_metrics"
        ]
        for key in required_keys:
            self.assertIn(key, result)

        self.assertIn(result["label"], ["REAL", "FAKE"])
        self.assertAlmostEqual(result["prob_real"] + result["prob_fake"], 1.0, places=4)
        self.assertIsInstance(result["overlay_image"], Image.Image)
        self.assertIsInstance(result["heatmap_colored"], Image.Image)

        fm = result["forensic_metrics"]
        self.assertIn("center_attention_pct", fm)
        self.assertIn("background_attention_pct", fm)
        self.assertIn("hotspot_area_pct", fm)
        self.assertIn("narrative", fm)

    def test_07_auth_system(self):
        """Test registration and authentication operations."""
        test_uname = f"testuser_{int(np.random.randint(1000, 9999))}"
        # Register new
        success, msg = register_user(
            username=test_uname,
            name="Unit Tester",
            email="tester@example.com",
            mobile="+1234567890",
            password="testpassword123"
        )
        self.assertTrue(success)

        # Duplicate register should fail
        dup_success, dup_msg = register_user(
            username=test_uname,
            name="Unit Tester 2",
            email="tester2@example.com",
            mobile="+1234567890",
            password="testpassword123"
        )
        self.assertFalse(dup_success)

        # Authenticate valid
        auth_success, user = authenticate_user(test_uname, "testpassword123")
        self.assertTrue(auth_success)
        self.assertEqual(user["username"], test_uname)

        # Authenticate wrong password
        auth_bad, _ = authenticate_user(test_uname, "wrongpassword")
        self.assertFalse(auth_bad)

        # Default admin account verification
        admin_success, admin_user = authenticate_user("admin", "admin123")
        self.assertTrue(admin_success)

    def test_08_plotly_charts(self):
        """Test generation of all Plotly benchmark figures."""
        fig_comp = create_model_comparison_chart()
        self.assertIsNotNone(fig_comp)

        fig_cm = create_confusion_matrix_figure("Modified_CNN2D")
        self.assertIsNotNone(fig_cm)

        fig_train = create_training_curves()
        self.assertIsNotNone(fig_train)

        fig_roc = create_roc_curve()
        self.assertIsNotNone(fig_roc)

        fig_dist = create_dataset_distribution_chart()
        self.assertIsNotNone(fig_dist)


if __name__ == "__main__":
    unittest.main()
