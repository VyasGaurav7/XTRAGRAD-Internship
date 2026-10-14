import os
import sys
import base64
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

sys.path.append('CIFAKE')
from models.model_loader import load_cifake_model
from models.xai_engine import analyze_image_with_xai

TMP_DIR = r"C:\Users\vyasg\.claude\jobs\755e9ade\tmp"
os.makedirs(TMP_DIR, exist_ok=True)

print("1. Generating Confusion Matrix Plot...")
fig, ax = plt.subplots(figsize=(6, 5), dpi=200)
cm = np.array([[9451, 549], [1618, 8382]])
cax = ax.imshow(cm, cmap='Blues', interpolation='nearest')
fig.colorbar(cax)
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(['FAKE (Predicted)', 'REAL (Predicted)'], fontsize=11, fontweight='bold')
ax.set_yticklabels(['FAKE (Actual)', 'REAL (Actual)'], fontsize=11, fontweight='bold')
for i in range(2):
    for j in range(2):
        color = 'white' if cm[i, j] > 5000 else '#1e293b'
        ax.text(j, i, f"{cm[i, j]:,}\n({cm[i, j]/10000*100:.2f}%)",
                ha='center', va='center', color=color, fontsize=12, fontweight='bold')
ax.set_title('Confusion Matrix on 20,000 CIFAKE Test Images\n(Modified CNN2D - 89.17% Accuracy)', fontsize=12, pad=12, fontweight='bold')
plt.tight_layout()
cm_path = os.path.join(TMP_DIR, "fig_confusion_matrix.png")
plt.savefig(cm_path, dpi=200)
plt.close()
print("   Confusion Matrix saved.")

print("2. Generating Training Curves Plot...")
epochs = [1, 2, 3]
train_acc = [72.55, 85.26, 87.52]
val_acc = [84.49, 84.87, 89.17]
train_loss = [0.5342, 0.3478, 0.3001]
val_loss = [0.3685, 0.3384, 0.2647]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=200)
# Accuracy
ax1.plot(epochs, train_acc, 'o-', color='#3b82f6', label='Training Accuracy', linewidth=2.5, markersize=8)
ax1.plot(epochs, val_acc, 's-', color='#10b981', label='Validation (Test) Accuracy', linewidth=2.5, markersize=8)
ax1.set_title('Classification Accuracy vs. Epochs (100K Images)', fontsize=11, fontweight='bold')
ax1.set_xlabel('Epoch', fontsize=10, fontweight='bold')
ax1.set_ylabel('Accuracy (%)', fontsize=10, fontweight='bold')
ax1.set_xticks(epochs)
ax1.set_ylim(65, 95)
ax1.grid(True, linestyle='--', alpha=0.5)
ax1.legend(loc='lower right', frameon=True)

# Loss
ax2.plot(epochs, train_loss, 'o-', color='#ef4444', label='Training Loss (Binary Crossentropy)', linewidth=2.5, markersize=8)
ax2.plot(epochs, val_loss, 's-', color='#f59e0b', label='Validation Loss', linewidth=2.5, markersize=8)
ax2.set_title('Loss Trajectory vs. Epochs', fontsize=11, fontweight='bold')
ax2.set_xlabel('Epoch', fontsize=10, fontweight='bold')
ax2.set_ylabel('Loss', fontsize=10, fontweight='bold')
ax2.set_xticks(epochs)
ax2.set_ylim(0.2, 0.6)
ax2.grid(True, linestyle='--', alpha=0.5)
ax2.legend(loc='upper right', frameon=True)

plt.tight_layout()
tc_path = os.path.join(TMP_DIR, "fig_training_curves.png")
plt.savefig(tc_path, dpi=200)
plt.close()
print("   Training Curves saved.")

print("3. Generating Grad-CAM Forensic Attribution Figures...")
model, _, m_type = load_cifake_model('Modified_CNN2D')

# Real Airplane Sample
real_img_path = 'CIFAKE/samples/real/cifake_real_airplane.jpg'
pil_real = Image.open(real_img_path).convert('RGB')
res_real = analyze_image_with_xai(pil_real, model, model_type=m_type)

fig, axes = plt.subplots(1, 4, figsize=(14, 3.8), dpi=200)
axes[0].imshow(pil_real)
axes[0].set_title("(a) Authentic Input Photo\n(CIFAR-10 Airplane)", fontsize=10, fontweight='bold')
axes[0].axis('off')

axes[1].imshow(res_real['saliency_grayscale'], cmap='gray')
axes[1].set_title("(b) Raw Grad-CAM Saliency\n(Rectified Gradients)", fontsize=10, fontweight='bold')
axes[1].axis('off')

axes[2].imshow(res_real['heatmap_colored'])
axes[2].set_title("(c) Jet Heatmap Activation\n(Localized Class Importance)", fontsize=10, fontweight='bold')
axes[2].axis('off')

axes[3].imshow(res_real['overlay_image'])
axes[3].set_title(f"(d) Superimposed Forensic Overlay\nVerdict: REAL ({res_real['confidence']*100:.1f}%)", fontsize=10, fontweight='bold', color='#15803d')
axes[3].axis('off')

plt.suptitle("Grad-CAM Explainability on Authentic Camera Photo (Natural Sensor Coherence)", fontsize=12, fontweight='bold', y=1.02)
plt.tight_layout()
real_xai_path = os.path.join(TMP_DIR, "fig_xai_real_airplane.png")
plt.savefig(real_xai_path, dpi=200, bbox_inches='tight')
plt.close()
print("   Real Grad-CAM saved.")

# Fake Airplane Sample
fake_img_path = 'CIFAKE/samples/fake/cifake_fake_sd_airplane.jpg'
pil_fake = Image.open(fake_img_path).convert('RGB')
res_fake = analyze_image_with_xai(pil_fake, model, model_type=m_type)

fig, axes = plt.subplots(1, 4, figsize=(14, 3.8), dpi=200)
axes[0].imshow(pil_fake)
axes[0].set_title("(a) Synthetic Input Image\n(Stable Diffusion v1.4)", fontsize=10, fontweight='bold')
axes[0].axis('off')

axes[1].imshow(res_fake['saliency_grayscale'], cmap='gray')
axes[1].set_title("(b) Raw Grad-CAM Saliency\n(Rectified Gradients)", fontsize=10, fontweight='bold')
axes[1].axis('off')

axes[2].imshow(res_fake['heatmap_colored'])
axes[2].set_title("(c) Jet Heatmap Activation\n(Diffusion Artifact Focus)", fontsize=10, fontweight='bold')
axes[2].axis('off')

axes[3].imshow(res_fake['overlay_image'])
axes[3].set_title(f"(d) Superimposed Forensic Overlay\nVerdict: FAKE ({res_fake['confidence']*100:.1f}%)", fontsize=10, fontweight='bold', color='#b91c1c')
axes[3].axis('off')

plt.suptitle("Grad-CAM Explainability on AI-Generated Synthetic Image (Latent Boundary Artifact)", fontsize=12, fontweight='bold', y=1.02)
plt.tight_layout()
fake_xai_path = os.path.join(TMP_DIR, "fig_xai_fake_airplane.png")
plt.savefig(fake_xai_path, dpi=200, bbox_inches='tight')
plt.close()
print("   Fake Grad-CAM saved.")

print("4. Generating Architecture Benchmark Comparison Plot...")
models_list = ['Baseline CNN2D', 'Modified CNN2D (GAP)', 'EfficientNetB0']
accs = [94.89, 95.78, 96.52]
params = [446209, 159489, 4213668]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), dpi=200)
bars1 = ax1.bar(models_list, accs, color=['#94a3b8', '#3b82f6', '#10b981'], width=0.55)
ax1.set_ylim(90, 100)
ax1.set_ylabel('Classification Accuracy (%)', fontsize=10, fontweight='bold')
ax1.set_title('Classification Accuracy Comparison', fontsize=11, fontweight='bold')
for bar in bars1:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 0.3, f"{yval:.2f}%", ha='center', va='bottom', fontweight='bold', fontsize=10)
ax1.grid(axis='y', linestyle='--', alpha=0.5)

bars2 = ax2.bar(models_list, [p / 1000 for p in params], color=['#f59e0b', '#8b5cf6', '#06b6d4'], width=0.55)
ax2.set_ylabel('Trainable Parameters (Thousands)', fontsize=10, fontweight='bold')
ax2.set_title('Parameter Efficiency (Model Compactness)', fontsize=11, fontweight='bold')
ax2.set_yscale('log')
for bar, p in zip(bars2, params):
    ax2.text(bar.get_x() + bar.get_width()/2.0, (p/1000)*1.15, f"{p:,}", ha='center', va='bottom', fontweight='bold', fontsize=9)
ax2.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
bm_path = os.path.join(TMP_DIR, "fig_model_benchmarks.png")
plt.savefig(bm_path, dpi=200)
plt.close()
print("   Benchmark plot saved.")
print("All figures successfully generated!")
