"""
inference_folder_analysis_dual_branch.py
-----------------------------------------
Batch-level equivariance/invariance analysis across folder of images.
Generates ECM/IDM/Correlation statistics and visualization plots.

Author: Vineet (REU)
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm


# -------------------------------
# DualBranch Model (same as training)
# -------------------------------
class DualBranch(nn.Module):
    def __init__(self, emb_dim=64):
        super().__init__()
        base = torchvision.models.resnet18(weights=None)
        self.backbone = nn.Sequential(
            base.conv1, base.bn1, base.relu, base.maxpool,
            base.layer1, base.layer2, base.layer3
        )
        self.eqv_head = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, emb_dim, 3, padding=1)
        )
        self.inv_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, emb_dim)
        )

    def forward(self, x):
        feat = self.backbone(x)
        eqv_map = self.eqv_head(feat)
        inv_vec = F.normalize(self.inv_head(feat), dim=1)
        return {"equiv_map": eqv_map, "inv_vec": inv_vec}


# -------------------------------
# Helper functions
# -------------------------------
def get_transform(size=128):
    return T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225])
    ])


def load_model(weights_path, device):
    model = DualBranch()
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict, strict=False)
    model.eval().to(device)
    print(f"✅ Loaded model: {weights_path}")
    return model


def compute_metrics(eqv_orig, eqv_trans, inv_orig, inv_trans):
    # Equivariance consistency
    ecm = torch.mean(torch.abs(eqv_orig - eqv_trans)).item()
    # Invariance consistency
    idm = torch.mean(torch.abs(inv_orig - inv_trans)).item()
    # Correlation
    corr = torch.corrcoef(torch.stack([
        eqv_orig.flatten().cpu(), eqv_trans.flatten().cpu()
    ]))[0, 1].item()
    return ecm, idm, corr


# -------------------------------
# Folder-level inference
# -------------------------------
def analyze_folder(model, folder, device, save_dir="results_folder_eqv"):
    os.makedirs(save_dir, exist_ok=True)
    transform = get_transform()
    paths = [os.path.join(folder, f) for f in os.listdir(folder)
             if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    print(f"📁 Found {len(paths)} images in folder")

    metrics_flip, metrics_rot = [], []

    for path in tqdm(paths, desc="Analyzing images"):
        img_pil = Image.open(path).convert("RGB")
        img_t = transform(img_pil).unsqueeze(0).to(device)
        flip_t = transform(img_pil.transpose(Image.FLIP_LEFT_RIGHT)).unsqueeze(0).to(device)
        rot_t = transform(img_pil.rotate(90)).unsqueeze(0).to(device)

        with torch.no_grad():
            out_o = model(img_t)
            out_f = model(flip_t)
            out_r = model(rot_t)

        # Flip alignment
        eqv_f_aligned = torch.flip(out_f["equiv_map"], dims=[-1])
        eqv_r_aligned = torch.rot90(out_r["equiv_map"], 3, (2, 3))

        ecm_f, idm_f, corr_f = compute_metrics(out_o["equiv_map"], eqv_f_aligned,
                                               out_o["inv_vec"], out_f["inv_vec"])
        ecm_r, idm_r, corr_r = compute_metrics(out_o["equiv_map"], eqv_r_aligned,
                                               out_o["inv_vec"], out_r["inv_vec"])
        metrics_flip.append((ecm_f, idm_f, corr_f))
        metrics_rot.append((ecm_r, idm_r, corr_r))

    # Convert to arrays
    metrics_flip = np.array(metrics_flip)
    metrics_rot = np.array(metrics_rot)

    np.savez(os.path.join(save_dir, "eqv_summary.npz"),
             ecm_flip=metrics_flip[:, 0],
             idm_flip=metrics_flip[:, 1],
             corr_flip=metrics_flip[:, 2],
             ecm_rot=metrics_rot[:, 0],
             idm_rot=metrics_rot[:, 1],
             corr_rot=metrics_rot[:, 2])

    print("\n📈 Mean Metrics (Flip):")
    print(f"ECM: {metrics_flip[:,0].mean():.6f}, IDM: {metrics_flip[:,1].mean():.6f}, Corr: {metrics_flip[:,2].mean():.4f}")
    print("📈 Mean Metrics (Rotate):")
    print(f"ECM: {metrics_rot[:,0].mean():.6f}, IDM: {metrics_rot[:,1].mean():.6f}, Corr: {metrics_rot[:,2].mean():.4f}")

    # Histogram plots
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.hist(metrics_flip[:, 0], bins=20, alpha=0.7, label="Flip ECM")
    plt.hist(metrics_rot[:, 0], bins=20, alpha=0.7, label="Rotate ECM")
    plt.title("Equivariance Consistency Distribution")
    plt.xlabel("ECM"); plt.ylabel("Frequency"); plt.legend()

    plt.subplot(1, 2, 2)
    plt.hist(metrics_flip[:, 1], bins=20, alpha=0.7, label="Flip IDM")
    plt.hist(metrics_rot[:, 1], bins=20, alpha=0.7, label="Rotate IDM")
    plt.title("Invariance Consistency Distribution")
    plt.xlabel("IDM"); plt.ylabel("Frequency"); plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "folder_metrics_hist.png"))
    plt.close()

    print(f"💾 Saved summary & histograms in {save_dir}")


# -------------------------------
# MAIN
# -------------------------------
if __name__ == "__main__":
    weights = r"G:\\REU\\checkpoints\\dual_branch_semisupervised_final.pth"
    folder = r"G:\\REU\\bdd_100k_subset\\val"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(weights, device)
    analyze_folder(model, folder, device)
