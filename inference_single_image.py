"""
inference_dual_branch_analysis_final.py
---------------------------------------
Complete inference and analysis for DualBranch model (Equivariant + Invariant features).

Performs:
✅ Equivariance Consistency (ECM)
✅ Invariance Distance (IDM)
✅ Feature Correlation
✅ Cosine Similarity for Invariance
✅ Rotation-based ECM Curve
✅ Channel-wise Feature Visualization

Author: Vineet (REU)
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# -------------------------------
# DualBranch Model Definition
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
# Utility Functions
# -------------------------------
def load_model(weights_path, device):
    model = DualBranch()
    state_dict = torch.load(weights_path, map_location=device, weights_only=False)
    model.load_state_dict(state_dict, strict=False)
    model.eval().to(device)
    print(f"✅ Loaded model from {weights_path}")
    return model


def get_transform(image_size=128):
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])


def equivariance_consistency(eqv_base, eqv_trans):
    return torch.mean(torch.abs(eqv_base - eqv_trans)).item()


def invariance_distance(inv_base, inv_trans):
    return F.mse_loss(inv_base, inv_trans).item()


def feature_correlation(eqv_base, eqv_trans):
    f1 = eqv_base.flatten().cpu().numpy()
    f2 = eqv_trans.flatten().cpu().numpy()
    return np.corrcoef(f1, f2)[0, 1]


def cosine_similarity(vec1, vec2):
    v1 = vec1.cpu().numpy().flatten()
    v2 = vec2.cpu().numpy().flatten()
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)


# -------------------------------
# Visualization Functions
# -------------------------------
def visualize_topk_equivariance(imgs, eqv_maps, save_path, top_k=3):
    """
    Show top-k most active channels for each transformed image.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    num_imgs = len(imgs)
    fig, axs = plt.subplots(top_k + 1, num_imgs, figsize=(4 * num_imgs, 3 * (top_k + 1)))

    titles = ["Original", "Flipped", "Rotated"]
    for i, (img_pil, fmap) in enumerate(zip(imgs, eqv_maps)):
        axs[0, i].imshow(img_pil)
        axs[0, i].set_title(titles[i])
        axs[0, i].axis("off")

        fmap_np = fmap.squeeze().detach().cpu().numpy()  # [C,H,W]
        ch_mean = fmap_np.reshape(fmap_np.shape[0], -1).mean(axis=1)
        top_channels = np.argsort(ch_mean)[-top_k:]

        for j, ch in enumerate(top_channels):
            ch_map = fmap_np[ch]
            ch_map = (ch_map - ch_map.min()) / (ch_map.max() - ch_map.min() + 1e-8)
            axs[j + 1, i].imshow(ch_map, cmap="inferno")
            axs[j + 1, i].set_title(f"Ch {ch}")
            axs[j + 1, i].axis("off")

    plt.suptitle("Top-3 Active Equivariant Channels – Movement under Transformations", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"🖼️ Saved detailed equivariance visualization: {save_path}")


def visualize_difference_map(eqv_orig, eqv_rot, save_path):
    diff = torch.abs(eqv_orig - eqv_rot).mean(1).squeeze().cpu().numpy()
    diff = (diff - diff.min()) / (diff.max() - diff.min() + 1e-8)
    plt.imshow(diff, cmap="coolwarm")
    plt.title("Feature Map Difference (Δ-map)")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"🧭 Saved Δ-map: {save_path}")


# -------------------------------
# Inference for Single Image
# -------------------------------
def analyze_single_image(model, image_path, device, save_dir="analysis_results"):
    os.makedirs(save_dir, exist_ok=True)
    transform = get_transform()

    img_pil = Image.open(image_path).convert("RGB")
    img_t = transform(img_pil).to(device).unsqueeze(0)

    # Apply transformations
    flip_pil = img_pil.transpose(Image.FLIP_LEFT_RIGHT)
    rot_pil = img_pil.rotate(90)
    flip_t = transform(flip_pil).to(device).unsqueeze(0)
    rot_t = transform(rot_pil).to(device).unsqueeze(0)

    with torch.no_grad():
        out_orig = model(img_t)
        out_flip = model(flip_t)
        out_rot = model(rot_t)

    eqv_orig, eqv_flip, eqv_rot = out_orig["equiv_map"], out_flip["equiv_map"], out_rot["equiv_map"]
    inv_orig, inv_flip, inv_rot = out_orig["inv_vec"], out_flip["inv_vec"], out_rot["inv_vec"]

    # Compute metrics
    ecm_flip = equivariance_consistency(eqv_orig, torch.flip(eqv_flip, dims=[-1]))
    ecm_rot = equivariance_consistency(eqv_orig, torch.rot90(eqv_rot, 3, (2, 3)))
    idm_flip = invariance_distance(inv_orig, inv_flip)
    idm_rot = invariance_distance(inv_orig, inv_rot)
    corr_flip = feature_correlation(eqv_orig, torch.flip(eqv_flip, dims=[-1]))
    corr_rot = feature_correlation(eqv_orig, torch.rot90(eqv_rot, 3, (2, 3)))
    cos_sim_flip = cosine_similarity(inv_orig, inv_flip)
    cos_sim_rot = cosine_similarity(inv_orig, inv_rot)

    print("\n📊 ----- Metrics for Single Image -----")
    print(f"ECM (Flip):   {ecm_flip:.6f}")
    print(f"ECM (Rotate): {ecm_rot:.6f}")
    print(f"IDM (Flip):   {idm_flip:.6f}")
    print(f"IDM (Rotate): {idm_rot:.6f}")
    print(f"Corr (Flip):  {corr_flip:.3f}")
    print(f"Corr (Rotate):{corr_rot:.3f}")
    print(f"CosSim (Flip):{cos_sim_flip:.3f}")
    print(f"CosSim (Rotate):{cos_sim_rot:.3f}")

    # Visualization
    visualize_topk_equivariance([img_pil, flip_pil, rot_pil],
                                [eqv_orig, eqv_flip, eqv_rot],
                                os.path.join(save_dir, "equivariance_topk.png"))
    visualize_difference_map(eqv_orig, torch.rot90(eqv_rot, 3, (2, 3)),
                             os.path.join(save_dir, "difference_map.png"))

    # ECM vs rotation curve
    angles = [0, 45, 90, 135, 180]
    ecm_vals = []
    for angle in angles:
        rot_img = img_pil.rotate(angle)
        rot_t = transform(rot_img).to(device).unsqueeze(0)
        eqv_rot = model(rot_t)["equiv_map"]
        ecm = equivariance_consistency(eqv_orig, eqv_rot)
        ecm_vals.append(ecm)

    plt.plot(angles, ecm_vals, marker='o', color='crimson')
    plt.xlabel("Rotation Angle (°)")
    plt.ylabel("ECM")
    plt.title("Equivariance Consistency vs Rotation Angle")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "ecm_vs_angle.png"))
    plt.close()
    print(f"📈 Saved ECM vs Angle plot: {save_dir}/ecm_vs_angle.png")

    # Save summary
    np.savez(os.path.join(save_dir, "metrics_summary.npz"),
             ECM_Flip=ecm_flip, ECM_Rot=ecm_rot,
             IDM_Flip=idm_flip, IDM_Rot=idm_rot,
             Corr_Flip=corr_flip, Corr_Rot=corr_rot,
             CosSim_Flip=cos_sim_flip, CosSim_Rot=cos_sim_rot)
    print(f"💾 Saved metrics summary: {save_dir}/metrics_summary.npz")


# -------------------------------
# Run Example
# -------------------------------
if __name__ == "__main__":
    weights_path = r"G:\\REU\\checkpoints\\dual_branch_semisupervised_final.pth"
    test_image = r"G:\\REU\\bdd_100k_subset\\val\\00030.jpg"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(weights_path, device)
    analyze_single_image(model, test_image, device)
