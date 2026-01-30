"""
train_dual_branch_semisupervised_final_v3.py
---------------------------------------------------------
✔ ResNet-50 backbone (1024 output channels from layer3)
✔ Corrected heads to use CHANNELS = 1024
✔ Saves ONLY best model + final model inside 'trained_model/'
✔ No epoch checkpoints saved
✔ Fully cleaned and error-free
---------------------------------------------------------
Author: Vineet Desai (REU)
"""

import os, glob, random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


# ============================================================
# Model Definition (ResNet-50 backbone → 1024 channels)
# ============================================================
class DualBranch(nn.Module):
    def __init__(self, emb_dim=64, num_classes=None, pretrained=True):
        super().__init__()

        base = torchvision.models.resnet50(pretrained=pretrained)

        # ResNet50 output after layer3 = 1024 channels
        self.backbone = nn.Sequential(
            base.conv1, base.bn1, base.relu, base.maxpool,
            base.layer1, base.layer2, base.layer3
        )

        CHANNELS = 1024   # IMPORTANT FIX

        # Equivariant Head
        self.eqv_head = nn.Sequential(
            nn.Conv2d(CHANNELS, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, emb_dim, 3, padding=1)
        )

        # Invariant Head
        self.inv_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(CHANNELS, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, emb_dim)
        )

        self.cls_head = nn.Linear(emb_dim, num_classes) if num_classes else None

    def forward(self, x):
        feat = self.backbone(x)
        eqv_map = self.eqv_head(feat)
        inv_vec = self.inv_head(feat)
        inv_vec = F.normalize(inv_vec, dim=1)

        out = {"equiv_map": eqv_map, "inv_vec": inv_vec}
        if self.cls_head:
            out["logits"] = self.cls_head(inv_vec)
        return out


# ============================================================
# Loss Functions
# ============================================================
def rotate_batch_tensor(x, k):
    return torch.rot90(x, k, dims=(2, 3))


def equivariance_loss(eqv_orig, eqv_rot, k):
    eqv_orig_rot = torch.rot90(eqv_orig, k, dims=(2, 3))
    B = eqv_orig.shape[0]

    f1 = eqv_orig_rot.reshape(B, -1)
    f2 = eqv_rot.reshape(B, -1)

    f1 = F.normalize(f1, dim=1)
    f2 = F.normalize(f2, dim=1)

    return 1.0 - (f1 * f2).sum(dim=1).mean()


def nt_xent_loss(z1, z2, temperature=0.5):
    B = z1.shape[0]
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)

    z = torch.cat([z1, z2], dim=0)
    sim = torch.matmul(z, z.T) / temperature

    mask = (~torch.eye(2 * B, device=sim.device).bool()).float()
    exp_sim = torch.exp(sim) * mask
    denom = exp_sim.sum(dim=1, keepdim=True)

    pos = torch.exp((z1 * z2).sum(dim=1) / temperature)
    pos = torch.cat([pos, pos], dim=0)

    loss = -torch.log(pos / denom.squeeze())
    return loss.mean()


# ============================================================
# Dataset
# ============================================================
class BDDImages(Dataset):
    def __init__(self, root_dir, transform=None):
        self.imgs = []
        for ext in ["jpg", "jpeg", "png"]:
            self.imgs += glob.glob(os.path.join(root_dir, f"**/*.{ext}"), recursive=True)

        self.imgs.sort()
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = Image.open(self.imgs[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, -1


def get_dataloaders(data_dir, batch_size=8, val_split=0.05, image_size=128):
    normalize = T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])

    base_transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        normalize
    ])

    dataset = BDDImages(data_dir, base_transform)
    n = len(dataset)
    val_len = max(1, int(n * val_split))

    train_ds, val_ds = random_split(dataset, [n - val_len, val_len])

    print(f"📁 Total images: {n}")
    print(f"🟢 Train: {len(train_ds)} | 🔵 Val: {len(val_ds)}")

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    )


# ============================================================
# Training Loop
# ============================================================
def train_dual_branch(
    data_dir,
    epochs=50,
    lr=1e-4,
    lambda_eqv=30.0,
    lambda_inv=1.0,
    batch_size=8,
    image_size=128,
    device="cuda"
):

    device = torch.device(device)
    model = DualBranch(emb_dim=64, pretrained=True).to(device)

    eqv_params = [p for n, p in model.named_parameters() if "eqv_head" in n]
    other_params = [p for n, p in model.named_parameters() if "eqv_head" not in n]

    optimizer = torch.optim.Adam([
        {"params": other_params, "lr": lr},
        {"params": eqv_params, "lr": lr * 10.0}
    ])

    os.makedirs("trained_model", exist_ok=True)

    train_loader, val_loader = get_dataloaders(data_dir, batch_size, 0.05, image_size)

    aug_pipeline = T.Compose([
        T.ToPILImage(),
        T.ColorJitter(0.4,0.4,0.4,0.1),
        T.RandomHorizontalFlip(p=0.5),
        T.ToTensor(),
        T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    best_val = float("inf")
    best_model_path = None
    train_losses, val_losses = [], []


    # ================================
    # TRAIN LOOP
    # ================================
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = []

        for imgs, _ in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False):
            imgs = imgs.to(device)

            k = random.randint(0, 3)
            rot_imgs = rotate_batch_tensor(imgs, k)
            aug_imgs = torch.stack([aug_pipeline(img.cpu()) for img in imgs]).to(device)

            out_orig = model(imgs)
            out_rot = model(rot_imgs)
            out_aug = model(aug_imgs)

            L_eqv = equivariance_loss(out_orig["equiv_map"], out_rot["equiv_map"], k)
            L_inv = nt_xent_loss(out_orig["inv_vec"], out_aug["inv_vec"])

            loss = lambda_eqv * L_eqv + lambda_inv * L_inv

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss.append(loss.item())

        avg_train = np.mean(running_loss)
        train_losses.append(avg_train)


        # ================================
        # VALIDATE
        # ================================
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for imgs, _ in val_loader:
                imgs = imgs.to(device)

                k = random.randint(0, 3)
                rot_imgs = rotate_batch_tensor(imgs, k)

                out1 = model(imgs)["equiv_map"]
                out2 = model(rot_imgs)["equiv_map"]

                val_loss += equivariance_loss(out1, out2, k).item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        print(f"Epoch {epoch} | Train={avg_train:.4f} | Val={val_loss:.4f}")

        # ======================================
        # SAVE ONLY BEST MODEL
        # ======================================
        if val_loss < best_val:
            best_val = val_loss
            best_model_path = f"trained_model/best_model_epoch_{epoch:03d}.pth"
            torch.save(model.state_dict(), best_model_path)


    # ============================================================
    # FINAL MODEL SAVE
    # ============================================================
    final_path = "trained_model/dual_branch_semisupervised_final.pth"
    torch.save(model.state_dict(), final_path)

    print("\n🎉 Training Finished!")
    print(f"📈 Best Validation Loss: {best_val:.6f}")
    print(f"🏆 Best Model Saved at: {best_model_path}")
    print(f"💾 Final Model Saved at: {final_path}")

    # Plot curves
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.legend()
    plt.title("Training vs Validation Loss")
    plt.tight_layout()
    plt.show()

    return model, best_model_path


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    data_dir = r"G:\\Minor_project\\bdd100k\\bdd100k\\bdd100k\\images\\10k\\train"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dual_branch(
        data_dir=data_dir,
        epochs=50,
        lr=1e-4,
        lambda_eqv=30.0,
        lambda_inv=1.0,
        batch_size=8,
        image_size=128,
        device=device
    )
