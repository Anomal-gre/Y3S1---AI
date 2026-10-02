"""
Module train.py - Huấn luyện mô hình ResNet18 phân loại chất liệu vải.

Bao gồm:
- Thiết lập Random Seed đảm bảo khả năng tái lập kết quả (reproducibility).
- Huấn luyện theo chiến lược Transfer Learning (2-stage: Warm-up head -> Fine-tune backbone).
- Theo dõi Train/Val Loss, Train/Val Accuracy qua từng epoch.
- Cơ chế Early Stopping và tự động lưu mô hình tốt nhất (models/best_model.pth).
- Xuất biểu đồ Training Curve (outputs/training_curve.png).
"""

import os
import sys
import time
import json
import random
from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

# Thiết lập đường dẫn import src khi chạy độc lập
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.dataset import get_data_loaders, CLASS_NAMES
from src.model import get_model

# Hỗ trợ hiển thị tiếng Việt trên Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def set_seed(seed: int = 42) -> None:
    """
    Cố định random seed cho toàn bộ môi trường để đảm bảo tính tái lập kết quả.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> Tuple[float, float]:
    """
    Huấn luyện mô hình trên 1 epoch.
    Returns: (epoch_loss, epoch_accuracy)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc="Training", leave=False)
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        # Xóa gradient cũ
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass & cập nhật trọng số
        loss.backward()
        optimizer.step()

        # Tính toán thống kê
        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels.data).item()
        total += labels.size(0)

        # Cập nhật thanh tiến trình
        batch_acc = (preds == labels.data).float().mean().item() * 100.0
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{batch_acc:.1f}%"})

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """
    Đánh giá mô hình trên tập validation.
    Returns: (val_loss, val_accuracy)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation", leave=False)
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data).item()
            total += labels.size(0)

    val_loss = running_loss / total
    val_acc = correct / total
    return val_loss, val_acc


def plot_training_history(history: Dict[str, List[float]], output_path: str) -> None:
    """
    Vẽ và lưu biểu đồ Loss và Accuracy qua các Epochs.
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(14, 5))

    # Đồ thị 1: Loss Curve
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], "b-o", label="Train Loss", linewidth=2)
    plt.plot(epochs, history["val_loss"], "r--s", label="Validation Loss", linewidth=2)
    plt.title("Đường cong Mất mát (Loss Curve)", fontsize=13, fontweight="bold")
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Loss (CrossEntropy)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=11)

    # Đồ thị 2: Accuracy Curve
    plt.subplot(1, 2, 2)
    plt.plot(epochs, [a * 100 for a in history["train_acc"]], "b-o", label="Train Accuracy", linewidth=2)
    plt.plot(epochs, [a * 100 for a in history["val_acc"]], "g--^", label="Validation Accuracy", linewidth=2)
    plt.title("Đường cong Độ chính xác (Accuracy Curve)", fontsize=13, fontweight="bold")
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Độ chính xác (%)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Đã lưu biểu đồ huấn luyện: {output_path}")


def train_pipeline(
    data_dir: str = "dataset",
    models_dir: str = "models",
    outputs_dir: str = "outputs",
    batch_size: int = 32,
    stage1_epochs: int = 1,
    stage2_epochs: int = 5,
    lr_head: float = 1e-3,
    lr_backbone: float = 1e-4,
    patience: int = 3,
    seed: int = 42
) -> None:
    """
    Toàn bộ quy trình huấn luyện Transfer Learning 2 giai đoạn.
    """
    set_seed(seed)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("           BẮT ĐẦU QUY TRÌNH HUẤN LUYỆN RESNET18")
    print(f"[*] Thiết bị tính toán (Device): {device}")
    print(f"[*] Random Seed: {seed}")
    print(f"[*] Batch size: {batch_size}")
    print(f"[*] Chiến lược 2 bước: Stage 1 ({stage1_epochs} epochs) -> Stage 2 ({stage2_epochs} epochs)")
    print("=" * 70)

    # 1. Tải DataLoaders
    train_loader, val_loader, _, class_names, class_to_idx = get_data_loaders(
        data_dir=data_dir,
        batch_size=batch_size
    )

    # 2. Khởi tạo mô hình
    model = get_model(num_classes=len(class_names), pretrained=True, device=device)
    criterion = nn.CrossEntropyLoss()

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": []
    }

    best_val_acc = 0.0
    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_model_path = os.path.join(models_dir, "best_model.pth")
    total_epochs = stage1_epochs + stage2_epochs

    current_epoch = 0

    # -------------------------------------------------------------
    # GIAI ĐOẠN 1: FREEZE BACKBONE, TRAIN CLASSIFICATION HEAD
    # -------------------------------------------------------------
    if stage1_epochs > 0:
        print("\n>>> GIAI ĐOẠN 1: Đóng băng Backbone, huấn luyện Classification Head...")
        model.freeze_backbone()
        optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr_head, weight_decay=1e-4)

        for epoch in range(1, stage1_epochs + 1):
            current_epoch += 1
            start_time = time.time()
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = validate(model, val_loader, criterion, device)
            elapsed = time.time() - start_time

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)

            print(f"Epoch [{current_epoch}/{total_epochs}] (Stage 1) - {elapsed:.1f}s | "
                  f"Train Loss: {train_loss:.4f} - Train Acc: {train_acc*100:.2f}% | "
                  f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc*100:.2f}%")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_val_loss = val_loss
                torch.save({
                    "epoch": current_epoch,
                    "model_state_dict": model.state_dict(),
                    "class_names": class_names,
                    "class_to_idx": class_to_idx,
                    "best_val_acc": best_val_acc,
                    "best_val_loss": best_val_loss
                }, best_model_path)
                print(f"  [+] Đã lưu checkpoint mới tốt nhất: Val Acc = {val_acc*100:.2f}%")

    # -------------------------------------------------------------
    # GIAI ĐOẠN 2: UNFREEZE TOÀN BỘ, FINE-TUNE VỚI LEARNING RATE NHỎ
    # -------------------------------------------------------------
    if stage2_epochs > 0:
        print("\n>>> GIAI ĐOẠN 2: Mở băng Backbone, Fine-tuning toàn bộ mô hình...")
        model.unfreeze_backbone()
        optimizer = AdamW(model.parameters(), lr=lr_backbone, weight_decay=1e-4)
        scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=1)

        for epoch in range(1, stage2_epochs + 1):
            current_epoch += 1
            start_time = time.time()
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = validate(model, val_loader, criterion, device)
            elapsed = time.time() - start_time

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)

            scheduler.step(val_acc)
            current_lr = optimizer.param_groups[0]["lr"]

            print(f"Epoch [{current_epoch}/{total_epochs}] (Stage 2) - {elapsed:.1f}s (LR: {current_lr:.1e}) | "
                  f"Train Loss: {train_loss:.4f} - Train Acc: {train_acc*100:.2f}% | "
                  f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc*100:.2f}%")

            # Kiểm tra cải thiện để lưu checkpoint
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_val_loss = val_loss
                epochs_no_improve = 0
                torch.save({
                    "epoch": current_epoch,
                    "model_state_dict": model.state_dict(),
                    "class_names": class_names,
                    "class_to_idx": class_to_idx,
                    "best_val_acc": best_val_acc,
                    "best_val_loss": best_val_loss
                }, best_model_path)
                print(f"  [+] Đã lưu checkpoint mới tốt nhất: Val Acc = {val_acc*100:.2f}%")
            else:
                epochs_no_improve += 1
                print(f"  [-] Chưa cải thiện ({epochs_no_improve}/{patience} epochs)")
                if epochs_no_improve >= patience:
                    print(f"\n[!] EARLY STOPPING kích hoạt tại epoch {current_epoch} do không cải thiện sau {patience} epochs.")
                    break

    # 3. Lưu lịch sử và vẽ biểu đồ
    history_path = os.path.join(outputs_dir, "history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=4)

    curve_path = os.path.join(outputs_dir, "training_curve.png")
    plot_training_history(history, curve_path)

    print("=" * 70)
    print("           HUẤN LUYỆN HOÀN TẤT THÀNH CÔNG!")
    print(f"[*] Trọng số mô hình tốt nhất: {best_model_path}")
    print(f"[*] Kỷ lục Validation Accuracy: {best_val_acc*100:.2f}% (Loss: {best_val_loss:.4f})")
    print(f"[*] Biểu đồ đường cong học tập: {curve_path}")
    print("=" * 70)


if __name__ == "__main__":
    train_pipeline(
        data_dir="dataset",
        models_dir="models",
        outputs_dir="outputs",
        batch_size=32,
        stage1_epochs=1,
        stage2_epochs=4,
        lr_head=1e-3,
        lr_backbone=1e-4,
        patience=3,
        seed=42
    )
