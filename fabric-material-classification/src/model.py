"""
Module model.py - Xây dựng kiến trúc mô hình ResNet18 Transfer Learning cho bài toán phân loại chất liệu vải.

Bao gồm:
- Tải ResNet18 pretrained từ ImageNet.
- Tùy biến Classification Head với output 7 classes.
- Cung cấp cơ chế đóng băng (freeze) và mở băng (unfreeze) backbone phục vụ 2-stage fine-tuning.
"""

import sys
from typing import Tuple
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

# Đảm bảo hiển thị tiếng Việt trên Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


class FabricResNet18(nn.Module):
    """
    Mô hình phân loại chất liệu vải dựa trên ResNet18 Pretrained.
    """
    def __init__(self, num_classes: int = 7, pretrained: bool = True, dropout_rate: float = 0.2):
        super(FabricResNet18, self).__init__()
        
        # 1. Tải kiến trúc ResNet18 cùng trọng số tiền huấn luyện ImageNet
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = resnet18(weights=weights)
        
        # 2. Thay thế lớp Fully Connected (fc) cuối cùng
        # ResNet18 có kích thước vector đặc trưng đầu ra của backbone là 512
        in_features = self.backbone.fc.in_features
        
        # Thay thế bằng classification head phù hợp với 7 classes vải
        if dropout_rate > 0.0:
            self.backbone.fc = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(in_features=in_features, out_features=num_classes)
            )
        else:
            self.backbone.fc = nn.Linear(in_features=in_features, out_features=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Thực hiện lan truyền tiến (Forward Pass).
        Input: Tensor ảnh [B, 3, 224, 224]
        Output: Logits thô [B, 7] (chưa qua Softmax)
        """
        return self.backbone(x)

    def freeze_backbone(self) -> None:
        """
        Đóng băng toàn bộ trọng số của Backbone (các tầng Convolutional Feature Extractor).
        Chỉ cho phép cập nhật trọng số ở Classification Head (lớp fc).
        """
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # Mở khóa riêng lớp fully connected cuối cùng
        for param in self.backbone.fc.parameters():
            param.requires_grad = True

    def unfreeze_backbone(self, unfreeze_layer4_only: bool = False) -> None:
        """
        Mở băng các trọng số để Fine-Tuning.
        - unfreeze_layer4_only = True: chỉ mở các tầng sâu nhất (layer4 + fc)
        - unfreeze_layer4_only = False: mở toàn bộ mạng để fine-tune toàn bộ
        """
        if unfreeze_layer4_only:
            # Đóng băng tất cả trước
            for param in self.backbone.parameters():
                param.requires_grad = False
            # Mở layer4 (các block tích chập mức cao) và fc
            for param in self.backbone.layer4.parameters():
                param.requires_grad = True
            for param in self.backbone.fc.parameters():
                param.requires_grad = True
        else:
            for param in self.backbone.parameters():
                param.requires_grad = True


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    Đếm số lượng tham số tổng và tham số huấn luyện được (trainable).
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def get_model(num_classes: int = 7, pretrained: bool = True, device: str = "cpu") -> FabricResNet18:
    """
    Khởi tạo mô hình và chuyển sang thiết bị tính toán (CPU hoặc GPU).
    """
    model = FabricResNet18(num_classes=num_classes, pretrained=pretrained)
    model = model.to(device)
    return model


if __name__ == "__main__":
    print("=" * 60)
    print("      KIỂM TRA KIẾN TRÚC MÔ HÌNH (src/model.py)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Thiết bị kiểm tra: {device}")

    # Khởi tạo mô hình
    model = get_model(num_classes=7, pretrained=True, device=device)
    total_p, train_p = count_parameters(model)
    print(f"[OK] Đã tải ResNet18 ImageNet weights thành công!")
    print(f"     - Tổng số tham số (Total Params):      {total_p:,}")
    print(f"     - Tham số huấn luyện (Trainable):      {train_p:,}")

    # Kiểm tra đóng băng backbone
    model.freeze_backbone()
    _, frozen_train_p = count_parameters(model)
    print(f"[OK] Sau khi đóng băng Backbone (Freeze feature extractor):")
    print(f"     - Tham số còn lại để train (Chỉ Head): {frozen_train_p:,}")

    # Mở lại để chuẩn bị
    model.unfreeze_backbone()

    # Kiểm tra Forward Pass với Dummy Tensor
    dummy_input = torch.randn(4, 3, 224, 224, device=device)
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)
        probs = torch.softmax(output, dim=1)

    print("\n[OK] Kiểm tra Forward Pass thành công:")
    print(f"     - Kích thước đầu vào (Input shape):   {dummy_input.shape}")
    print(f"     - Kích thước đầu ra Logits (Output):  {output.shape} -> (Batch size, 7 classes)")
    print(f"     - Tổng xác suất mỗi sample (Softmax): {probs.sum(dim=1).cpu().numpy().tolist()}")
    print("=" * 60)
    print("[OK] KIẾN TRÚC MÔ HÌNH RESNET18 ĐÃ SẴN SÀNG!")
    print("=" * 60)
