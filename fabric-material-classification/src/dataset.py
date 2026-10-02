"""
Module dataset.py - Xử lý dữ liệu và tạo PyTorch DataLoader cho TextileClass-7.

Bao gồm:
- Data transforms cho Train (có Augmentation) và Val/Test (chuẩn hóa cố định).
- Hàm xây dựng DataLoader cho train, valid, test.
- Kiểm tra dữ liệu và ánh xạ nhãn cố định 7 classes.
"""

import os
import sys
from typing import Dict, List, Tuple
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Hỗ trợ hiển thị tiếng Việt trên Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Danh sách chuẩn 7 lớp chất liệu vải của TextileClass-7
CLASS_NAMES: List[str] = [
    "Cotton",
    "Cotton Mixed",
    "Denim",
    "Polyester",
    "Silk",
    "Viscose",
    "Wool"
]

# Thông số chuẩn hóa theo ImageNet (mean và std trên 3 kênh R, G, B)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms() -> Dict[str, transforms.Compose]:
    """
    Tạo pipeline tiền xử lý ảnh và Data Augmentation.
    - Train: Resize, Lật ngang ngẫu nhiên, Xoay nhẹ (+/-15 độ), ColorJitter nhẹ, Normalize.
    - Valid / Test: Resize và Normalize cố định (không áp dụng augmentation ngẫu nhiên).
    """
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    return {
        "train": train_transform,
        "valid": val_test_transform,
        "test": val_test_transform
    }


def get_data_loaders(
    data_dir: str = "dataset",
    batch_size: int = 32,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str], Dict[str, int]]:
    """
    Khởi tạo các DataLoader cho tập train, valid và test.
    
    Args:
        data_dir: Đường dẫn đến thư mục dataset (chứa train/, valid/, test/)
        batch_size: Kích thước batch (mặc định 32)
        num_workers: Số luồng tải dữ liệu (trên Windows để 0 để tối ưu độ ổn định)
        
    Returns:
        (train_loader, val_loader, test_loader, class_names, class_to_idx)
    """
    transforms_dict = get_transforms()
    
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "valid")
    test_dir = os.path.join(data_dir, "test")

    # Kiểm tra sự tồn tại của các thư mục dữ liệu
    for path, name in [(train_dir, "train"), (val_dir, "valid"), (test_dir, "test")]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Thư mục '{path}' không tồn tại. Vui lòng kiểm tra lại cấu trúc thư mục dataset/{name}."
            )

    # Sử dụng ImageFolder từ torchvision
    train_dataset = datasets.ImageFolder(root=train_dir, transform=transforms_dict["train"])
    val_dataset = datasets.ImageFolder(root=val_dir, transform=transforms_dict["valid"])
    test_dataset = datasets.ImageFolder(root=test_dir, transform=transforms_dict["test"])

    # Xác thực danh sách lớp
    detected_classes = train_dataset.classes
    class_to_idx = train_dataset.class_to_idx

    # Khởi tạo DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,       # Xáo trộn dữ liệu ở mỗi epoch huấn luyện
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,      # Không xáo trộn ở tập validation
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,      # Không xáo trộn ở tập test để giữ thứ tự đánh giá
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, val_loader, test_loader, detected_classes, class_to_idx


if __name__ == "__main__":
    print("=" * 60)
    print("      KIỂM TRA PIPELINE DỮ LIỆU (src/dataset.py)")
    print("=" * 60)
    
    dataset_path = "dataset"
    if not os.path.exists(dataset_path):
        # Hỗ trợ chạy từ thư mục gốc hoặc thư mục src
        dataset_path = os.path.join("..", "dataset")

    try:
        train_loader, val_loader, test_loader, classes, class_to_idx = get_data_loaders(
            data_dir=dataset_path,
            batch_size=32
        )
        
        print(f"[OK] Số lượng lớp nhận diện: {len(classes)}")
        print("     Ánh xạ nhãn (class_to_idx):")
        for cls_name, idx in class_to_idx.items():
            print(f"       - {idx}: {cls_name}")

        print("\n[OK] Kích thước từng tập dữ liệu:")
        print(f"     - Train samples: {len(train_loader.dataset)} (Số batch: {len(train_loader)})")
        print(f"     - Valid samples: {len(val_loader.dataset)} (Số batch: {len(val_loader)})")
        print(f"     - Test samples:  {len(test_loader.dataset)} (Số batch: {len(test_loader)})")

        # Thử lấy 1 batch dữ liệu mẫu để kiểm tra shape tensor
        images, labels = next(iter(train_loader))
        print("\n[OK] Kiểm tra 1 Batch mẫu từ Train DataLoader:")
        print(f"     - Tensor Images shape: {images.shape} -> (Batch_size, Channels, Height, Width)")
        print(f"     - Tensor Labels shape: {labels.shape}")
        print(f"     - Kiểu dữ liệu ảnh: {images.dtype}, min: {images.min():.2f}, max: {images.max():.2f}")
        print(f"     - Mẫu nhãn trong batch: {labels[:8].tolist()}")

        print("=" * 60)
        print("[OK] DATASET PIPELINE HOẠT ĐỘNG HOÀN HẢO!")
        print("=" * 60)

    except Exception as e:
        print(f"[LỖI] Xảy ra lỗi khi kiểm tra pipeline dữ liệu: {e}")
        import traceback
        traceback.print_exc()
