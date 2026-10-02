"""
Module predict.py - Chức năng dự đoán chất liệu vải cho một ảnh mới bất kỳ.

Quy trình:
Image Input (File path hoặc PIL Image)
  -> Preprocessing (Resize 224x224, Normalize ImageNet)
  -> ResNet18 Forward Pass
  -> Softmax Activation
  -> Trích xuất nhãn dự đoán, Confidence, Top-3 và xác suất cả 7 lớp.
"""

import os
import sys
from typing import Dict, List, Any, Union
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Thiết lập đường dẫn import
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.dataset import IMAGENET_MEAN, IMAGENET_STD, CLASS_NAMES
from src.model import get_model

# Hỗ trợ hiển thị tiếng Việt trên Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


class FabricPredictor:
    """
    Lớp xử lý suy luận (Inference) cho bài toán phân loại chất liệu vải.
    """
    def __init__(self, model_path: str = "models/best_model.pth", device: str = None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Không tìm thấy file checkpoint '{model_path}'. Vui lòng kiểm tra lại đường dẫn."
            )

        # 1. Tải checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        self.class_names = checkpoint.get("class_names", CLASS_NAMES)
        self.class_to_idx = checkpoint.get("class_to_idx", {c: i for i, c in enumerate(self.class_names)})

        # 2. Khởi tạo mô hình
        self.model = get_model(num_classes=len(self.class_names), pretrained=False, device=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        # 3. Pipeline tiền xử lý cho ảnh đầu vào (Deterministic transforms)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])

    def predict(self, image_input: Union[str, Image.Image]) -> Dict[str, Any]:
        """
        Dự đoán chất liệu vải từ đường dẫn ảnh hoặc đối tượng PIL Image.
        
        Returns:
            Dict chứa:
            - predicted_class: Tên chất liệu vải dự đoán
            - confidence: Độ tin cậy (%) của nhãn cao nhất
            - top3: Danh sách Top 3 nhãn kèm % xác suất
            - all_probabilities: Dict xác suất của cả 7 loại vải
        """
        # Đọc ảnh nếu đầu vào là đường dẫn
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"Không tìm thấy file ảnh: {image_input}")
            img = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            img = image_input.convert("RGB")
        else:
            raise TypeError("image_input phải là đường dẫn file (str) hoặc đối tượng PIL.Image")

        # Áp dụng Transform và thêm chiều Batch: [1, 3, 224, 224]
        tensor = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0)  # Shape [7]

        # Sắp xếp các lớp theo xác suất giảm dần
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)

        top1_idx = sorted_indices[0].item()
        top1_class = self.class_names[top1_idx]
        top1_conf = sorted_probs[0].item() * 100.0

        # Top 3 predictions
        top3_list = []
        for i in range(min(3, len(self.class_names))):
            idx = sorted_indices[i].item()
            c_name = self.class_names[idx]
            conf = sorted_probs[i].item() * 100.0
            top3_list.append({"class": c_name, "confidence": conf})

        # Toàn bộ xác suất 7 lớp
        all_probs_dict = {
            self.class_names[i]: probs[i].item() * 100.0 for i in range(len(self.class_names))
        }

        return {
            "predicted_class": top1_class,
            "confidence": round(top1_conf, 2),
            "top3": top3_list,
            "all_probabilities": all_probs_dict
        }


if __name__ == "__main__":
    print("=" * 60)
    print("      KIỂM TRA CHỨC NĂNG DỰ ĐOÁN (src/predict.py)")
    print("=" * 60)

    model_file = "models/best_model.pth"
    if not os.path.exists(model_file):
        print(f"[!] File {model_file} chưa sẵn sàng. Hãy đợi quá trình huấn luyện hoàn tất.")
    else:
        predictor = FabricPredictor(model_path=model_file)
        print(f"[OK] Đã khởi tạo FabricPredictor thành công!")

        # Thử nghiệm với một vài ảnh mẫu từ tập test
        test_samples = [
            ("Denim", "dataset/test/Denim"),
            ("Silk", "dataset/test/Silk"),
            ("Wool", "dataset/test/Wool")
        ]

        for expected_class, folder in test_samples:
            if os.path.exists(folder):
                files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.png'))]
                if files:
                    sample_path = os.path.join(folder, files[0])
                    result = predictor.predict(sample_path)
                    print(f"\n[*] Test ảnh mẫu từ lớp [{expected_class}]:")
                    print(f"    - File: {sample_path}")
                    print(f"    - Dự đoán:   {result['predicted_class']} (Độ tin cậy: {result['confidence']}%)")
                    print("    - Top 3:")
                    for item in result["top3"]:
                        print(f"      + {item['class']}: {item['confidence']:.2f}%")
        print("=" * 60)
