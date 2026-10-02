"""
Module evaluate.py - Đánh giá mô hình đã huấn luyện trên tập Test độc lập.

Bao gồm:
- Tải mô hình tốt nhất từ models/best_model.pth.
- Chạy suy luận trên toàn bộ tập test_loader.
- Tính toán các chỉ số: Accuracy, Precision, Recall, F1-score (Macro & Weighted).
- Xuất Classification Report ra file outputs/classification_report.txt.
- Vẽ và lưu Ma trận nhầm lẫn (Confusion Matrix) ra file outputs/confusion_matrix.png.
- Phân tích chi tiết các lớp nhận diện tốt nhất và các cặp vải hay bị nhầm lẫn.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Thiết lập đường dẫn import
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.dataset import get_data_loaders, CLASS_NAMES
from src.model import get_model

# Hỗ trợ hiển thị tiếng Việt trên Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def evaluate_model(
    model_path: str = "models/best_model.pth",
    data_dir: str = "dataset",
    outputs_dir: str = "outputs",
    batch_size: int = 32
):
    """
    Quy trình đánh giá chi tiết trên tập kiểm thử (Test Set).
    """
    os.makedirs(outputs_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("           BẮT ĐẦU ĐÁNH GIÁ MÔ HÌNH TRÊN TẬP TEST")
    print(f"[*] Đường dẫn model checkpoint: {model_path}")
    print(f"[*] Thiết bị tính toán:         {device}")
    print("=" * 70)

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Không tìm thấy file trọng số '{model_path}'. Vui lòng chạy train.py trước."
        )

    # 1. Tải checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    class_names = checkpoint.get("class_names", CLASS_NAMES)
    num_classes = len(class_names)

    # 2. Khởi tạo mô hình và nạp trọng số
    model = get_model(num_classes=num_classes, pretrained=False, device=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    best_val_acc = checkpoint.get("best_val_acc", 0.0)
    print(f"[OK] Đã tải thành công checkpoint từ Epoch {checkpoint.get('epoch', 'N/A')}")
    print(f"     - Kỷ lục Validation Accuracy khi train: {best_val_acc * 100:.2f}%\n")

    # 3. Tải DataLoader tập Test
    _, _, test_loader, _, _ = get_data_loaders(
        data_dir=data_dir,
        batch_size=batch_size
    )

    y_true = []
    y_pred = []
    all_probs = []

    print("[*] Đang tiến hành dự đoán trên toàn bộ tập Test...")
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing", leave=False):
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    all_probs = np.array(all_probs)

    # 4. Tính toán các chỉ số thống kê
    acc = accuracy_score(y_true, y_pred)
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average="macro")
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")

    report_str = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4
    )

    cm = confusion_matrix(y_true, y_pred)

    # 5. Lưu classification report
    report_file = os.path.join(outputs_dir, "classification_report.txt")
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("      BÁO CÁO ĐÁNH GIÁ MÔ HÌNH TRÊN TẬP TEST (FABRIC MATERIAL AI)\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Mô hình sử dụng: ResNet18 Transfer Learning\n")
        f.write(f"Tổng số mẫu kiểm thử: {len(y_true)}\n\n")
        f.write("TỔNG HỢP CÁC CHỈ SỐ CHÍNH:\n")
        f.write(f"- Test Accuracy:           {acc * 100:.2f}%\n")
        f.write(f"- Precision (Macro):       {prec_macro * 100:.2f}%\n")
        f.write(f"- Recall (Macro):          {rec_macro * 100:.2f}%\n")
        f.write(f"- F1-Score (Macro):        {f1_macro * 100:.2f}%\n")
        f.write(f"- Precision (Weighted):    {prec_weighted * 100:.2f}%\n")
        f.write(f"- Recall (Weighted):       {rec_weighted * 100:.2f}%\n")
        f.write(f"- F1-Score (Weighted):     {f1_weighted * 100:.2f}%\n\n")
        f.write("CHI TIẾT THEO TỪNG LỚP CHẤT LIỆU VẢI:\n")
        f.write(report_str)
        f.write("\n" + "=" * 70 + "\n")

    print(f"[OK] Đã lưu báo cáo phân loại: {report_file}")

    # 6. Vẽ và lưu Ma trận nhầm lẫn (Confusion Matrix)
    plt.figure(figsize=(9, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        annot_kws={"size": 11, "weight": "bold"}
    )
    plt.title("Ma trận nhầm lẫn (Confusion Matrix) - ResNet18 Test Set", fontsize=13, fontweight="bold", pad=15)
    plt.xlabel("Nhãn dự đoán (Predicted Label)", fontsize=11, fontweight="bold", labelpad=10)
    plt.ylabel("Nhãn thực tế (True Label)", fontsize=11, fontweight="bold", labelpad=10)
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()

    cm_file = os.path.join(outputs_dir, "confusion_matrix.png")
    plt.savefig(cm_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Đã lưu hình ảnh Confusion Matrix: {cm_file}")

    # 7. Phân tích chi tiết hiệu năng và các cặp dễ nhầm lẫn
    class_prec, class_rec, class_f1, class_supp = precision_recall_fscore_support(y_true, y_pred, average=None)
    
    # Tìm class nhận diện tốt nhất theo F1
    best_class_idx = np.argmax(class_f1)
    worst_class_idx = np.argmin(class_f1)

    # Tìm các cặp dễ nhầm lẫn nhất (không tính đường chéo chính)
    confused_pairs = []
    for i in range(num_classes):
        for j in range(num_classes):
            if i != j and cm[i][j] > 0:
                confused_pairs.append((class_names[i], class_names[j], cm[i][j]))
    confused_pairs.sort(key=lambda x: x[2], reverse=True)

    print("\n" + "=" * 70)
    print("                  KẾT QUẢ ĐÁNH GIÁ THỰC TẾ TRÊN TEST SET")
    print("=" * 70)
    print(f"[*] TEST ACCURACY:   {acc * 100:.2f}%")
    print(f"[*] MACRO F1-SCORE:  {f1_macro * 100:.2f}%")
    print(f"[*] WEIGHTED F1:     {f1_weighted * 100:.2f}%")
    print("\n[*] Chi tiết từng lớp:")
    for i, cname in enumerate(class_names):
        print(f"    - {cname:<14}: Precision={class_prec[i]*100:5.1f}% | Recall={class_rec[i]*100:5.1f}% | F1={class_f1[i]*100:5.1f}% | Samples={class_supp[i]}")

    print("\n[*] PHÂN TÍCH CHẤT LƯỢNG MÔ HÌNH:")
    print(f"    + Lớp nhận diện TỐT NHẤT: '{class_names[best_class_idx]}' (F1-Score = {class_f1[best_class_idx]*100:.2f}%)")
    print(f"    + Lớp nhận diện THẤP NHẤT: '{class_names[worst_class_idx]}' (F1-Score = {class_f1[worst_class_idx]*100:.2f}%)")
    
    if confused_pairs:
        print("\n[*] CÁC CẶP CHẤT LIỆU DỄ BỊ NHẦM LẪN NHẤT:")
        for true_c, pred_c, count in confused_pairs[:5]:
            print(f"    + Thực tế là '{true_c}' bị dự đoán nhầm thành '{pred_c}': {count} mẫu")
    print("=" * 70)

    return {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "confusion_matrix": cm,
        "class_names": class_names
    }


if __name__ == "__main__":
    evaluate_model(
        model_path="models/best_model.pth",
        data_dir="dataset",
        outputs_dir="outputs"
    )
