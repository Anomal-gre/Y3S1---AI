# 🧵 ỨNG DỤNG DEEP LEARNING TRONG NHẬN DIỆN CHẤT LIỆU VẢI TỪ HÌNH ẢNH
*(Fabric Material Classification using Deep Learning & Transfer Learning)*

> **Đề tài bài tập lớn / giữa kỳ môn Trí tuệ nhân tạo & Học sâu**  
> **Kiến trúc:** ResNet18 (Pretrained on ImageNet - Transfer Learning)  
> **Tập dữ liệu:** TextileClass-7 (7 lớp chất liệu vải dệt)  
> **Giao diện Demo:** Streamlit Web Application  

---

## 📌 1. Mục tiêu đề tài
Xây dựng một hệ thống thị giác máy tính hoàn chỉnh có khả năng nhận một bức ảnh chụp cận cảnh bề mặt vải làm đầu vào và tự động dự đoán chất liệu vải tương ứng.

- **Đầu vào (Input):** Ảnh bề mặt vải định dạng `.jpg`, `.jpeg`, `.png`.
- **Đầu ra (Output):**
  - Tên loại vải được dự đoán chính xác nhất (Top-1 Prediction).
  - Độ tin cậy dự đoán (Confidence score, %).
  - Top 3 chất liệu có xác suất cao nhất.
  - Biểu đồ phân phối xác suất trên toàn bộ 7 lớp vải.

---

## 📊 2. Bộ dữ liệu (Dataset TextileClass-7)
Dự án sử dụng bộ dữ liệu chuẩn **TextileClass-7** trích xuất từ nguồn nghiên cứu học thuật:
- **Nguồn dữ liệu:** [Mendeley Data - TextileClass-7](https://data.mendeley.com/datasets/m2h53zr39s/2)
- **7 Lớp chất liệu vải (7 Classes):**
  1. `Cotton` (Vải bông tự nhiên)
  2. `Cotton Mixed` (Vải bông pha sợi tổng hợp)
  3. `Denim` (Vải Jean/Denim dệt chéo dày)
  4. `Polyester` (Vải sợi tổng hợp bóng nhẹ)
  5. `Silk` (Lụa tơ tằm mịn, độ bóng cao)
  6. `Viscose` (Vải bán tổng hợp từ cellulose)
  7. `Wool` (Vải len sợi xoắn tự nhiên)

### Thống kê dữ liệu thực tế:
- **Tổng số mẫu ảnh:** 6,284 ảnh (100% ảnh đọc tốt, không có ảnh hỏng, hệ màu RGB).
- **Phân chia không rò rỉ dữ liệu (No Data Leakage):**
  - **Tập Huấn luyện (Train set):** 5,499 ảnh (~87.5%)
  - **Tập Xác thực (Validation set):** 392 ảnh (~6.2%)
  - **Tập Kiểm thử (Test set):** 393 ảnh (~6.3%)
- **Độ cân bằng lớp:** Phân bố giữa các lớp tương đối đồng đều (~800 ảnh/lớp ở train, ~50-60 ảnh/lớp ở valid và test).

---

## ⚙️ 3. Công nghệ và Thư viện sử dụng
- **Ngôn ngữ:** Python 3.10+ (Đã kiểm thử trên Python 3.11)
- **Framework Học sâu:** `PyTorch >= 2.0.0`, `torchvision >= 0.15.0`
- **Học máy & Đánh giá:** `scikit-learn`
- **Xử lý ảnh & Tính toán ma trận:** `Pillow (PIL)`, `NumPy`
- **Trực quan hóa dữ liệu:** `Matplotlib`, `Seaborn`
- **Giao diện Web Demo:** `Streamlit >= 1.28.0`
- **Hỗ trợ thiết bị:** Tự động nhận diện GPU (CUDA) nếu có, chạy mượt mà và tối ưu trên CPU.

---

## 🏗️ 4. Cấu trúc thư mục dự án
```text
fabric-material-classification/
│
├── dataset/                        # Dữ liệu ảnh chia theo train/valid/test
│   ├── train/                      # 5,499 ảnh (7 thư mục tương ứng 7 classes)
│   ├── valid/                      # 392 ảnh
│   └── test/                       # 393 ảnh
│
├── models/
│   └── best_model.pth              # Trọng số mô hình tốt nhất lưu lại theo Val Accuracy
│
├── outputs/
│   ├── training_curve.png          # Đồ thị Train/Val Loss & Accuracy qua các Epochs
│   ├── confusion_matrix.png        # Ma trận nhầm lẫn 7 lớp trên Test set
│   ├── classification_report.txt   # Báo cáo chi tiết Precision, Recall, F1-score
│   └── history.json                # Nhật ký số liệu qua các Epochs
│
├── src/
│   ├── __init__.py
│   ├── dataset.py                  # Pipeline xử lý dữ liệu, Augmentation, DataLoader
│   ├── model.py                    # Kiến trúc ResNet18 Transfer Learning
│   ├── train.py                    # Vòng lặp huấn luyện, Early Stopping, Checkpoint
│   ├── evaluate.py                 # Đánh giá độc lập trên tập Test, phân tích lỗi
│   └── predict.py                  # Module suy luận cho 1 ảnh bất kỳ
│
├── app.py                          # Ứng dụng Web demo trực quan bằng Streamlit
├── check_env.py                    # Script kiểm tra phiên bản môi trường & thiết bị
├── requirements.txt                # Danh sách thư viện phụ thuộc
└── README.md                       # Tài liệu hướng dẫn và báo cáo kết quả
```

---

## 🚀 5. Hướng dẫn cài đặt và thực thi

### Bước 1: Mở Terminal tại thư mục dự án
```powershell
cd "d:\Software\Y3S1 - AI\fabric-material-classification"
```

### Bước 2: Cài đặt các thư viện phụ thuộc
```powershell
pip install -r requirements.txt
```

### Bước 3: Kiểm tra môi trường hệ thống
```powershell
python check_env.py
```

### Bước 4: Kiểm tra Pipeline xử lý dữ liệu
```powershell
python src/dataset.py
```

### Bước 5: Kiểm tra kiến trúc mô hình ResNet18
```powershell
python src/model.py
```

### Bước 6: Huấn luyện mô hình (Training)
```powershell
python src/train.py
```
*Mô hình tự động lưu checkpoint có Validation Accuracy cao nhất vào `models/best_model.pth` và vẽ đồ thị vào `outputs/training_curve.png`.*

### Bước 7: Đánh giá mô hình trên tập Test (Evaluation)
```powershell
python src/evaluate.py
```
*Kết quả sẽ xuất ra ma trận nhầm lẫn `outputs/confusion_matrix.png` và bảng báo cáo `outputs/classification_report.txt`.*

### Bước 8: Kiểm tra suy luận trên ảnh đơn lẻ (Inference)
```powershell
python src/predict.py
```

### Bước 9: Khởi chạy ứng dụng Web Streamlit
```powershell
streamlit run app.py
```
*Trình duyệt sẽ tự động mở giao diện tại địa chỉ `http://localhost:8501` để upload ảnh và trải nghiệm demo.*

---

## 📈 6. KẾT QUẢ THỰC NGHIỆM THỰC TẾ (EXPERIMENTAL RESULTS)

*Lưu ý: Toàn bộ số liệu dưới đây được đo lường thực tế từ quá trình huấn luyện và đánh giá trên tập Test độc lập (393 ảnh), không sử dụng số liệu giả định.*

### 6.1. Tổng quan hiệu năng mô hình
| Chỉ số (Metric) | Giá trị thực nghiệm |
| :--- | :--- |
| **Kỷ lục Validation Accuracy** | **71.43%** (tại Epoch 4) |
| **Test Accuracy** | **68.70%** |
| **Macro Precision** | **68.97%** |
| **Macro Recall** | **69.28%** |
| **Macro F1-Score** | **68.86%** |
| **Weighted F1-Score** | **68.59%** |

### 6.2. Hiệu năng chi tiết trên từng loại chất liệu vải
| Chất liệu vải (Class) | Precision | Recall | F1-Score | Số lượng mẫu (Support) |
| :--- | :---: | :---: | :---: | :---: |
| 👖 **Denim** | **82.81%** | **88.33%** | **85.48%** *(Tốt nhất)* | 60 |
| 🧥 **Wool** | **90.00%** | **77.59%** | **83.33%** | 58 |
| 🧣 **Silk** | **72.55%** | **86.05%** | **78.72%** | 43 |
| 🧵 **Cotton** | **66.67%** | **63.33%** | **64.96%** | 60 |
| 👚 **Viscose** | **55.56%** | **62.50%** | **58.82%** | 56 |
| 👕 **Polyester** | **55.17%** | **57.14%** | **56.14%** | 56 |
| 🧶 **Cotton Mixed** | **60.00%** | **50.00%** | **54.55%** | 60 |

### 6.3. Phân tích nguyên nhân và các cặp dễ nhầm lẫn
1. **Denim, Wool và Silk đạt độ chính xác cao nhất (F1 > 78 - 85%):**
   - *Denim* có cấu trúc dệt chéo đặc trưng (twill weave) và màu chàm tương phản rõ nét.
   - *Wool* có độ xơ sợi tự nhiên gồ ghề, cấu trúc sợi len cuộn đặc trưng.
   - *Silk* có bề mặt mịn, độ bắt sáng và phản chiếu bề mặt cao.
2. **Các cặp chất liệu dễ bị phân loại nhầm:**
   - **Viscose ↔ Polyester (18 mẫu nhầm lẫn):** Đều là dạng sợi nhân tạo/bán tổng hợp có bề mặt dệt phẳng và độ phản quang tương đồng.
   - **Cotton ↔ Cotton Mixed (15 mẫu nhầm lẫn):** Bản chất *Cotton Mixed* chứa từ 50% đến 80% sợi Cotton thật, dẫn đến mật độ dệt và kết cấu bề mặt có độ tương đồng sinh học rất cao.

---

## 🎓 7. BỘ TÀI LIỆU PHẢN BIỆN DÀNH CHO THUYẾT TRÌNH (7 Ý CHÍNH)

Phần này được biên soạn ngắn gọn, súc tích để sinh viên tự tin trả lời câu hỏi vấn đáp trước hội đồng:

### 1. Image Classification là gì?
- **Khái niệm:** Là bài toán cơ bản trong Computer Vision, nhận đầu vào là một ma trận điểm ảnh $3 \times 224 \times 224$ và dự đoán nhãn rời rạc tương ứng trong tập $K$ lớp cố định.
- **Trong đề tài:** Phân loại bề mặt sợi vải thành 1 trong 7 nhóm chất liệu dệt may.

### 2. Vì sao lựa chọn kiến trúc ResNet18?
- **Giải quyết triệt để hiện tượng tiêu biến Gradient (Vanishing Gradient):** Các mạng nơ-ron truyền thống càng sâu thì gradient khi lan truyền ngược càng bị suy giảm về 0. ResNet giới thiệu cơ chế **Skip Connection (Residual Block)**:
  $$\mathbf{y} = \mathcal{F}(\mathbf{x}) + \mathbf{x}$$
  cho phép dòng gradient chảy trực tiếp qua các tầng mà không bị suy hao.
- **Kích thước gọn nhẹ và tốc độ tối ưu:** Với ~11.18 triệu tham số, ResNet18 vừa đủ mạnh để học các đặc trưng texture vi mô của vải, vừa nhẹ để huấn luyện và suy luận nhanh trên cả CPU thông thường.

### 3. Transfer Learning là gì và áp dụng ra sao?
- **Khái niệm:** Là kỹ thuật tận dụng tri thức đã được học từ một mô hình được đào tạo trên tập dữ liệu khổng lồ (ImageNet với 1.2 triệu ảnh, 1000 lớp) để giải quyết bài toán mới với tập dữ liệu nhỏ hơn.
- **Áp dụng vào đề tài:**
  - Giữ lại các tầng tích chập ban đầu (Backbone Feature Extractor) vì chúng đã thành thạo việc nhận diện các cạnh, góc, hoa văn và kết cấu sợi vải.
  - Thay thế tầng phân loại cuối cùng (`model.fc`) bằng một tầng `Linear(512, 7)` mới phù hợp với 7 loại vải.
  - Áp dụng chiến lược 2 bước: Bước 1 đóng băng backbone để huấn luyện lớp phân loại mới, sau đó Bước 2 fine-tuning nhẹ nhàng toàn bộ mạng.

### 4. Dataset được tiền xử lý và chia như thế nào?
- **Chuẩn hóa kích thước:** Resize tất cả ảnh về $224 \times 224$ px.
- **Chuẩn hóa điểm ảnh (ImageNet Normalization):** Đưa phân phối điểm ảnh về $\mathcal{N}(0, 1)$ giúp Gradient Descent hội tụ nhanh và mượt mà hơn.
- **Data Augmentation ở tập Train:** Lật ngang (Horizontal Flip), xoay nhẹ ($\pm 15^\circ$), đổi màu nhẹ (Color Jitter) giúp mô hình không phụ thuộc vào hướng đặt mẫu vải, chống hiện tượng học vẹt (**Overfitting**).
- **Chống rò rỉ dữ liệu (No Data Leakage):** Tập Validation và Test được cô lập hoàn toàn, chỉ áp dụng các biến đổi tất định (Resize + Normalize) để phản ánh trung thực bài toán thực tế.

### 5. Mô hình được huấn luyện (Train) như thế nào?
- **Hàm mất mát (Loss Function):** `CrossEntropyLoss` - hàm tối ưu tiêu chuẩn cho bài toán phân loại đa lớp:
  $$\mathcal{L} = -\sum_{i=1}^{C} y_i \log(\hat{y}_i)$$
- **Thuật toán tối ưu (Optimizer):** `AdamW` với cơ chế phân rã trọng số (Weight Decay) giúp kiểm soát độ phức tạp mô hình.
- **Cơ chế Early Stopping & Best Checkpoint:** Theo dõi liên tục độ chính xác trên tập Validation. Tự động lưu checkpoint tốt nhất và ngắt huấn luyện nếu mô hình không còn cải thiện để tránh lãng phí tài nguyên và hạn chế overfitting.

### 6. Ý nghĩa của các chỉ số: Accuracy, Precision, Recall, F1-Score?
- **Accuracy (Độ chính xác tổng quát):** Tỷ lệ dự đoán đúng trên toàn bộ tập dữ liệu:
  $$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$
- **Precision (Độ chuẩn xác):** Trong các mẫu mà mô hình *nói là loại vải X*, có bao nhiêu % thực sự là vải X:
  $$\text{Precision} = \frac{TP}{TP + FP}$$
- **Recall (Độ nhạy / Độ bao phủ):** Trong tất cả các mẫu *vải X thực tế*, mô hình tìm ra được bao nhiêu %:
  $$\text{Recall} = \frac{TP}{TP + FN}$$
- **F1-Score:** Trung bình điều hòa giữa Precision và Recall. Đây là thước đo công bằng nhất khi đánh giá hiệu năng từng lớp độc lập:
  $$\text{F1} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

### 7. Ma trận nhầm lẫn (Confusion Matrix) cho biết điều gì?
- Là bảng vuông kích thước $7 \times 7$ đối chiếu giữa **Nhãn thực tế (True Label)** và **Nhãn dự đoán (Predicted Label)**.
- **Đường chéo chính:** Thể hiện số lượng mẫu dự đoán chính xác của từng loại vải (càng đậm càng tốt).
- **Các ô nằm ngoài đường chéo:** Chỉ rõ mô hình đang nhầm loại vải nào với loại vải nào (Ví dụ: giữa *Cotton* và *Cotton Mixed*, hoặc *Polyester* và *Silk* do đặc tính dệt sợi hoặc bề mặt bóng tương đồng). Nhờ đó, người nghiên cứu có thể đưa ra giải pháp cải tiến chuyên sâu về đặc trưng dữ liệu.

---
*Dự án hoàn thành phục vụ học tập và nghiên cứu.*
