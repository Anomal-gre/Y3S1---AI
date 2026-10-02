"""
Script kiểm tra môi trường hệ thống cho dự án Fabric Material AI.
Kiểm tra phiên bản Python, PyTorch, thiết bị (CPU/GPU) và các thư viện cần thiết.
"""

import sys

# Đảm bảo in tiếng Việt trên console Windows không bị lỗi cp1252
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

def check_environment():
    print("=" * 60)
    print("      KIỂM TRA MÔI TRƯỜNG DỰ ÁN FABRIC MATERIAL AI")
    print("=" * 60)
    
    # 1. Python version
    py_version = sys.version_info
    print(f"[*] Python version: {sys.version.split()[0]}", end="")
    if py_version >= (3, 10):
        print(" -> [OK] (Yêu cầu >= 3.10)")
    else:
        print(" -> [CẢNH BÁO] Cần Python >= 3.10")

    # 2. PyTorch & Device
    try:
        import torch
        print(f"[*] PyTorch version: {torch.__version__} -> [OK]")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[*] Thiết bị tính toán (Device): {device.upper()}")
        if torch.cuda.is_available():
            print(f"    - Tên GPU: {torch.cuda.get_device_name(0)}")
            print(f"    - Số lượng GPU: {torch.cuda.device_count()}")
        else:
            print("    - Đang sử dụng CPU (Project vẫn chạy bình thường trên CPU).")
    except ImportError:
        print("[!] PyTorch: CHƯA CÀI ĐẶT (chạy: pip install torch)")

    # 3. Torchvision
    try:
        import torchvision
        print(f"[*] Torchvision version: {torchvision.__version__} -> [OK]")
    except ImportError:
        print("[!] Torchvision: CHƯA CÀI ĐẶT (chạy: pip install torchvision)")

    # 4. Scikit-learn, PIL, NumPy, Matplotlib, Seaborn, Tqdm, Streamlit
    libraries = [
        ("numpy", "NumPy"),
        ("PIL", "Pillow"),
        ("sklearn", "Scikit-learn"),
        ("matplotlib", "Matplotlib"),
        ("seaborn", "Seaborn"),
        ("tqdm", "tqdm"),
        ("streamlit", "Streamlit")
    ]

    missing = []
    for mod_name, display_name in libraries:
        try:
            mod = __import__(mod_name)
            ver = getattr(mod, "__version__", "OK")
            print(f"[*] {display_name} version: {ver} -> [OK]")
        except ImportError:
            print(f"[!] {display_name}: CHƯA CÀI ĐẶT")
            missing.append(mod_name)

    print("=" * 60)
    if missing:
        print(f"[!] Còn thiếu các thư viện: {', '.join(missing)}")
        print("    Vui lòng chạy lệnh: pip install -r requirements.txt")
        print("=" * 60)
        return False
    else:
        print("[OK] MÔI TRƯỜNG ĐÃ SẴN SÀNG CHO CHECKPOINT TIẾP THEO!")
        print("=" * 60)
        return True

if __name__ == "__main__":
    check_environment()
