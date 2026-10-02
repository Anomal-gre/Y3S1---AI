"""
Ứng dụng Web Fabric Material AI - Giao diện Vector Icons cao cấp chuẩn Dribbble Showcase.
Hỗ trợ song song cả 2 cách: Tải ảnh từ máy tính VÀ Bốc ngẫu nhiên từ tập Test.
Khắc phục 100% xung đột state giữa file tải lên và nút chọn ngẫu nhiên bằng Dynamic Uploader Key.
"""

import os
import sys
import glob
import random
from PIL import Image
import streamlit as st

# Thiết lập đường dẫn import
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from src.predict import FabricPredictor

# Cấu hình trang Streamlit mở rộng (Responsive Wide)
st.set_page_config(
    page_title="Fabric Material AI - Design Showcase",
    page_icon="🧵",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# HỆ THỐNG VECTOR SVG ICONS DẠNG 1 DÒNG CHUẨN
SVG_ICONS = {
    "Denim": '<svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M4 3h16l1 18-5.5-1-3.5-9-3.5 9-5.5 1 1-18z"/><path d="M8 3v4"/><path d="M16 3v4"/><path d="M4 8h16"/></svg>',
    "Wool": '<svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><path d="M8 12c0-2.2 1.8-4 4-4s4 1.8 4 4"/><path d="M6 14.5c1.5-3 5-4.5 8.5-2.5"/><path d="M9.5 18c3 1 7-0.5 8-4.5"/></svg>',
    "Silk": '<svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M2 17c4-4 7 2 11-1.5s4-7 9-7.5"/><path d="M2 12c4-4 7 2 11-1.5s4-7 9-7.5"/><path d="M2 7c4-4 7 2 11-1.5s4-7 9-7.5"/></svg>',
    "Cotton": '<svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3a4 4 0 0 0-4 4 4 4 0 0 0-4 4c0 3 2.5 5 5 5h6c2.5 0 5-2 5-5a4 4 0 0 0-4-4 4 4 0 0 0-4-4z"/><path d="M12 16v5"/><path d="M9 19c2 0 3 2 3 2s1-2 3-2"/></svg>',
    "Viscose": '<svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2C6.5 2 2 6.5 2 12c0 5 3.5 9 8.5 9.8"/><path d="M12 2c5.5 0 10 4.5 10 10 0 5-3.5 9-8.5 9.8"/><path d="M12 2v20"/>{<path d="M7 8l5 4 5-4"/><path d="M7 14l5 4 5-4"/>}</svg>',
    "Polyester": '<svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><polygon points="12 2 21 7 21 17 12 22 3 17 3 7 12 2"/><line x1="12" y1="22" x2="12" y2="12"/><line x1="21" y1="7" x2="12" y2="12"/><line x1="3" y1="7" x2="12" y2="12"/></svg>',
    "Cotton Mixed": '<svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><circle cx="8" cy="12" r="5"/><circle cx="16" cy="12" r="5"/><path d="M12 8a5 5 0 0 1 0 8"/></svg>',
    "Logo": '<svg viewBox="0 0 24 24" width="24" height="24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="M12 2a14.5 14.5 0 0 0 0 20 14.5 14.5 0 0 0 0-20"/><path d="M2 12h20"/></svg>'
}

CLASS_SUBS = {
    "Denim": "Vải Jeans dệt chéo",
    "Wool": "Len lông cừu tự nhiên",
    "Silk": "Lụa tơ tằm mềm mịn",
    "Cotton": "Bông dệt sợi tự nhiên",
    "Viscose": "Vải bán tổng hợp mềm mại",
    "Polyester": "Sợi tổng hợp trơn bóng",
    "Cotton Mixed": "Bông pha sợi tổng hợp"
}

CLASS_COLORS = {
    "Denim": "#60A5FA",
    "Wool": "#FBBF24",
    "Silk": "#F472B6",
    "Cotton": "#34D399",
    "Viscose": "#A78BFA",
    "Polyester": "#38BDF8",
    "Cotton Mixed": "#FB923C"
}

# Custom CSS cho giao diện Vector Icons
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    background: radial-gradient(circle at 30% 30%, #1e3a8a 0%, #0d1b3e 35%, #020617 90%) !important;
    color: #ffffff !important;
    min-height: 100vh;
}

[data-testid="stHeader"] { background: transparent !important; }
#MainMenu, footer, header { visibility: hidden; }

.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 2rem !important;
    max-width: 1080px !important;
    margin: 0 auto !important;
}

.top-meta {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1.2rem;
    padding: 0 6px;
    font-size: 0.9rem;
    font-weight: 600;
    color: #94A3B8;
}
.top-meta span.highlight {
    color: #E2E8F0;
}

/* ÁP DỤNG TRỰC TIẾP KHUNG KÍNH CHO CỘT TRÁI CHÍNH */
[data-testid="stColumn"]:first-child > div:first-child {
    background: linear-gradient(180deg, #131c33 0%, #0c1222 100%);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 32px;
    padding: 24px;
    box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.7), 0 0 50px -10px rgba(37, 99, 235, 0.25);
    height: 100%;
}

/* TRIỆT TIÊU HOÀN TOÀN BACKGROUND VÀ PADDING CHO CÁC CỘT CON LỒNG NHAU (FIX LỆCH NÚT) */
[data-testid="stColumn"] [data-testid="stColumn"] > div:first-child,
[data-testid="stColumn"] [data-testid="stColumn"]:first-child > div:first-child {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
    box-shadow: none !important;
    border-radius: 0 !important;
    height: auto !important;
}

.panel-title {
    font-size: 1.15rem;
    font-weight: 700;
    color: #F8FAFC;
    margin-bottom: 4px;
}
.panel-subtitle {
    font-size: 0.8rem;
    color: #94A3B8;
    margin-bottom: 14px;
}

.divider-tag {
    font-size: 0.75rem;
    font-weight: 700;
    color: #94A3B8;
    margin: 14px 0 8px 0;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    display: flex;
    align-items: center;
    gap: 8px;
}
.divider-tag::before, .divider-tag::after {
    content: '';
    flex: 1;
    height: 1px;
    background: rgba(255, 255, 255, 0.1);
}

/* KHỐI PHẢI: DRIBBLE SHOWCASE CARD */
.right-showcase-card {
    background: linear-gradient(180deg, #151d33 0%, #0c1220 100%);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 36px;
    padding: 20px;
    box-shadow: 0 30px 60px -15px rgba(0, 0, 0, 0.8), 0 0 70px -10px rgba(37, 99, 235, 0.4);
}

/* KHỐI HERO XANH TRÊN CÙNG */
.hero-blue-card {
    background: linear-gradient(180deg, #2563EB 0%, #1D4ED8 60%, #1E40AF 100%);
    border-radius: 28px;
    padding: 22px 20px;
    box-shadow: inset 0 1px 2px rgba(255, 255, 255, 0.4), 0 14px 28px -6px rgba(29, 78, 216, 0.5);
    color: white;
    text-align: center;
    margin-bottom: 16px;
}

.hero-top-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 10px;
}
.user-pill {
    display: flex;
    align-items: center;
    gap: 8px;
    background: rgba(255, 255, 255, 0.18);
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
}
.info-btn {
    background: rgba(255, 255, 255, 0.18);
    width: 26px;
    height: 26px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.75rem;
    font-weight: 700;
}

.hero-tag {
    display: inline-block;
    background: rgba(255, 255, 255, 0.2);
    padding: 4px 14px;
    border-radius: 14px;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 6px;
}

.hero-metric-value {
    font-size: 2.8rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    margin: 2px 0 4px 0;
    line-height: 1.1;
    text-shadow: 0 2px 10px rgba(0, 0, 0, 0.25);
}

.hero-material-name {
    font-size: 1.25rem;
    font-weight: 700;
    letter-spacing: 0.03em;
    margin-bottom: 16px;
    text-shadow: 0 2px 6px rgba(0, 0, 0, 0.2);
}

/* DÃY AVATAR VECTOR */
.avatar-row {
    display: flex !important;
    justify-content: center !important;
    gap: 10px !important;
    align-items: center !important;
    flex-wrap: nowrap !important;
}
.avatar-item {
    width: 44px;
    height: 44px;
    border-radius: 50%;
    background: rgba(255, 255, 255, 0.16);
    display: inline-flex;
    align-items: center;
    justify-content: center;
    border: 2px solid rgba(255, 255, 255, 0.3);
    box-shadow: 0 4px 10px rgba(0,0,0,0.15);
    transition: all 0.3s ease;
    color: #E2E8F0;
}
.avatar-item:hover {
    transform: translateY(-2px);
    background: rgba(255, 255, 255, 0.25);
    color: #FFFFFF;
}
.avatar-item.active {
    background: #FFFFFF !important;
    transform: scale(1.22);
    border: 3px solid #60A5FA !important;
    box-shadow: 0 0 20px rgba(255, 255, 255, 0.9), 0 4px 12px rgba(0,0,0,0.3);
    color: #1E40AF !important;
}

/* KHỐI KÍNH TỐI MỜ (MIDDLE GLASS) */
.glass-section {
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(255, 255, 255, 0.07);
    border-radius: 26px;
    padding: 16px 18px;
}
.glass-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
    font-size: 0.8rem;
    color: #94A3B8;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.material-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 0;
    border-bottom: 1px solid rgba(255, 255, 255, 0.05);
}
.material-row:last-child {
    border-bottom: none;
}
.material-info {
    display: flex;
    align-items: center;
    gap: 12px;
}
.material-icon-box {
    width: 38px;
    height: 38px;
    border-radius: 12px;
    background: rgba(59, 130, 246, 0.12);
    border: 1px solid rgba(59, 130, 246, 0.25);
    display: flex;
    align-items: center;
    justify-content: center;
}
.material-name {
    font-size: 0.9rem;
    font-weight: 600;
    color: #F8FAFC;
}
.material-sub {
    font-size: 0.7rem;
    color: #94A3B8;
}
.material-percentage {
    font-size: 1rem;
    font-weight: 700;
    color: #F8FAFC;
    text-align: right;
}

.prob-bar-bg {
    width: 80px;
    height: 5px;
    background: rgba(255, 255, 255, 0.08);
    border-radius: 4px;
    margin-top: 3px;
    overflow: hidden;
}
.prob-bar-fill {
    height: 100%;
    background: linear-gradient(90deg, #3B82F6 0%, #60A5FA 100%);
    border-radius: 4px;
}

/* ĐỊNH DẠNG NÚT BẤM (BUTTONS) */
.stButton>button {
    border-radius: 20px !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    padding: 11px 18px !important;
    transition: all 0.25s ease !important;
    background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%) !important;
    color: #F8FAFC !important;
    border: 1px solid rgba(255, 255, 255, 0.18) !important;
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.4) !important;
    width: 100% !important;
}
.stButton>button:hover {
    background: #2563EB !important;
    border-color: #60A5FA !important;
    color: white !important;
    transform: translateY(-2px);
    box-shadow: 0 8px 18px rgba(37, 99, 235, 0.5) !important;
}

button[kind="primary"] {
    background: linear-gradient(180deg, #3B82F6 0%, #2563EB 60%, #1D4ED8 100%) !important;
    color: white !important;
    font-size: 1rem !important;
    padding: 13px 20px !important;
    border-radius: 22px !important;
    border: 1px solid rgba(255, 255, 255, 0.3) !important;
    box-shadow: inset 0 1px 2px rgba(255, 255, 255, 0.4), 0 10px 25px -4px rgba(37, 99, 235, 0.7) !important;
    width: 100% !important;
}
button[kind="primary"]:hover {
    transform: translateY(-2px);
    box-shadow: inset 0 1px 2px rgba(255, 255, 255, 0.5), 0 14px 30px -4px rgba(37, 99, 235, 0.9) !important;
}

[data-testid="stFileUploaderDropzone"] {
    background: rgba(255, 255, 255, 0.03) !important;
    border: 1.5px dashed rgba(96, 165, 250, 0.35) !important;
    border-radius: 18px !important;
    padding: 12px !important;
    color: #E2E8F0 !important;
}
[data-testid="stFileUploaderDropzone"]:hover {
    border-color: #60A5FA !important;
    background: rgba(37, 99, 235, 0.08) !important;
}
[data-testid="stFileUploaderDropzone"] button {
    background: rgba(37, 99, 235, 0.25) !important;
    border: 1px solid rgba(96, 165, 250, 0.4) !important;
    color: #FFFFFF !important;
    border-radius: 12px !important;
}

.bottom-meta {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-top: 1.2rem;
    padding: 0 6px;
    font-size: 0.8rem;
    color: #64748B;
}

/* ================================================================= */
/* LOADING SCREEN DẠNG FULLSCREEN OVERLAY FADE-OUT                   */
/* ================================================================= */
@keyframes overlayFadeOut {
    0%   { opacity: 1; visibility: visible; pointer-events: all; }
    75%  { opacity: 1; visibility: visible; pointer-events: all; }
    95%  { opacity: 0; visibility: visible; pointer-events: none; }
    100% { opacity: 0; visibility: hidden;  pointer-events: none; display: none; }
}

@keyframes fillProgressAnimation {
    0%   { width: 0%; }
    30%  { width: 35%; }
    70%  { width: 75%; }
    100% { width: 100%; }
}

#loading-overlay {
    position: fixed;
    top: 0;
    left: 0;
    width: 100vw;
    height: 100vh;
    background: radial-gradient(circle at 30% 30%, #1e3a8a 0%, #0d1b3e 35%, #020617 90%);
    z-index: 9999999;
    display: flex;
    align-items: center;
    justify-content: center;
    animation: overlayFadeOut 2.2s ease-in-out forwards;
}

.loading-card {
    background: linear-gradient(180deg, #131c33 0%, #0c1222 100%);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 36px;
    padding: 44px 36px;
    text-align: center;
    max-width: 440px;
    width: 90%;
    box-shadow: 0 30px 60px -15px rgba(0, 0, 0, 0.8), 0 0 80px -10px rgba(37, 99, 235, 0.4);
}

.loading-orb {
    width: 72px;
    height: 72px;
    border-radius: 50%;
    margin: 0 auto 20px auto;
    background: linear-gradient(180deg, #3B82F6 0%, #1D4ED8 100%);
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    box-shadow: 0 0 30px rgba(37, 99, 235, 0.6), inset 0 1px 2px rgba(255, 255, 255, 0.5);
    animation: pulseGlow 1.8s infinite alternate ease-in-out;
}
@keyframes pulseGlow {
    from { transform: scale(1); box-shadow: 0 0 20px rgba(37, 99, 235, 0.4); }
    to { transform: scale(1.08); box-shadow: 0 0 40px rgba(96, 165, 250, 0.8); }
}

.loading-title {
    font-size: 1.45rem;
    font-weight: 800;
    color: #F8FAFC;
    letter-spacing: -0.02em;
    margin-bottom: 6px;
}
.loading-subtitle {
    font-size: 0.82rem;
    color: #94A3B8;
    margin-bottom: 26px;
}
.loading-progress-bg {
    width: 100%;
    height: 8px;
    background: rgba(255, 255, 255, 0.08);
    border-radius: 6px;
    overflow: hidden;
    margin-bottom: 14px;
}
.loading-progress-fill {
    height: 100%;
    background: linear-gradient(90deg, #2563EB 0%, #60A5FA 100%);
    border-radius: 6px;
    animation: fillProgressAnimation 1.9s ease-in-out forwards;
    box-shadow: 0 0 12px rgba(96, 165, 250, 0.6);
}
.loading-status-text {
    font-size: 0.78rem;
    color: #CBD5E1;
    font-weight: 600;
}
</style>""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def get_predictor():
    p = os.path.join(current_dir, "models", "best_model.pth")
    return FabricPredictor(model_path=p) if os.path.exists(p) else None


def main():
    predictor = get_predictor()

    # Màn hình loading screen dạng overlay 2.2s chỉ xuất hiện khi F5 hoặc mở trang lần đầu
    if "first_visit" not in st.session_state:
        st.session_state.first_visit = True

    if st.session_state.first_visit:
        overlay_html = (
            '<div id="loading-overlay">'
            '<div class="loading-card">'
            f'<div class="loading-orb">{SVG_ICONS["Logo"]}</div>'
            '<div class="loading-title">Fabric Material AI</div>'
            '<div class="loading-subtitle">Hệ thống nhận diện chất liệu vải tự động</div>'
            '<div class="loading-progress-bg"><div class="loading-progress-fill"></div></div>'
            '<div class="loading-status-text">Đang nạp mô hình ResNet18 & Giao diện...</div>'
            '</div>'
            '</div>'
        )
        st.markdown(overlay_html, unsafe_allow_html=True)
        st.session_state.first_visit = False

    st.markdown('<div class="top-meta"><div><span class="highlight">Design Showcase</span> • Fabric Material AI</div><div>01/07 • ResNet18 Pretrained</div></div>', unsafe_allow_html=True)

    # Khởi tạo session state
    if "current_image" not in st.session_state:
        st.session_state.current_image = None
    if "res" not in st.session_state:
        st.session_state.res = None
    if "current_image_info" not in st.session_state:
        st.session_state.current_image_info = ""
    if "last_image_path" not in st.session_state:
        st.session_state.last_image_path = None
    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 0
    if "last_uploaded_signature" not in st.session_state:
        st.session_state.last_uploaded_signature = None

    col_left, col_right = st.columns([1, 1.2], gap="large")

    # =========================================================================
    # CỘT TRÁI: HỖ TRỢ CẢ TẢI ẢNH LÊN VÀ CHỌN NGẪU NHIÊN TỪ TEST SET
    # =========================================================================
    with col_left:
        st.markdown('<div class="panel-title">Tải ảnh bề mặt vải vào AI</div><div class="panel-subtitle">Kéo thả ảnh từ máy tính hoặc bấm bốc ngẫu nhiên từ tập Test</div>', unsafe_allow_html=True)

        # 1. KHỐI TẢI ẢNH LÊN (Dùng dynamic key để xóa sạch khi bấm ngẫu nhiên)
        uploaded = st.file_uploader(
            "Tải file ảnh vải",
            type=["jpg", "png", "jpeg"],
            key=f"fabric_uploader_{st.session_state.uploader_key}",
            label_visibility="collapsed"
        )

        # Xử lý khi người dùng chọn tải ảnh
        if uploaded is not None:
            upload_sig = (uploaded.name, uploaded.size)
            if upload_sig != st.session_state.last_uploaded_signature:
                st.session_state.last_uploaded_signature = upload_sig
                try:
                    uploaded_img = Image.open(uploaded).convert("RGB")
                    st.session_state.current_image = uploaded_img
                    st.session_state.current_image_info = f"Ảnh tải lên: {uploaded.name}"
                    if predictor:
                        st.session_state.res = predictor.predict(uploaded_img)
                    st.rerun()
                except Exception:
                    pass

        # 2. KHỐI HOẶC CHỌN NHANH MẪU TỪ TẬP TEST (CĂN GIỮA)
        st.markdown('<div class="divider-tag">Hoặc chọn nhanh mẫu từ tập Test</div>', unsafe_allow_html=True)

        test_dir = os.path.join(current_dir, "dataset", "test")

        # NÚT BẤM CHỌN NGẪU NHIÊN ĐƯỢC CĂN GIỮA
        _, btn_center_col, _ = st.columns([0.08, 0.84, 0.08])
        with btn_center_col:
            clicked_random = st.button(
                "Chọn ngẫu nhiên 1 mẫu vải trong 7 lớp",
                key="btn_random_sample",
                use_container_width=True
            )

        if clicked_random:
            if os.path.exists(test_dir):
                all_imgs = (
                    glob.glob(os.path.join(test_dir, "*", "*.jpg")) +
                    glob.glob(os.path.join(test_dir, "*", "*.png")) +
                    glob.glob(os.path.join(test_dir, "*", "*.jpeg"))
                )

                if all_imgs:
                    # Loại trừ ảnh vừa chọn để đảm bảo luôn đổi ảnh mới
                    candidates = [p for p in all_imgs if p != st.session_state.last_image_path]
                    chosen_path = random.choice(candidates if candidates else all_imgs)
                    
                    st.session_state.last_image_path = chosen_path
                    
                    # ĐỔI KEY UPLOADER: Xóa sạch file cũ trong uploader, không còn bị ghi đè!
                    st.session_state.uploader_key += 1
                    st.session_state.last_uploaded_signature = None
                    
                    new_img = Image.open(chosen_path).convert("RGB")
                    st.session_state.current_image = new_img
                    
                    true_label = os.path.basename(os.path.dirname(chosen_path))
                    st.session_state.current_image_info = f"Mẫu ngẫu nhiên: {true_label}"
                    
                    if predictor:
                        st.session_state.res = predictor.predict(new_img)
                    
                    st.rerun()

        # 3. KHUNG XEM TRƯỚC ẢNH VẢI ĐANG CHỌN
        st.markdown('<div style="margin-top:14px;"></div>', unsafe_allow_html=True)
        if st.session_state.current_image is not None:
            st.image(
                st.session_state.current_image,
                caption=st.session_state.current_image_info,
                use_container_width=True
            )
        else:
            st.info("Hãy kéo thả ảnh vào khung trên hoặc bấm 'Chọn ngẫu nhiên 1 mẫu vải' để bắt đầu.")

    # ==========================================
    # CỘT PHẢI: KHỐI THẺ DRIBBLE SHOWCASE CHUẨN MỰC
    # ==========================================
    with col_right:
        res = st.session_state.res
        if res:
            top_name = res["predicted_class"]
            top_conf = res["confidence"]
            all_probs = res["all_probabilities"]
        else:
            top_name = "CHỜ DỰ ĐOÁN"
            top_conf = 0.0
            all_probs = {c: 0.0 for c in ["Denim", "Wool", "Silk", "Cotton", "Viscose", "Polyester", "Cotton Mixed"]}

        fabric_order = ["Denim", "Wool", "Silk", "Cotton", "Viscose", "Polyester", "Cotton Mixed"]
        avatar_items_list = []
        for c in fabric_order:
            act = "active" if (res and c == top_name) else ""
            svg = SVG_ICONS.get(c, "")
            avatar_items_list.append(f'<div class="avatar-item {act}" title="{c}">{svg}</div>')
        avatar_html = "".join(avatar_items_list)

        list_items = []
        sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
        for c_name, prob in sorted_probs:
            svg_icon = SVG_ICONS.get(c_name, "")
            desc = CLASS_SUBS.get(c_name, "Chất liệu dệt")
            accent = CLASS_COLORS.get(c_name, "#93C5FD")
            fill_w = max(prob, 2.0) if prob > 0 else 0
            list_items.append(
                '<div class="material-row">'
                f'<div class="material-info">'
                f'<div class="material-icon-box" style="color: {accent};">{svg_icon}</div>'
                f'<div><div class="material-name">{c_name}</div><div class="material-sub">{desc}</div></div>'
                f'</div>'
                f'<div><div class="material-percentage">{prob:.1f}%</div>'
                f'<div class="prob-bar-bg"><div class="prob-bar-fill" style="width: {fill_w}%;"></div></div></div>'
                '</div>'
            )
        rows_html = "".join(list_items)

        showcase_html = (
            '<div class="right-showcase-card">'
            '<div class="hero-blue-card">'
            '<div class="hero-top-row">'
            f'<div class="user-pill">{SVG_ICONS["Logo"]}<span>Fabric Material AI</span></div>'
            '<div class="info-btn" title="7 Lớp chất liệu vải">i</div>'
            '</div>'
            '<div class="hero-tag">ĐỘ TIN CẬY (CONFIDENCE)</div>'
            f'<div class="hero-metric-value">{top_conf:.1f}%</div>'
            f'<div class="hero-material-name">{top_name.upper()}</div>'
            f'<div class="avatar-row">{avatar_html}</div>'
            '</div>'
            '<div class="glass-section">'
            '<div class="glass-header"><span>XÁC SUẤT CÁC NHÓM VẢI</span><span>7 CLASSES</span></div>'
            f'{rows_html}'
            '</div>'
            '</div>'
        )
        st.markdown(showcase_html, unsafe_allow_html=True)

    st.markdown('<div class="bottom-meta"><div>@fabric_material__ai • PyTorch ResNet18</div><div>Midterm Project • Save for later</div></div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
