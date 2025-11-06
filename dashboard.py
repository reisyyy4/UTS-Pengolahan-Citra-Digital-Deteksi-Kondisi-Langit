# dashboard.py (versi stepper: HSV → Blur → ROI → Mask → Hasil)
import streamlit as st
import cv2
import numpy as np
import tempfile
from io import BytesIO
from PIL import Image, ImageOps

from model import (
    deteksi_langit,
    deteksi_langit_v2,
    compute_masks_and_props,  # untuk tampilan preprocessing
    ROI_RATIO,
    THRESH,
)

# ======================
# Konfigurasi halaman
# ======================
st.set_page_config(page_title="Deteksi Kondisi Langit", page_icon="☀️", layout="wide")
st.title("🌤️ Deteksi Kondisi Langit")
st.caption("Unggah/ambil foto langit → **HSV → Gaussian Blur → ROI → Mask** → barulah tampil hasil klasifikasi.")

# ======================
# Helper utils
# ======================
def _rerun():
    """Rerun kompatibel untuk berbagai versi Streamlit."""
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()

def correct_orientation(image_bytes: bytes) -> Image.Image:
    """Koreksi EXIF supaya orientasi foto HP benar."""
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    try:
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass
    return img

def resize_for_preview(pil_img: Image.Image, max_long_edge: int) -> Image.Image:
    w, h = pil_img.size
    scale = max(w, h) / max_long_edge
    if scale <= 1:
        return pil_img
    new_size = (int(w/scale), int(h/scale))
    return pil_img.resize(new_size, Image.LANCZOS)

def pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    arr = np.array(pil_img)  # RGB
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

# ======================
# Sidebar opsi
# ======================
with st.sidebar:
    st.header("Pengaturan")
    focus_top = st.checkbox(
        "Fokus area atas (≈60%)",
        value=True,
        help="Membatasi analisis ke bagian atas gambar yang umumnya langit."
    )
    use_v2 = st.checkbox("Gunakan model v2 (lebih robust)", value=True)
    max_edge = st.slider("Batas sisi terpanjang pratinjau", 480, 2048, 1024, 64)
    st.caption("Tip: Arahkan kamera ke langit & minim foreground besar.")

# ======================
# Input gambar
# ======================
uploaded_file = st.file_uploader("Unggah gambar (.jpg, .jpeg, .png)", type=["jpg", "jpeg", "png"])
camera_file = st.camera_input("Atau ambil foto dengan kamera")
file_src = uploaded_file or camera_file

if not file_src:
    st.info("Silakan unggah atau ambil foto langit terlebih dahulu.")
    st.stop()

# ======================
# Preprocess tampilan & simpan file untuk model
# ======================
try:
    raw_bytes = file_src.getvalue()
    pil_img = correct_orientation(raw_bytes)              # koreksi EXIF
    preview_img = resize_for_preview(pil_img, max_edge)   # ringan untuk UI
    cv_bgr_preview = pil_to_bgr(preview_img)
    disp_rgb = cv2.cvtColor(cv_bgr_preview, cv2.COLOR_BGR2RGB)

    # Simpan versi yang sudah benar orientasinya ke file sementara (dipakai analisis)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        pil_img.save(tmp.name, format="JPEG", quality=95)
        image_path = tmp.name
except Exception as e:
    st.error(f"Gagal menyiapkan gambar: {e}")
    st.stop()

# ======================
# Pratinjau
# ======================
st.subheader("Pratinjau")
st.image(disp_rgb, caption="Gambar (preview; orientasi sudah benar)", use_container_width=True)

# ======================
# Inisialisasi stepper
# ======================
if "step" not in st.session_state:
    st.session_state.step = 1

# Reset step ke 1 bila pengguna ganti gambar / opsi utama
fingerprint = f"{len(raw_bytes)}-{preview_img.size}-{focus_top}-{use_v2}"
if st.session_state.get("fp") != fingerprint:
    st.session_state.step = 1
    st.session_state.fp = fingerprint

# ======================
# Siapkan data preprocessing (sekali per rerun)
# ======================
img_bgr_model = cv2.imread(image_path)
if img_bgr_model is None:
    st.error("Gagal membaca ulang gambar dari file sementara.")
    st.stop()

img_rgb_model = cv2.cvtColor(img_bgr_model, cv2.COLOR_BGR2RGB)
hsv = cv2.cvtColor(img_bgr_model, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv)

# Komputasi helper (blur & ROI di dalam helper)
(blue_mask, bright_mask, dark_mask,
 p_blue, p_bright, p_dark, roi_mask,
 p_global_dark, mean_v_global) = compute_masks_and_props(
    h, s, v, focus_top=focus_top, roi_ratio=ROI_RATIO
)
roi_bool = roi_mask.astype(bool)

# Visual untuk H & H_blur
h_vis = cv2.normalize(h, None, 0, 255, cv2.NORM_MINMAX)
h_blur_vis = cv2.GaussianBlur(h, (5, 5), 0)
h_blur_vis = cv2.normalize(h_blur_vis, None, 0, 255, cv2.NORM_MINMAX)

# Overlay ROI
overlay = img_rgb_model.copy()
alpha = 0.35
overlay[roi_bool] = (overlay[roi_bool] * (1 - alpha) + np.array([0, 255, 0]) * alpha).astype(np.uint8)

H, W = h.shape

# ======================
# Stepper UI
# ======================
st.markdown("### Langkah Analisis")
steps = [
    "1) HSV",
    "2) Gaussian Blur (Hue)",
    "3) ROI (bagian atas)",
    "4) Mask Kategori",
    "5) Hasil Klasifikasi"
]
# Tanpa argumen text agar kompatibel ke banyak versi Streamlit
st.progress(st.session_state.step / len(steps))
st.write(f"Langkah aktif: **{steps[st.session_state.step - 1]}**")

step = st.session_state.step

# Konten per langkah
if step == 1:
    st.subheader("1) Kanal HSV")
    c1, c2, c3 = st.columns(3)
    c1.image(h_vis, caption="Hue (0–179 → diskalakan)", use_container_width=True, clamp=True)
    c2.image(s, caption="Saturation (0–255)", use_container_width=True, clamp=True)
    c3.image(v, caption="Value (0–255)", use_container_width=True, clamp=True)
    st.caption(f"Resolusi dianalisis: {W}×{H}px")

elif step == 2:
    st.subheader("2) Hue setelah Gaussian Blur")
    st.image(h_blur_vis, caption="Hue setelah Gaussian Blur (5×5)", use_container_width=True, clamp=True)

elif step == 3:
    st.subheader("3) ROI (≈60% bagian atas)")
    st.image(overlay, caption=f"ROI atas (≈{int(ROI_RATIO*100)}%) ditandai hijau transparan", use_container_width=True)

elif step == 4:
    st.subheader("4) Mask Kategori")
    c1, c2, c3 = st.columns(3)
    c1.image(blue_mask,   caption=f"Blue Sky — proporsi ROI: {p_blue:.2f}",       use_container_width=True, clamp=True)
    c2.image(bright_mask, caption=f"Bright Cloud — proporsi ROI: {p_bright:.2f}", use_container_width=True, clamp=True)
    c3.image(dark_mask,   caption=f"Dark Cloud — proporsi ROI: {p_dark:.2f}",     use_container_width=True, clamp=True)

    st.caption(
        f"Ambang: blue(H∈[{THRESH['blue_H_min']},{THRESH['blue_H_max']}], S>{THRESH['blue_S_min']}), "
        f"bright(S<{THRESH['bright_S_max']}, V>{THRESH['bright_V_min']}), "
        f"dark(S<{THRESH['dark_S_max']}, V<{THRESH['dark_V_max']}); "
        f"night(V<{THRESH.get('night_V_max','?')}, proporsi≥{THRESH.get('night_p_dark_min','?')})"
    )
    st.markdown("---")
    cA, cB = st.columns(2)
    cA.metric("Proporsi Gelap Global (p_global_dark)", f"{p_global_dark:.2f}")
    cB.metric("Mean Value Global (mean V)", f"{mean_v_global:.1f}")

elif step == 5:
    st.subheader("5) Hasil Klasifikasi")
    with st.spinner("Menganalisis langit..."):
        try:
            if use_v2:
                _, hasil, info = deteksi_langit_v2(image_path, focus_top=focus_top)
                st.success(
                    f"**{hasil}** — "
                    f"p_blue={info.get('p_blue',0):.2f}, "
                    f"p_bright={info.get('p_bright_cloud',0):.2f}, "
                    f"p_dark={info.get('p_dark_cloud',0):.2f}"
                    + (f", p_global_dark={info.get('p_global_dark',0):.2f}" if 'p_global_dark' in info else "")
                )
            else:
                _, hasil = deteksi_langit(image_path)
                st.success(f"**{hasil}** (v1)")
        except Exception as e:
            st.error(f"Gagal menganalisis: {e}")

# ======================
# Navigasi stepper
# ======================
st.markdown("---")
col_prev, col_spacer, col_next = st.columns([1, 5, 1])

with col_prev:
    if st.session_state.step > 1:
        if st.button("⬅️ Sebelumnya"):
            st.session_state.step -= 1
            _rerun()

with col_next:
    if st.session_state.step < len(steps):
        if st.button("Berikutnya ➡️"):
            st.session_state.step += 1
            _rerun()
    else:
        st.button("Selesai ✅", disabled=True)
