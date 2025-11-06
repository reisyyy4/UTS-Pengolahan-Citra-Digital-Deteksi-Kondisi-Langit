# model.py
import cv2
import numpy as np
from typing import Tuple, Dict

# Ambang & parameter
THRESH = {
    # Siang / awan
    "blue_H_min": 95, "blue_H_max": 135, "blue_S_min": 60,
    "bright_S_max": 45, "bright_V_min": 170,
    "dark_S_max": 90,  "dark_V_max": 120,

    # Malam (global gelap)
    "night_V_max": 60,           # piksel dengan V < ini dianggap gelap sekali
    "night_p_dark_min": 0.65,    # proporsi piksel gelap global agar layak disebut "malam"

    # Pemisah Malam Cerah vs Malam Berawan (skyglow)
    "night_glow_V_min": 70,      # rentang V menengah (indikasi skyglow)
    "night_glow_V_max": 140,
    "night_glow_S_max": 60,      # S rendah = abu-abu/oranye pucat
    "night_glow_p_min": 0.22,    # proporsi ROI yang "glow" agar disebut malam berawan
    "night_clear_meanV_max": 55, # rata-rata V global sangat gelap → malam cerah
}

ROI_RATIO = 0.6  # 60% atas

def _read_image(image_path: str):
    return cv2.imread(image_path)

def _build_roi_mask(shape_hw: Tuple[int, int], ratio: float = ROI_RATIO) -> np.ndarray:
    H, W = shape_hw
    top_h = int(ratio * H)
    roi = np.zeros((H, W), dtype=np.uint8)
    roi[:top_h, :] = 255
    return roi

def compute_masks_and_props(h: np.ndarray, s: np.ndarray, v: np.ndarray,
                            focus_top: bool = True,
                            roi_ratio: float = ROI_RATIO) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                                                   float, float, float, np.ndarray,
                                                                   float, float]:
    """
    Mengembalikan 9 nilai:
    (blue_mask, bright_mask, dark_mask, p_blue, p_bright, p_dark, roi_mask, p_global_dark, mean_v_global)
    """
    # Blur H (2D)
    h_blur = cv2.GaussianBlur(h, (5, 5), 0)

    # ROI mask
    roi_mask = _build_roi_mask(h.shape, roi_ratio) if focus_top else np.full_like(h, 255, np.uint8)
    roi_bool = roi_mask.astype(bool)

    t = THRESH
    # Mask biner 0/255
    blue_mask   = ((h_blur >= t["blue_H_min"]) & (h_blur <= t["blue_H_max"]) & (s > t["blue_S_min"])).astype(np.uint8) * 255
    bright_mask = ((s < t["bright_S_max"]) & (v > t["bright_V_min"])).astype(np.uint8) * 255
    dark_mask   = ((s < t["dark_S_max"]) & (v < t["dark_V_max"])).astype(np.uint8) * 255

    # Proporsi di ROI
    total = max(1, int(np.count_nonzero(roi_bool)))
    p_blue   = float(np.count_nonzero(blue_mask[roi_bool]))   / total
    p_bright = float(np.count_nonzero(bright_mask[roi_bool])) / total
    p_dark   = float(np.count_nonzero(dark_mask[roi_bool]))   / total

    # Metrik global untuk malam
    global_dark_mask = (v < t["night_V_max"]).astype(np.uint8)
    total_all = max(1, v.size)
    p_global_dark = float(np.count_nonzero(global_dark_mask)) / total_all
    mean_v_global = float(np.mean(v))

    return blue_mask, bright_mask, dark_mask, p_blue, p_bright, p_dark, roi_mask, p_global_dark, mean_v_global

def deteksi_langit_v2(image_path: str, focus_top: bool = True):
    img = _read_image(image_path)
    if img is None:
        return None, "Gagal membaca gambar.", {}

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    (blue_mask, bright_mask, dark_mask,
     p_blue, p_bright, p_dark, roi_mask,
     p_global_dark, mean_v_global) = compute_masks_and_props(
        h, s, v, focus_top=focus_top, roi_ratio=ROI_RATIO
    )

    t = THRESH
    roi_bool = roi_mask.astype(bool)

    # Mask/prop untuk skyglow malam (hanya dipakai klasifikasi malam)
    night_glow_mask = (
        (v >= t["night_glow_V_min"]) &
        (v <= t["night_glow_V_max"]) &
        (s <= t["night_glow_S_max"])
    )
    total_roi = max(1, int(np.count_nonzero(roi_bool)))
    p_night_glow = float(np.count_nonzero(night_glow_mask[roi_bool])) / total_roi

    # ---------------- Aturan Keputusan ----------------
    # 1) Deteksi MALAM paling awal (global gelap, area sangat terang kecil, langit biru kecil)
    is_night = (p_global_dark >= t["night_p_dark_min"] and p_bright <= 0.10 and p_blue <= 0.15)
    if is_night:
        # Pisahkan Malam Cerah vs Malam Berawan (skyglow)
        if p_night_glow >= t["night_glow_p_min"] or mean_v_global > t["night_clear_meanV_max"]:
            kondisi = "Malam Berawan/Overcast 🌙☁️"
        else:
            kondisi = "Malam Cerah 🌙✨"

    # 2) Siang (tetap ada)
    elif p_blue >= 0.35 and p_bright <= 0.35 and p_dark <= 0.25:
        kondisi = "Langit Cerah ☀️"
    elif p_dark >= 0.30:
        kondisi = "Langit Mendung Gelap 🌧️"
    elif p_bright >= 0.40:
        kondisi = "Langit Berawan/Overcast ☁️"
    else:
        # Heuristik fallback di ROI
        sb = s[roi_bool]; vb = v[roi_bool]
        mean_s = float(np.mean(sb)) if sb.size else float(np.mean(s))
        mean_v = float(np.mean(vb)) if vb.size else float(np.mean(v))

        # Jika keseluruhan gelap tapi syarat malam utama tak sepenuhnya terpenuhi
        if mean_v_global < 70 and p_blue < 0.10:
            if p_night_glow >= t["night_glow_p_min"] or mean_v_global > t["night_clear_meanV_max"]:
                kondisi = "Malam Berawan/Overcast 🌙☁️"
            else:
                kondisi = "Malam Cerah 🌙✨"
        elif mean_s < 60 and mean_v > 150:
            kondisi = "Langit Berawan/Overcast ☁️"
        elif mean_s < 80 and mean_v < 130:
            kondisi = "Langit Mendung Gelap 🌧️"
        else:
            kondisi = "Langit Campuran/Parsial ⛅️"

    info: Dict[str, float] = {
        "p_blue": round(p_blue, 4),
        "p_bright_cloud": round(p_bright, 4),
        "p_dark_cloud": round(p_dark, 4),
        # Tambahan agar bisa dimunculkan di dashboard
        "p_global_dark": round(p_global_dark, 4),
        "mean_V_global": round(mean_v_global, 2),
        "p_night_glow": round(p_night_glow, 4),
    }
    return img, kondisi, info

# ------- Alias agar tampilan/flow lama tetap jalan -------
def deteksi_langit(image_path: str):
    """
    Alias ke v2 (kompatibilitas dengan kode lama).
    Mengembalikan (img_bgr, kondisi_str) seperti v1.
    """
    img, kondisi, _ = deteksi_langit_v2(image_path, focus_top=True)
    return img, kondisi
