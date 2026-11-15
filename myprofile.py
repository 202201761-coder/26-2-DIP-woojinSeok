import csv
import io

import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont, ImageOps
from scipy import ndimage as ndi, signal
from skimage.feature import peak_local_max
from skimage.segmentation import watershed


st.set_page_config(page_title="HW06 | Object Counting", page_icon="🔎", layout="wide")

METHODS = [
    "자동 분석", "HW05 기본", "밝은 객체", "어두운 객체", "노란색·갈색 객체",
    "파란색 객체", "여러 색 객체", "원형 객체", "밝은 문자", "노란색·갈색 객체 (확장)",
]

METHOD_LABELS = {
    "자동 분석": "Auto",
    "HW05 기본": "HW05 (Original)",
    "밝은 객체": "Bright Objects",
    "어두운 객체": "Dark Objects",
    "노란색·갈색 객체": "Yellow / Brown Objects",
    "파란색 객체": "Blue Objects",
    "여러 색 객체": "Multicolor Objects",
    "원형 객체": "Circular Objects",
    "밝은 문자": "Bright Text",
    "노란색·갈색 객체 (확장)": "Yellow / Brown (Enhanced)",
}


@st.cache_data(show_spinner=False, max_entries=3)
def load_image(data):
    with Image.open(io.BytesIO(data)) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        original_size = image.size
        if image.width * image.height > 90_000_000:
            raise ValueError("The image exceeds the maximum pixel limit.")
        image.thumbnail((2300, 2300), Image.Resampling.LANCZOS)
        rgb = np.asarray(image).copy()
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    filtered = signal.medfilt2d(gray, 7)
    return rgb, gray, filtered, original_size


def auto_method(rgb, gray):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    channel = rgb.astype(np.int16)
    warm = np.minimum(channel[:, :, 0], channel[:, :, 1]) - channel[:, :, 2]
    blue = channel[:, :, 2] - np.maximum(channel[:, :, 0], channel[:, :, 1])
    warm_ratio = np.mean(warm > 28)
    blue_ratio = np.mean(blue > 28)
    colorful_ratio = np.mean(hsv[:, :, 1] > 65)
    white_ratio = np.mean(gray > 220)
    dark_ratio = np.mean(gray < 80)

    n, _, stats, _ = cv2.connectedComponentsWithStats((gray < 100).astype(np.uint8), 8)
    dark_blob = stats[1:, cv2.CC_STAT_AREA].max() / gray.size if n > 1 else 0

    if dark_ratio > 0.88 and white_ratio < 0.05:
        return "HW05 기본"
    if dark_blob > 0.26 and white_ratio > 0.10 and dark_ratio > 0.30:
        return "밝은 문자"
    if blue_ratio > 0.014 and colorful_ratio < 0.17:
        return "파란색 객체"
    if blue_ratio > 0.008 and colorful_ratio > 0.13:
        return "여러 색 객체"
    if warm_ratio > 0.007 and blue_ratio < 0.008:
        return "노란색·갈색 객체"
    if colorful_ratio > 0.09:
        return "여러 색 객체"
    if white_ratio > 0.26 and np.median(gray) > 160 and colorful_ratio < 0.025:
        return "원형 객체"
    return "밝은 객체" if np.median(gray) < 105 else "어두운 객체"


def otsu(gray):
    return cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]


def create_mask(rgb, gray, filtered, method, threshold, sensitivity):
    if method == "HW05 기본":
        return (filtered > threshold).astype(np.uint8)
    if method == "밝은 객체":
        return (filtered > np.clip(otsu(filtered) - sensitivity, 0, 255)).astype(np.uint8)
    if method == "어두운 객체":
        return (filtered < np.clip(otsu(filtered) + sensitivity, 0, 255)).astype(np.uint8)

    smooth = cv2.medianBlur(rgb, 5).astype(np.int16)
    red, green, blue = smooth[:, :, 0], smooth[:, :, 1], smooth[:, :, 2]

    if method == "노란색·갈색 객체 (확장)":
        warm = np.clip(np.minimum(red, green) - blue, 0, 255).astype(np.uint8)
        reference = cv2.GaussianBlur(warm, (0, 0), 15)
