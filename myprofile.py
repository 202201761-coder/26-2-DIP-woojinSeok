"""Homework #6 (Object Counting) implemented as a Streamlit web app.

Pipeline follows the original HW05 notebook:
1) grayscale, 2) 7x7 median filter, 3) thresholding,
4) scipy.ndimage.label (default 4-connectivity), 5) largest/smallest component.
"""

import io

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as nd
import scipy.signal as signal
import streamlit as st
from PIL import Image, UnidentifiedImageError

st.set_page_config(page_title="HW05 Object Counting", layout="wide")
st.title("HW06 - Object Counting")
st.write("사진을 업로드하면 HW05 실습의 1~5단계를 순서대로 수행합니다.")

# Streamlit에서는 Google Drive 마운트 대신 웹에서 직접 사진을 업로드합니다.
uploaded_file = st.file_uploader(
    "분석할 사진을 선택하세요", type=["jpg", "jpeg", "png", "tif", "tiff", "bmp"]
)

# 교수님 원본과 같은 초기값. 사용자가 임계값만 선택적으로 조정할 수 있습니다.
th_val = st.sidebar.slider("Threshold (임계값)", min_value=0, max_value=255, value=128)
st.sidebar.caption("Median filter: 7 × 7 | Connectivity: 4 (nd.label 기본값)")

if uploaded_file is None:
    st.info("먼저 IMG_9627.jpg 또는 다른 사진을 업로드하세요.")
    st.stop()

# 1. Object counting에 사용할 image를 grayscale로 불러오기
st.header("1. Grayscale 영상 불러오기")
try:
    I = Image.open(uploaded_file).convert("L")
except (UnidentifiedImageError, OSError) as error:
    st.error(f"사진을 열 수 없습니다: {error}")
    st.stop()
f = np.array(I)

fig, ax = plt.subplots(figsize=(4, 5))
image = ax.imshow(f, cmap="gray", vmin=0, vmax=255)
ax.axis("off")
fig.colorbar(image, ax=ax)

st.pyplot(fig, width=400)
plt.close(fig)

# 2. 적절한 전처리를 통해 영상의 noise를 줄이기
st.header("2. Median Filtering (Noise 제거)")
f_denoised = signal.medfilt2d(f, 7)
fig, ax = plt.subplots(figsize=(4, 5))
image = ax.imshow(f_denoised, cmap="gray", vmin=0, vmax=255)
ax.axis("off")
fig.colorbar(image, ax=ax)

st.pyplot(fig, width=400)
plt.close(fig)

# 3. 적절한 임계값을 찾아 thresholding을 통해 binary mask를 생성하기
st.header("3. Thresholding 및 Binary Mask")
col_hist, col_mask = st.columns(2)
with col_hist:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(f_denoised.ravel(), bins=256, range=(0, 256), color="gray")
    ax.axvline(th_val, color="red", linestyle="--", label=f"Threshold = {th_val}")
    ax.set_xlabel("Pixel Intensity")
    ax.set_ylabel("Frequency")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

f_mask = f_denoised > th_val
with col_mask:
    fig, ax = plt.subplots(figsize=(4, 5))
    ax.imshow(f_mask, cmap="gray")
    ax.axis("off")
    ax.set_title("Binary Mask")
    st.pyplot(fig, width=400)
    plt.close(fig)

# 4. Connected component 추출 및 각 객체의 pixel 수 계산
st.header("4. Connected Component Labeling")
labels, nums = nd.label(f_mask)  # 교수님 원본: 기본 4-connectivity

fig, ax = plt.subplots(figsize=(4, 5))
ax.imshow(labels, cmap="Spectral_r")
ax.axis("off")

st.pyplot(fig, width=400)
plt.close(fig)

st.metric("Connected Components", nums)
if nums == 0:
    st.warning("검출된 객체가 없습니다. 사이드바의 Threshold를 낮춰보세요.")
    st.stop()

# 교수님 원본의 마지막 라벨 누락 문제만 수정: nums-1 -> nums
cc = np.zeros((nums, 2))
for i in range(1, nums + 1):
    cc[i - 1, 0] = i
    cc[i - 1, 1] = np.sum(labels == i)

st.dataframe(
    {
        "Component Label": cc[:, 0].astype(int),
        "Number of pixels": cc[:, 1].astype(int),
    },
    hide_index=True,
)

# 5. 가장 큰 연결 요소와 작은 연결 요소 시각화
st.header("5. 최대·최소 Connected Component")
ind_max = np.argmax(cc[:, 1])
ind_min = np.argmin(cc[:, 1])
max_label = int(cc[ind_max, 0])
min_label = int(cc[ind_min, 0])
max_pixels = int(cc[ind_max, 1])
min_pixels = int(cc[ind_min, 1])

st.write(f"가장 큰 객체: Label **{max_label}**, **{max_pixels:,} pixels**")
st.write(f"가장 작은 객체: Label **{min_label}**, **{min_pixels:,} pixels**")

# 원본의 마지막 시각화 코드를 그대로 활용합니다.
fig = plt.figure(figsize=(16, 9))
plt.subplot(131)
plt.imshow(f, cmap="gray")
plt.axis("off")
plt.title("input image")
plt.subplot(132)
plt.imshow(labels == cc[ind_max, 0], cmap="Spectral_r")
plt.axis("off")
plt.title("the largest component")
plt.subplot(133)
plt.imshow(labels == cc[ind_min, 0], cmap="Spectral_r")
plt.axis("off")
plt.title("the smallest component")

st.pyplot(fig)
result_bytes = io.BytesIO()
fig.savefig(result_bytes, format="png", dpi=150, bbox_inches="tight")
plt.close(fig)
st.download_button(
    "최종 결과 이미지 다운로드",
    data=result_bytes.getvalue(),
    file_name="HW05_streamlit_result.png",
    mime="image/png",
)

st.info(
    "Connected Component는 실제 물체의 개수가 아니라 연결된 픽셀 영역의 개수입니다. "
    "서로 교차한 면봉은 하나의 연결 요소로 검출될 수 있습니다."
)

