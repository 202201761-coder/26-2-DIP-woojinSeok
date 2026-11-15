
import csv
import io

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as nd
import scipy.signal as signal
import streamlit as st
from PIL import Image, ImageDraw, ImageFont, ImageOps, UnidentifiedImageError


st.set_page_config(page_title="HW06 | 이미지 객체 계수", page_icon="🔍", layout="wide")


@st.cache_data(show_spinner=False)
def preprocess(image_bytes):
    with Image.open(io.BytesIO(image_bytes)) as image:
        image = ImageOps.exif_transpose(image)
        if image.width * image.height > 20_000_000:
            raise ValueError("이미지 크기는 2,000만 픽셀 이하여야 합니다.")
        original = np.array(image.convert("RGB"))
        gray = np.array(image.convert("L"))

    filtered = signal.medfilt2d(gray, 7)
    return original, gray, filtered


def detect_objects(filtered, threshold):
    mask = filtered > threshold
    labels, count = nd.label(mask)  # 4-connectivity
    areas = np.bincount(labels.ravel(), minlength=count + 1)[1:]
    return mask, labels, count, areas


def mark_objects(original, labels, areas):
    result = Image.fromarray(original)
    draw = ImageDraw.Draw(result)
    width, height = result.size
    line_width = max(2, min(width, height) // 400)
    font_size = max(14, min(width, height) // 65)

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()

    boxes = nd.find_objects(labels)
    for index in np.argsort(areas)[-150:]:
        rows, cols = boxes[int(index)]
        left, top = cols.start, rows.start
        right, bottom = cols.stop - 1, rows.stop - 1

        draw.rectangle(
            (left, top, right, bottom),
            outline="#32D1A0",
            width=line_width
        )
        draw.text(
            (left, max(0, top - font_size - 5)),
            str(index + 1),
            fill="white",
            font=font,
            stroke_width=2,
            stroke_fill="black"
        )

    return result


def color_labels(labels, count):
    colors = (
        plt.colormaps["Spectral_r"](
            np.linspace(0.05, 0.95, max(1, count))
        )[:, :3] * 255
    ).astype(np.uint8)

    palette = np.vstack(([35, 40, 55], colors)).astype(np.uint8)
    return palette[labels]


def preview(image):
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    image = image.copy()
    image.thumbnail((520, 620))
    return image


def download_png(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def download_csv(areas):
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["Label", "Number of pixels"])
    writer.writerows(
        (i, int(area)) for i, area in enumerate(areas, 1)
    )
    return buffer.getvalue().encode("utf-8-sig")


st.title("HW06 · 이미지 객체 계수")
st.caption("의료영상처리의 이론 및 실습 | HW05 Object Counting 기반 웹 애플리케이션")
st.write("이미지를 업로드하면 연결된 영역의 개수와 검출 결과를 확인할 수 있습니다.")

threshold = st.sidebar.slider("Threshold (임계값)", 0, 255, 128)
st.sidebar.caption("Median Filter: 7×7 · Connectivity: 4")

uploaded = st.file_uploader(
    "분석할 이미지 업로드",
    type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"]
)

if uploaded is None:
    st.info("분석할 이미지를 업로드해 주세요.")
    st.stop()

try:
    content = uploaded.getvalue()
    if len(content) > 25 * 1024 * 1024:
        raise ValueError("파일 크기는 25MB 이하여야 합니다.")
    original, gray, filtered = preprocess(content)
except (OSError, ValueError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
    st.error(f"이미지를 열 수 없습니다: {exc}")
    st.stop()

mask, labels, count, areas = detect_objects(filtered, threshold)
result = mark_objects(original, labels, areas)

st.subheader("객체 검출 결과")

a, b, c = st.columns(3)
a.metric("연결 요소 수", f"{count:,}개")
b.metric("가장 큰 영역", f"{areas.max():,} px" if count else "—")
c.metric("가장 작은 영역", f"{areas.min():,} px" if count else "—")

left, right = st.columns(2, gap="large")

with left:
    st.markdown("#### 원본 이미지")
    st.image(preview(original), width="stretch")

with right:
    st.markdown("#### 검출 결과")
    st.image(preview(result), width="stretch")

st.caption(
    f"{uploaded.name} · {gray.shape[1]} × {gray.shape[0]} px · Threshold {threshold}"
)
st.caption(
    "연결 요소 수는 실제 물체 수와 다를 수 있으며, "
    "서로 닿은 물체는 하나의 영역으로 검출될 수 있습니다."
)

if count > 150:
    st.warning(
        "검출된 영역 중 면적이 큰 150개에만 경계 상자와 번호를 표시했습니다. "
        "총 개수에는 모든 영역이 포함됩니다."
    )

st.download_button(
    "검출 결과 이미지 다운로드 (PNG)",
    download_png(result),
    file_name="HW06_detection.png",
    mime="image/png"
)


with st.expander("영상처리 과정 확인"):
    x, y = st.columns(2)

    with x:
        st.markdown("**1. Grayscale**")
        st.image(preview(gray), width="stretch")

        st.markdown(f"**3. Binary Mask (Threshold {threshold})**")
        st.image(preview(mask.astype(np.uint8) * 255), width="stretch")

    with y:
        st.markdown("**2. Median Filtering (7×7)**")
        st.image(preview(filtered), width="stretch")

        st.markdown("**4. Connected Component Labeling**")
        st.image(preview(color_labels(labels, count)), width="stretch")

    fig, ax = plt.subplots(figsize=(7, 2.8))
    ax.hist(filtered.ravel(), bins=256, range=(0, 256), color="gray")
    ax.axvline(
        threshold,
        color="red",
        linestyle="--",
        label=f"Threshold = {threshold}"
    )
    ax.set(xlabel="Pixel Intensity", ylabel="Frequency")
    ax.legend()
    st.pyplot(fig, width="stretch")
    plt.close(fig)


with st.expander("객체별 픽셀 수 및 최대·최소 영역"):
    if count:
        max_label = int(np.argmax(areas)) + 1
        min_label = int(np.argmin(areas)) + 1

        x, y = st.columns(2)

        with x:
            st.markdown(f"**최대 영역 · Label {max_label}**")
            st.image(
                preview((labels == max_label).astype(np.uint8) * 255),
                width="stretch"
            )
            st.caption(f"{areas[max_label - 1]:,} pixels")

        with y:
            st.markdown(f"**최소 영역 · Label {min_label}**")
            st.image(
                preview((labels == min_label).astype(np.uint8) * 255),
                width="stretch"
            )
            st.caption(f"{areas[min_label - 1]:,} pixels")

        st.dataframe(
            [
                {"Label": i, "픽셀 수": int(area)}
                for i, area in enumerate(areas[:500], 1)
            ],
            hide_index=True,
            width="stretch"
        )

        if count > 500:
            st.caption(
                "표에는 500개 영역만 표시됩니다. "
                "CSV 파일에는 전체 결과가 저장됩니다."
            )
    else:
        st.info("검출된 영역이 없습니다. 임계값을 조정해 주세요.")

    st.download_button(
        "객체별 픽셀 수 다운로드 (CSV)",
        download_csv(areas),
        file_name="HW06_areas.csv",
        mime="text/csv"
    )
