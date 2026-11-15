
import csv
import io

import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont, ImageOps
from scipy import ndimage as ndi, signal
from skimage.feature import peak_local_max
from skimage.segmentation import watershed


st.set_page_config(
    page_title="HW06 | Object Counting",
    page_icon="🔎",
    layout="wide"
)

METHODS = [
    "자동 분석",
    "HW05 기본",
    "밝은 객체",
    "어두운 객체",
    "노란색·갈색 객체",
    "파란색 객체",
    "여러 색 객체",
    "원형 객체",
    "밝은 문자",
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
}


@st.cache_data(show_spinner=False, max_entries=3)
def load_image(data):
    with Image.open(io.BytesIO(data)) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        original_size = image.size

        if image.width * image.height > 90_000_000:
            raise ValueError(
                "The image exceeds the maximum pixel limit."
            )

        image.thumbnail((2300, 2300), Image.Resampling.LANCZOS)
        rgb = np.asarray(image).copy()

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    filtered = signal.medfilt2d(gray, 7)

    return rgb, gray, filtered, original_size


def auto_method(rgb, gray):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    channel = rgb.astype(np.int16)

    warm = (
        np.minimum(channel[:, :, 0], channel[:, :, 1])
        - channel[:, :, 2]
    )
    blue = (
        channel[:, :, 2]
        - np.maximum(channel[:, :, 0], channel[:, :, 1])
    )

    warm_ratio = np.mean(warm > 28)
    blue_ratio = np.mean(blue > 28)
    colorful_ratio = np.mean(hsv[:, :, 1] > 65)
    white_ratio = np.mean(gray > 220)
    dark_ratio = np.mean(gray < 80)

    n, _, stats, _ = cv2.connectedComponentsWithStats(
        (gray < 100).astype(np.uint8), 8
    )

    dark_blob = (
        stats[1:, cv2.CC_STAT_AREA].max() / gray.size
        if n > 1 else 0
    )

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

    if (
        white_ratio > 0.26
        and np.median(gray) > 160
        and colorful_ratio < 0.025
    ):
        return "원형 객체"

    return (
        "밝은 객체"
        if np.median(gray) < 105
        else "어두운 객체"
    )


def otsu(gray):
    return cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )[0]


def create_mask(rgb, gray, filtered, method, threshold, sensitivity):
    if method == "HW05 기본":
        return (filtered > threshold).astype(np.uint8)

    if method == "밝은 객체":
        value = np.clip(
            otsu(filtered) - sensitivity, 0, 255
        )
        return (filtered > value).astype(np.uint8)

    if method == "어두운 객체":
        value = np.clip(
            otsu(filtered) + sensitivity, 0, 255
        )
        return (filtered < value).astype(np.uint8)

    smooth = cv2.medianBlur(rgb, 5).astype(np.int16)
    red, green, blue = (
        smooth[:, :, 0],
        smooth[:, :, 1],
        smooth[:, :, 2]
    )

    if method in ("노란색·갈색 객체", "파란색 객체"):
        if method == "노란색·갈색 객체":
            score = np.clip(
                np.minimum(red, green) - blue,
                0,
                255
            ).astype(np.uint8)

            cutoff = max(
                10, max(21, otsu(score)) - sensitivity
            )
        else:
            score = np.clip(
                blue - np.maximum(red, green),
                0,
                255
            ).astype(np.uint8)

            cutoff = max(
                12, max(22, otsu(score)) - sensitivity
            )

        return (score > cutoff).astype(np.uint8)

    if method == "여러 색 객체":
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

        cutoff = max(
            30,
            max(56, otsu(hsv[:, :, 1])) - sensitivity
        )

        return (
            (hsv[:, :, 1] > cutoff)
            & (hsv[:, :, 2] > 35)
        ).astype(np.uint8)

    if method == "밝은 문자":
        dark = (filtered < 105).astype(np.uint8)

        dark = cv2.morphologyEx(
            dark,
            cv2.MORPH_CLOSE,
            np.ones((15, 15), np.uint8)
        )

        n, regions, stats, _ = cv2.connectedComponentsWithStats(
            dark, 8
        )

        if n <= 1:
            return (filtered > 185).astype(np.uint8)

        largest = 1 + np.argmax(
            stats[1:, cv2.CC_STAT_AREA]
        )

        region = ndi.binary_fill_holes(
            regions == largest
        ).astype(np.uint8)

        region = cv2.erode(
            region, np.ones((9, 9), np.uint8)
        )

        coverage = (region > 0).mean(axis=1)
        rows = np.flatnonzero(
            coverage
            > np.percentile(coverage, 90) * 0.72
        )

        if rows.size:
            region[:rows[0], :] = 0

        return (
            (filtered > 170)
            & (region > 0)
        ).astype(np.uint8)

    raise ValueError("Unsupported segmentation method.")


def split_touching(labels, areas, min_area):
    boxes = ndi.find_objects(labels)
    output = np.zeros_like(labels, dtype=np.int32)

    typical_area = float(np.median(areas))
    distance = max(
        5, round(np.sqrt(typical_area) * 0.39)
    )
    next_id = 0

    for index, box in enumerate(boxes, start=1):
        if box is None:
            continue

        region = labels[box] == index
        area = int(areas[index - 1])
        segments = None

        if (
            area >= 1.8 * typical_area
            and area >= 4 * min_area
        ):
            depth = cv2.distanceTransform(
                region.astype(np.uint8),
                cv2.DIST_L2,
                5
            )

            peaks = peak_local_max(
                depth,
                min_distance=distance,
                threshold_abs=max(
                    1.5,
                    float(depth.max()) * 0.40
                ),
                exclude_border=False
            )

            max_parts = min(
                35,
                int(area / (typical_area * 0.42))
            )

            if 2 <= len(peaks) <= max_parts:
                seeds = np.zeros(
                    region.shape,
                    dtype=np.int32
                )

                for number, (row, col) in enumerate(
                    peaks, start=1
                ):
                    seeds[row, col] = number

                segments = watershed(
                    -depth,
                    seeds,
                    mask=region
                )

        if segments is None:
            next_id += 1
            output[box][region] = next_id

        else:
            for number in range(
                1, int(segments.max()) + 1
            ):
                part = segments == number

                if int(part.sum()) >= min_area:
                    next_id += 1
                    output[box][part] = next_id

    return output


def count_regions(mask, min_percent, split):
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (3, 3)
    )

    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, kernel
    )
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, kernel
    )

    min_area = max(
        24,
        round(mask.size * min_percent / 100)
    )

    n, first, stats, _ = cv2.connectedComponentsWithStats(
        mask, 8
    )

    valid = np.flatnonzero(
        (stats[:, cv2.CC_STAT_AREA] >= min_area)
        & (stats[:, cv2.CC_STAT_AREA] < mask.size * 0.25)
    )
    valid = valid[valid != 0]

    mask = np.isin(
        first, valid
    ).astype(np.uint8)

    labels, _ = ndi.label(
        mask,
        structure=np.ones((3, 3), dtype=np.uint8)
    )

    areas = np.bincount(labels.ravel())[1:]

    if split and len(areas) >= 3:
        labels = split_touching(
            labels, areas, min_area
        )

        present = np.unique(labels)
        present = present[present > 0]

        mapping = np.zeros(
            int(labels.max()) + 1,
            dtype=np.int32
        )
        mapping[present] = np.arange(
            1, len(present) + 1
        )

        labels = mapping[labels]
        areas = np.bincount(labels.ravel())[1:]

    return (
        (labels > 0).astype(np.uint8),
        labels,
        areas
    )


def find_circles(gray, sensitivity):
    smooth = cv2.GaussianBlur(
        gray, (7, 7), 1.3
    )
    factor = min(gray.shape) / 686

    detected = cv2.HoughCircles(
        smooth,
        cv2.HOUGH_GRADIENT,
        dp=1.15,
        minDist=max(24, round(29 * factor)),
        param1=95,
        param2=max(
            18, 35 - int(sensitivity * 0.35)
        ),
        minRadius=max(8, round(19 * factor)),
        maxRadius=max(15, round(58 * factor))
    )

    if detected is None:
        return np.empty(
            (0, 3), dtype=np.int32
        )

    return np.rint(
        detected[0]
    ).astype(np.int32)


@st.cache_data(show_spinner=False, max_entries=8)
def analyze(
    rgb, gray, filtered, method,
    threshold, sensitivity, min_percent, split
):
    chosen = (
        auto_method(rgb, gray)
        if method == "자동 분석"
        else method
    )

    if chosen == "원형 객체":
        circles = find_circles(
            gray, sensitivity
        )

        return (
            chosen,
            None,
            None,
            np.pi * circles[:, 2] ** 2,
            circles
        )

    mask = create_mask(
        rgb,
        gray,
        filtered,
        chosen,
        threshold,
        sensitivity
    )

    if chosen == "HW05 기본":
        labels, _ = ndi.label(mask)
        areas = np.bincount(
            labels.ravel()
        )[1:]

    else:
        mask, labels, areas = count_regions(
            mask,
            min_percent,
            split
        )

    return chosen, mask, labels, areas, None


def annotate(rgb, labels, areas, circles):
    image = Image.fromarray(rgb.copy())
    draw = ImageDraw.Draw(image)

    height, width = rgb.shape[:2]
    line_width = max(
        2, round(min(height, width) / 500)
    )
    size = max(
        13, round(min(height, width) / 52)
    )

    try:
        font = ImageFont.truetype(
            "DejaVuSans.ttf", size
        )
    except OSError:
        font = ImageFont.load_default()

    if circles is not None:
        for index, (x, y, radius) in enumerate(
            circles, start=1
        ):
            draw.ellipse(
                (
                    x - radius,
                    y - radius,
                    x + radius,
                    y + radius
                ),
                outline="#19C37D",
                width=line_width
            )

            if index <= 100:
                draw.text(
                    (
                        x - radius,
                        max(0, y - radius - size)
                    ),
                    str(index),
                    font=font,
                    fill="white",
                    stroke_width=2,
                    stroke_fill="black"
                )

    elif len(areas):
        boxes = ndi.find_objects(labels)

        for index in np.argsort(areas)[-150:]:
            box = boxes[int(index)]

            if box is None:
                continue

            rows, cols = box

            x0, y0 = cols.start, rows.start
            x1, y1 = cols.stop - 1, rows.stop - 1

            draw.rectangle(
                (x0, y0, x1, y1),
                outline="#19C37D",
                width=line_width
            )

            if len(areas) <= 150:
                draw.text(
                    (
                        x0,
                        max(0, y0 - size - 4)
                    ),
                    str(index + 1),
                    font=font,
                    fill="white",
                    stroke_width=2,
                    stroke_fill="black"
                )

    return image


def preview(image):
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    image = image.copy()
    image.thumbnail(
        (630, 680),
        Image.Resampling.LANCZOS
    )

    return image


def png_data(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def csv_data(areas, circles):
    buffer = io.StringIO()
    writer = csv.writer(buffer)

    if circles is None:
        writer.writerow([
            "Label", "Area (pixels)"
        ])

        for index, area in enumerate(
            areas, start=1
        ):
            writer.writerow([
                index, int(area)
            ])

    else:
        writer.writerow([
            "Object ID",
            "Center X",
            "Center Y",
            "Radius (px)",
            "Estimated Area (px²)"
        ])

        for index, (x, y, radius) in enumerate(
            circles, start=1
        ):
            writer.writerow([
                index,
                x,
                y,
                radius,
                round(np.pi * radius ** 2)
            ])

    return buffer.getvalue().encode("utf-8-sig")


st.title("HW06 · Object Counting")

st.caption(
    "의료영상처리의 이론 및 실습 | HW05 기반 Streamlit 객체 검출"
)

st.write(
    "Upload an image to detect and count objects."
)

file = st.file_uploader(
    "Upload an Image",
    type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"]
)

if file is None:
    st.info("Choose an image to get started.")
    st.stop()

try:
    data = file.getvalue()

    if len(data) > 25 * 1024 * 1024:
        raise ValueError(
            "Please select an image smaller than 25 MB."
        )

    rgb, gray, filtered, original_size = load_image(
        data
    )

except (
    ValueError,
    OSError,
    Image.DecompressionBombError,
    cv2.error
) as error:
    st.error(
        f"Unable to open the image: {error}"
    )
    st.stop()


with st.sidebar:
    st.header("Detection Settings")

    method = st.selectbox(
        "Segmentation Method",
        METHODS,
        format_func=lambda option: METHOD_LABELS[option]
    )

    selected = (
        auto_method(rgb, gray)
        if method == "자동 분석"
        else method
    )

    st.caption(
        f"Selected method: {METHOD_LABELS[selected]}"
    )

    threshold = 128
    sensitivity = 0
    min_percent = 0.008
    split = False

    if selected == "HW05 기본":
        threshold = st.slider(
            "Threshold", 0, 255, 128
        )

        st.caption(
            "7×7 median filter · 4-connectivity · No area filtering"
        )

    elif selected == "원형 객체":
        sensitivity = st.slider(
            "Detection Sensitivity",
            -30, 30, 0, 5
        )

        st.caption(
            "Circular detection using the Hough transform"
        )

    else:
        sensitivity = st.slider(
            "Detection Sensitivity",
            -30, 30, 0, 5
        )

        min_percent = st.number_input(
            "Minimum Object Area (% of image)",
            0.001,
            1.0,
            0.008,
            0.005,
            format="%.3f"
        )

        split = st.checkbox(
            "Separate Touching Objects (Watershed)",
            value=(selected == "여러 색 객체"),
            key=f"split_{selected}"
        )

    st.caption(
        "Automatic detection is an estimate and may not match the actual object count."
    )


try:
    chosen, mask, labels, areas, circles = analyze(
        rgb,
        gray,
        filtered,
        method,
        threshold,
        sensitivity,
        min_percent,
        split
    )

except (
    ValueError,
    cv2.error,
    MemoryError
) as error:
    st.error(
        f"Image processing failed: {error}"
    )
    st.stop()


result = annotate(
    rgb, labels, areas, circles
)
count = len(areas)

st.subheader("Detection Results")

a, b, c = st.columns(3)

a.metric(
    "Detected Objects",
    f"{count:,}"
)

area_label = (
    "Estimated Area"
    if circles is not None
    else "Area"
)

b.metric(
    f"Largest {area_label}",
    f"{int(areas.max()):,} px²" if count else "—"
)

c.metric(
    f"Smallest {area_label}",
    f"{int(areas.min()):,} px²" if count else "—"
)


left, right = st.columns(
    2, gap="large"
)

with left:
    st.markdown("#### Original Image")
    st.image(
        preview(rgb),
        width="stretch"
    )

with right:
    st.markdown("#### Detection Result")
    st.image(
        preview(result),
        width="stretch"
    )


st.caption(
    f"Method: {METHOD_LABELS[chosen]} · "
    f"Analysis Resolution: {rgb.shape[1]} × {rgb.shape[0]} px"
    + (
        " · Image resized for processing"
        if original_size != (
            rgb.shape[1], rgb.shape[0]
        )
        else ""
    )
)

st.caption(
    "Detected regions may not match the actual object count. "
    "Overlapping objects and complex backgrounds can affect the result."
)

if circles is None and count > 150:
    st.warning(
        "Only the 150 largest regions are highlighted. "
        "The total count includes all detected regions."
    )


st.download_button(
    "Download Detection Image (PNG)",
    data=png_data(result),
    file_name="HW06_detection.png",
    mime="image/png"
)


with st.expander(
    "Processing Details & Object Statistics"
):
    x, y = st.columns(2)

    with x:
        st.markdown("**Grayscale**")
        st.image(
            preview(gray),
            width="stretch"
        )

    with y:
        st.markdown("**Median Filtering (7×7)**")
        st.image(
            preview(filtered),
            width="stretch"
        )

    if circles is None:
        x, y = st.columns(2)

        with x:
            st.markdown("**Binary Mask**")
            st.image(
                preview(mask * 255),
                width="stretch"
            )

        with y:
            st.markdown(
                "**Connected Component Labeling**"
            )

            palette = np.random.default_rng(12).integers(
                65,
                245,
                size=(count + 1, 3),
                dtype=np.uint8
            )
            palette[0] = [25, 29, 39]

            st.image(
                preview(palette[labels]),
                width="stretch"
            )

        if count:
            largest = int(
                np.argmax(areas)
            ) + 1

            smallest = int(
                np.argmin(areas)
            ) + 1

            x, y = st.columns(2)

            with x:
                st.markdown(
                    f"**Largest Region · Label {largest}**"
                )

                st.image(
                    preview(
                        (labels == largest).astype(np.uint8) * 255
                    ),
                    width="stretch"
                )

            with y:
                st.markdown(
                    f"**Smallest Region · Label {smallest}**"
                )

                st.image(
                    preview(
                        (labels == smallest).astype(np.uint8) * 255
                    ),
                    width="stretch"
                )

    else:
        st.info(
            "Circular object detection uses the Hough transform. "
            "Areas are estimated from the detected radii."
        )


    st.markdown("**Object Statistics**")

    st.dataframe(
        [
            {
                "Object ID": i,
                "Area (px²)": round(float(area))
            }
            for i, area in enumerate(areas[:300], 1)
        ],
        hide_index=True,
        width="stretch"
    )

    if count > 300:
        st.caption(
            "The table shows the first 300 objects. "
            "The CSV includes all detected objects."
        )

    st.download_button(
        "Download Object Statistics (CSV)",
        data=csv_data(areas, circles),
        file_name="HW06_areas.csv",
        mime="text/csv"
    )
