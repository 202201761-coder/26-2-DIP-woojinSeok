
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

METHODS = {
    "Auto": "auto",
    "HW05 (Original)": "hw05",
    "Bright Objects": "bright",
    "Dark Objects": "dark",
    "Yellow / Brown Objects": "warm",
    "Yellow / Brown (Enhanced)": "warm_enhanced",
    "Blue Objects": "blue",
    "Multicolor Objects": "multicolor",
    "Circular Objects": "circles",
    "Bright Text": "text",
}


@st.cache_data(show_spinner=False, max_entries=3)
def load_image(data):
    with Image.open(io.BytesIO(data)) as img:
        img = ImageOps.exif_transpose(img).convert("RGB")
        original_size = img.size

        if img.width * img.height > 90_000_000:
            raise ValueError(
                "Image resolution exceeds the supported limit."
            )

        img.thumbnail((2300, 2300), Image.Resampling.LANCZOS)
        rgb = np.asarray(img).copy()

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    filtered = signal.medfilt2d(gray, 7)

    return rgb, gray, filtered, original_size


def automatic_method(rgb, gray):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    values = rgb.astype(np.int16)

    warm = (
        np.minimum(values[:, :, 0], values[:, :, 1])
        - values[:, :, 2]
    )
    blue = (
        values[:, :, 2]
        - np.maximum(values[:, :, 0], values[:, :, 1])
    )

    warm_ratio = np.mean(warm > 28)
    blue_ratio = np.mean(blue > 28)
    colorful_ratio = np.mean(hsv[:, :, 1] > 65)
    white_ratio = np.mean(gray > 220)
    dark_ratio = np.mean(gray < 80)

    count, _, stats, _ = cv2.connectedComponentsWithStats(
        (gray < 100).astype(np.uint8), 8
    )

    dark_blob = (
        stats[1:, cv2.CC_STAT_AREA].max() / gray.size
        if count > 1 else 0
    )

    if dark_ratio > 0.88 and white_ratio < 0.05:
        return "hw05"

    if dark_blob > 0.26 and white_ratio > 0.10 and dark_ratio > 0.30:
        return "text"

    if blue_ratio > 0.014 and colorful_ratio < 0.17:
        return "blue"

    if blue_ratio > 0.008 and colorful_ratio > 0.13:
        return "multicolor"

    if warm_ratio > 0.007 and blue_ratio < 0.008:
        return "warm"

    if colorful_ratio > 0.09:
        return "multicolor"

    if (
        white_ratio > 0.26
        and np.median(gray) > 160
        and colorful_ratio < 0.025
    ):
        return "circles"

    return "bright" if np.median(gray) < 105 else "dark"


def otsu(gray):
    threshold, _ = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return threshold


def make_mask(rgb, gray, filtered, method, threshold, sensitivity):
    if method == "hw05":
        return (filtered > threshold).astype(np.uint8)

    if method == "bright":
        value = np.clip(
            otsu(filtered) - sensitivity, 0, 255
        )
        return (filtered > value).astype(np.uint8)

    if method == "dark":
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

    if method in ("warm", "warm_enhanced"):
        score = np.clip(
            np.minimum(red, green) - blue,
            0, 255
        ).astype(np.uint8)

        if method == "warm":
            cutoff = max(
                10,
                max(21, otsu(score)) - sensitivity
            )
            return (score > cutoff).astype(np.uint8)

        reference = cv2.GaussianBlur(score, (0, 0), 15)
        contrast = (
            score.astype(np.int16)
            - reference.astype(np.int16)
        )

        baseline = max(21, otsu(score)) - sensitivity

        extra = (
            (score > max(15, otsu(score) - 14 - sensitivity))
            & (contrast > 16)
        )

        return ((score > baseline) | extra).astype(np.uint8)

    if method == "blue":
        score = np.clip(
            blue - np.maximum(red, green),
            0, 255
        ).astype(np.uint8)

        cutoff = max(
            12,
            max(22, otsu(score)) - sensitivity
        )

        return (score > cutoff).astype(np.uint8)

    if method == "multicolor":
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

        cutoff = max(
            30,
            max(56, otsu(hsv[:, :, 1])) - sensitivity
        )

        return (
            (hsv[:, :, 1] > cutoff)
            & (hsv[:, :, 2] > 35)
        ).astype(np.uint8)

    if method == "text":
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
            region,
            np.ones((9, 9), np.uint8)
        )

        coverage = (region > 0).mean(axis=1)

        rows = np.flatnonzero(
            coverage > np.percentile(coverage, 90) * 0.72
        )

        if rows.size:
            region[:rows[0], :] = 0

        return (
            (filtered > 170)
            & (region > 0)
        ).astype(np.uint8)

    raise ValueError("Unsupported segmentation method.")


def split_regions(labels, areas, min_area):
    result = np.zeros_like(labels, dtype=np.int32)

    typical = float(np.median(areas))
    distance = max(
        5, round(np.sqrt(typical) * 0.39)
    )
    next_id = 0

    for index, box in enumerate(ndi.find_objects(labels), 1):
        if box is None:
            continue

        region = labels[box] == index
        area = int(areas[index - 1])
        parts = None

        if area >= 1.8 * typical and area >= 4 * min_area:
            depth = cv2.distanceTransform(
                region.astype(np.uint8),
                cv2.DIST_L2,
                5
            )

            peaks = peak_local_max(
                depth,
                min_distance=distance,
                threshold_abs=max(
                    1.5, float(depth.max()) * 0.40
                ),
                exclude_border=False
            )

            max_parts = min(
                35,
                int(area / (typical * 0.42))
            )

            if 2 <= len(peaks) <= max_parts:
                seeds = np.zeros(
                    region.shape,
                    dtype=np.int32
                )

                for number, (row, col) in enumerate(peaks, 1):
                    seeds[row, col] = number

                parts = watershed(
                    -depth,
                    seeds,
                    mask=region
                )

        if parts is None:
            next_id += 1
            result[box][region] = next_id

        else:
            for number in range(1, int(parts.max()) + 1):
                part = parts == number

                if int(part.sum()) >= min_area:
                    next_id += 1
                    result[box][part] = next_id

    return result


def count_regions(mask, min_percent, separate):
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

    mask = np.isin(first, valid).astype(np.uint8)

    labels, _ = ndi.label(
        mask,
        structure=np.ones((3, 3), dtype=np.uint8)
    )

    areas = np.bincount(labels.ravel())[1:]

    if separate and len(areas) >= 3:
        labels = split_regions(labels, areas, min_area)

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

    return (labels > 0).astype(np.uint8), labels, areas


def detect_circles(gray, sensitivity):
    smooth = cv2.GaussianBlur(
        gray, (7, 7), 1.3
    )
    factor = min(gray.shape) / 686

    circles = cv2.HoughCircles(
        smooth,
        cv2.HOUGH_GRADIENT,
        dp=1.15,
        minDist=max(24, round(29 * factor)),
        param1=95,
        param2=max(18, 35 - int(sensitivity * 0.35)),
        minRadius=max(8, round(19 * factor)),
        maxRadius=max(15, round(58 * factor))
    )

    if circles is None:
        return np.empty((0, 3), dtype=np.int32)

    return np.rint(circles[0]).astype(np.int32)


def automatic_area(
    rgb, gray, filtered,
    mode, selected, minimum, separate, adaptive
):
    if (
        not adaptive
        or mode != "auto"
        or selected != "warm"
        or separate
        or minimum != 0.008
    ):
        return minimum

    mask = make_mask(
        rgb, gray, filtered,
        selected, 128, 0
    )

    _, _, areas = count_regions(
        mask, minimum, False
    )

    if (
        len(areas) >= 15
        and np.median(areas) < 210
        and np.percentile(areas, 25) < 130
    ):
        return 0.004

    return minimum


@st.cache_data(show_spinner=False, max_entries=8)
def analyze(
    rgb, gray, filtered,
    mode, threshold, sensitivity,
    minimum, separate, adaptive
):
    selected = (
        automatic_method(rgb, gray)
        if mode == "auto"
        else mode
    )

    if selected == "circles":
        circles = detect_circles(
            gray, sensitivity
        )

        areas = np.pi * circles[:, 2] ** 2

        return selected, None, None, areas, circles

    mask = make_mask(
        rgb, gray, filtered,
        selected, threshold, sensitivity
    )

    if selected == "hw05":
        labels, _ = ndi.label(mask)
        areas = np.bincount(labels.ravel())[1:]

    else:
        actual_min = automatic_area(
            rgb, gray, filtered,
            mode, selected, minimum,
            separate, adaptive
        )

        mask, labels, areas = count_regions(
            mask, actual_min, separate
        )

    return selected, mask, labels, areas, None


def annotate(rgb, labels, areas, circles):
    image = Image.fromarray(rgb.copy())
    draw = ImageDraw.Draw(image)

    height, width = rgb.shape[:2]
    weight = max(
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
        for i, (x, y, radius) in enumerate(circles, 1):
            draw.ellipse(
                (
                    x - radius,
                    y - radius,
                    x + radius,
                    y + radius
                ),
                outline="#19C37D",
                width=weight
            )

            if i <= 100:
                draw.text(
                    (x - radius, max(0, y - radius - size)),
                    str(i),
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
                width=weight
            )

            if len(areas) <= 150:
                draw.text(
                    (x0, max(0, y0 - size - 4)),
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


def png_bytes(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def csv_bytes(areas, circles):
    buffer = io.StringIO()
    writer = csv.writer(buffer)

    if circles is None:
        writer.writerow([
            "Object ID", "Area (pixels)"
        ])

        for index, area in enumerate(areas, 1):
            writer.writerow([index, int(area)])

    else:
        writer.writerow([
            "Object ID",
            "Center X",
            "Center Y",
            "Radius (px)",
            "Estimated Area (px²)"
        ])

        for index, (x, y, radius) in enumerate(circles, 1):
            writer.writerow([
                index,
                x,
                y,
                radius,
                round(np.pi * radius ** 2)
            ])

    return buffer.getvalue().encode("utf-8-sig")


# Streamlit Interface

st.title("HW06 · Object Counting")

st.caption(
    "의료영상처리의 이론 및 실습 | HW05 기반 Streamlit 객체 검출"
)

st.write(
    "Upload an image to detect and count objects."
)

uploaded = st.file_uploader(
    "Upload an Image",
    type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"]
)

if uploaded is None:
    st.info("Choose an image to get started.")
    st.stop()

try:
    data = uploaded.getvalue()

    if len(data) > 25 * 1024 * 1024:
        raise ValueError(
            "Please choose a file smaller than 25 MB."
        )

    rgb, gray, filtered, original_size = load_image(data)

except (
    OSError,
    ValueError,
    Image.DecompressionBombError,
    cv2.error
) as exc:
    st.error(f"Unable to open the image: {exc}")
    st.stop()


with st.sidebar:
    st.header("Detection Settings")

    name = st.selectbox(
        "Segmentation Method",
        list(METHODS)
    )

    mode = METHODS[name]

    selected = (
        automatic_method(rgb, gray)
        if mode == "auto"
        else mode
    )

    labels_to_names = {
        value: key
        for key, value in METHODS.items()
    }

    st.caption(
        f"Selected method: {labels_to_names[selected]}"
    )

    threshold = 128
    sensitivity = 0
    minimum = 0.008
    separate = False
    adaptive = True

    if selected == "hw05":
        threshold = st.slider(
            "Threshold", 0, 255, 128
        )

        st.caption(
            "7×7 median filter · 4-connectivity · No area filtering"
        )

    elif selected == "circles":
        sensitivity = st.slider(
            "Detection Sensitivity",
            -30, 30, 0, 5
        )

        st.caption("Hough Circle Transform")

    else:
        sensitivity = st.slider(
            "Detection Sensitivity",
            -30, 30, 0, 5
        )

        minimum = st.number_input(
            "Minimum Object Area (% of image)",
            min_value=0.001,
            max_value=1.0,
            value=0.008,
            step=0.005,
            format="%.3f"
        )

        separate = st.checkbox(
            "Separate Touching Objects (Watershed)",
            value=(selected == "multicolor"),
            key=f"split_{selected}"
        )

        if mode == "auto" and selected == "warm":
            adaptive = st.checkbox(
                "Adaptive Small-Object Filtering",
                value=True
            )

    st.caption(
        "Counts are estimates and may differ from the actual number of objects."
    )


try:
    chosen, mask, labels, areas, circles = analyze(
        rgb,
        gray,
        filtered,
        mode,
        threshold,
        sensitivity,
        minimum,
        separate,
        adaptive
    )

    result = annotate(
        rgb, labels, areas, circles
    )

except (
    OSError,
    ValueError,
    cv2.error,
    MemoryError
) as exc:
    st.error(
        f"Image processing failed: {exc}"
    )
    st.stop()


count = len(areas)

st.subheader("Detection Results")

c1, c2, c3 = st.columns(3)

c1.metric(
    "Detected Objects",
    f"{count:,}"
)

area_kind = (
    "Estimated Area"
    if circles is not None
    else "Area"
)

c2.metric(
    f"Largest {area_kind}",
    f"{int(areas.max()):,} px²" if count else "—"
)

c3.metric(
    f"Smallest {area_kind}",
    f"{int(areas.min()):,} px²" if count else "—"
)


left, right = st.columns(2, gap="large")

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
    f"Method: {labels_to_names[chosen]} · "
    f"Analysis Resolution: {rgb.shape[1]} × {rgb.shape[0]} px"
    + (
        " · Resized for processing"
        if original_size != (rgb.shape[1], rgb.shape[0])
        else ""
    )
)

if circles is None and count > 150:
    st.warning(
        "Only the 150 largest objects are highlighted; "
        "the count includes all regions."
    )


st.download_button(
    "Download Detection Image (PNG)",
    data=png_bytes(result),
    file_name="HW06_detection.png",
    mime="image/png"
)


with st.expander("Processing Details & Object Statistics"):
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Grayscale**")
        st.image(
            preview(gray),
            width="stretch"
        )

    with col2:
        st.markdown("**Median Filtering (7×7)**")
        st.image(
            preview(filtered),
            width="stretch"
        )

    if circles is None:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Binary Mask**")
            st.image(
                preview(mask * 255),
                width="stretch"
            )

        with col2:
            st.markdown("**Connected Component Labeling**")

            palette = np.random.default_rng(12).integers(
                65, 245,
                size=(count + 1, 3),
                dtype=np.uint8
            )
            palette[0] = [25, 29, 39]

            st.image(
                preview(palette[labels]),
                width="stretch"
            )

        
        if count:
            ranked = np.argsort(areas)[::-1]

            biggest = int(ranked[0]) + 1
            smallest = int(ranked[-1]) + 1

            regions = [("Largest Region", biggest)]

            if count >= 3:
                second = int(ranked[1]) + 1
                regions.append(("Second Largest Region", second))

            if count >= 2:
                regions.append(("Smallest Region", smallest))

            boxes = ndi.find_objects(labels)

            side = max(
                max(
                    boxes[idx - 1][0].stop - boxes[idx - 1][0].start,
                    boxes[idx - 1][1].stop - boxes[idx - 1][1].start
                )
                for _, idx in regions
            ) + 24

            side = max(48, side)

            def component_crop(region_id):
                box = boxes[region_id - 1]

                cy = (box[0].start + box[0].stop) // 2
                cx = (box[1].start + box[1].stop) // 2

                y0 = cy - side // 2
                x0 = cx - side // 2

                h, w = labels.shape

                sy0, sy1 = max(0, y0), min(h, y0 + side)
                sx0, sx1 = max(0, x0), min(w, x0 + side)

                crop = np.zeros((side, side), dtype=np.uint8)

                crop[
                    sy0 - y0:sy1 - y0,
                    sx0 - x0:sx1 - x0
                ] = (
                    labels[sy0:sy1, sx0:sx1] == region_id
                ).astype(np.uint8) * 255

                return crop

            st.markdown("**Region Size Comparison**")

            columns = st.columns(len(regions))

            for column, (title, region_id) in zip(columns, regions):
                with column:
                    st.markdown(f"**{title}**")
                    st.image(
                        component_crop(region_id),
                        width="stretch"
                    )
                    st.caption(
                        f"Label {region_id} · "
                        f"{int(areas[region_id - 1]):,} px²"
                    )
                    
    else:
        st.info(
            "Circle areas are estimates calculated from their radii."
        )

    st.markdown("**Object Statistics**")

    st.dataframe(
        [
            {
                "Object ID": i,
                "Area (px²)": round(float(a))
            }
            for i, a in enumerate(areas[:300], 1)
        ],
        hide_index=True,
        width="stretch"
    )

    if count > 300:
        st.caption(
            "The table shows the first 300 objects; "
            "the CSV includes all objects."
        )

    st.download_button(
        "Download Object Statistics (CSV)",
        data=csv_bytes(areas, circles),
        file_name="HW06_areas.csv",
        mime="text/csv"
    )


if chosen == "warm":
    with st.expander("Compare Yellow / Brown Segmentation"):
        st.caption(
            "Enhanced masks may increase both true detections "
            "and false positives."
        )

        regular = make_mask(
            rgb, gray, filtered,
            "warm", threshold, sensitivity
        )

        enhanced = make_mask(
            rgb, gray, filtered,
            "warm_enhanced", threshold, sensitivity
        )

        _, reg_labels, reg_areas = count_regions(
            regular, minimum, separate
        )

        _, enh_labels, enh_areas = count_regions(
            enhanced, minimum, separate
        )

        a, b = st.columns(2)

        with a:
            st.markdown(
                f"**Standard · {len(reg_areas)} objects**"
            )
            st.image(
                preview(
                    annotate(rgb, reg_labels, reg_areas, None)
                ),
                width="stretch"
            )

        with b:
            st.markdown(
                f"**Enhanced · {len(enh_areas)} objects**"
            )
            st.image(
                preview(
                    annotate(rgb, enh_labels, enh_areas, None)
                ),
                width="stretch"
            )
