"""
extractor.py
Extract geometric, shape, Hu‑moment, texture and colour features from each defect contour.
"""

import cv2
import numpy as np
import pandas as pd


# ---------- Helper: geometric features ----------
def _geometric_features(cnt):
    """Area, perimeter, bounding‑box dimensions, aspect ratio, extent."""
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, closed=True)
    _, _, w, h = cv2.boundingRect(cnt)          # x, y not needed
    rect_area = w * h
    return {
        'area': area,
        'perimeter': perimeter,
        'bounding_width': w,
        'bounding_height': h,
        'aspect_ratio': w / h if h != 0 else 0,
        'rect_area': rect_area,
        'extent': area / rect_area if rect_area != 0 else 0,
    }


# ---------- Helper: shape descriptors ----------
def _shape_features(cnt):
    """Circularity, equivalent diameter, convexity, ellipse major/minor axis ratio."""
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, closed=True)
    circularity = (4 * np.pi * area) / (perimeter * perimeter) if perimeter != 0 else 0
    eq_diameter = np.sqrt(4 * area / np.pi)

    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    convexity = area / hull_area if hull_area != 0 else 0

    # Fit ellipse only if we have enough points
    ellipse = {'ellipse_major_axis': 0, 'ellipse_minor_axis': 0, 'ellipse_aspect_ratio': 0}
    if len(cnt) >= 5:
        (_, _), (minor, major), _ = cv2.fitEllipse(cnt)   # centre and angle ignored
        ellipse['ellipse_major_axis'] = major
        ellipse['ellipse_minor_axis'] = minor
        ellipse['ellipse_aspect_ratio'] = major / minor if minor != 0 else 0

    return {
        'circularity': circularity,
        'eq_diameter': eq_diameter,
        'convexity': convexity,
        **ellipse
    }


# ---------- Helper: Hu moments (7 invariants) ----------
def _hu_moments(cnt):
    """Log‑transformed Hu moments; invariant to translation, rotation, and scale."""
    moments = cv2.moments(cnt)
    hu = cv2.HuMoments(moments).flatten()
    hu_dict = {}
    for i, val in enumerate(hu):
        hu_dict[f'hu_{i+1}'] = -np.sign(val) * np.log10(abs(val)) if val != 0 else 0
    return hu_dict


# ---------- Helper: intensity and colour ----------
def _intensity_features(cnt, gray_img, color_img):
    """Mean, std, min, max, range of grayscale, plus average B,G,R values."""
    mask = np.zeros(gray_img.shape, dtype=np.uint8)
    cv2.drawContours(mask, [cnt], -1, 255, thickness=cv2.FILLED)

    pixels = gray_img[mask == 255]
    feat = {
        'mean_intensity': 0.0,
        'std_intensity': 0.0,
        'min_intensity': 0.0,
        'max_intensity': 0.0,
        'intensity_range': 0.0,
    }
    if len(pixels) > 0:
        feat['mean_intensity'] = float(np.mean(pixels))
        feat['std_intensity'] = float(np.std(pixels))
        feat['min_intensity'] = float(np.min(pixels))
        feat['max_intensity'] = float(np.max(pixels))
        feat['intensity_range'] = feat['max_intensity'] - feat['min_intensity']

    # Colour channels (only if the original image is colour)
    if len(color_img.shape) == 3:
        bgr = color_img[mask == 255]
        if len(bgr) > 0:
            mean_bgr = np.mean(bgr, axis=0)
            feat['mean_B'] = float(mean_bgr[0])
            feat['mean_G'] = float(mean_bgr[1])
            feat['mean_R'] = float(mean_bgr[2])
        else:
            feat['mean_B'] = feat['mean_G'] = feat['mean_R'] = 0.0

    return feat


# ---------- Main public function ----------
def extract_features(contours, original_image):
    """
    Extract all features for a list of contours.

    Parameters:
        contours        : list of contours (from detector)
        original_image  : BGR image (used for colour and texture)

    Returns:
        pandas.DataFrame with one row per defect and one column per feature.
    """
    if not contours:
        return pd.DataFrame()

    gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    all_rows = []

    for idx, cnt in enumerate(contours):
        row = {'defect_id': idx}
        row.update(_geometric_features(cnt))
        row.update(_shape_features(cnt))
        row.update(_hu_moments(cnt))
        row.update(_intensity_features(cnt, gray, original_image))
        all_rows.append(row)

    return pd.DataFrame(all_rows)


# ---------- Visualisation helper (optional) ----------
def visualize_features(original_image, contours, features_df):
    """Draw contours with IDs and show a small feature table."""
    import matplotlib.pyplot as plt

    img_copy = original_image.copy()
    for i, cnt in enumerate(contours):
        cv2.drawContours(img_copy, [cnt], -1, (0, 255, 0), 2)
        x, y, _, _ = cv2.boundingRect(cnt)
        cv2.putText(img_copy, str(i), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 255), 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.imshow(cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB))
    ax1.set_title(f'Defects detected: {len(contours)}')
    ax1.axis('off')

    if not features_df.empty:
        table_data = features_df[['defect_id', 'area', 'circularity',
                                  'aspect_ratio', 'mean_intensity']].head().round(2).values
        col_labels = ['ID', 'Area', 'Circ', 'Aspect', 'Mean Int']
        ax2.table(cellText=table_data, colLabels=col_labels, loc='center', cellLoc='center')
        ax2.axis('off')
        ax2.set_title('Sample features (first 5 defects)')
    else:
        ax2.text(0.5, 0.5, 'No defects', ha='center')
        ax2.axis('off')

    plt.tight_layout()
    plt.show()