"""
app.py
Streamlit Web Application for AOI Defect Classification.
Upload reference and test images, run the pipeline, and view results.
"""

import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import os
import tempfile

# Import your existing modules
from detector import detect_defects_template
from extractor import extract_features
from classifier import load_model, predict_defects

# ------------------ Page Configuration ------------------
st.set_page_config(page_title="AOI Defect Classifier", layout="wide")
st.title("🔍 AOI Defect Classification System")
st.markdown("Upload a **reference (golden)** image and a **test** image to detect and classify defects.")

# ------------------ Helper Functions ------------------
def load_uploaded_image(uploaded_file):
    """Convert uploaded image file to OpenCV BGR format."""
    if uploaded_file is None:
        return None
    # Read as PIL, convert to numpy RGB, then BGR for OpenCV
    pil_img = Image.open(uploaded_file)
    rgb_img = np.array(pil_img)
    # Convert RGB to BGR (OpenCV uses BGR)
    bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
    return bgr_img

def draw_results(image, contours, labels=None):
    """
    Draw contours and labels on the image.
    - contours: list of defect contours
    - labels: list of predicted label strings (optional)
    """
    img_copy = image.copy()
    for idx, cnt in enumerate(contours):
        # Draw contour in green
        cv2.drawContours(img_copy, [cnt], -1, (0, 255, 0), 2)
        # Get bounding box for text position
        x, y, w, h = cv2.boundingRect(cnt)
        # Draw rectangle for clarity (optional)
        cv2.rectangle(img_copy, (x, y), (x+w, y+h), (255, 0, 0), 1)
        # Prepare label text
        if labels and idx < len(labels):
            label_text = labels[idx]
        else:
            label_text = f"Defect {idx}"
        # Put text above the defect
        cv2.putText(img_copy, label_text, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return img_copy

# ------------------ UI Layout ------------------
# Two columns for file uploads
col1, col2 = st.columns(2)

with col1:
    st.subheader("📸 Reference Image (Golden)")
    ref_file = st.file_uploader("Upload reference image", type=["jpg", "jpeg", "png"], key="ref")

with col2:
    st.subheader("🔬 Test Image (Inspection)")
    test_file = st.file_uploader("Upload test image", type=["jpg", "jpeg", "png"], key="test")

# ------------------ Processing ------------------
if ref_file and test_file:
    # Load images
    with st.spinner("Loading images..."):
        ref_img = load_uploaded_image(ref_file)
        test_img = load_uploaded_image(test_file)

    # Display uploaded images side by side
    st.subheader("📋 Uploaded Images")
    preview_col1, preview_col2 = st.columns(2)
    with preview_col1:
        st.image(ref_file, caption="Reference", use_container_width=True)
    with preview_col2:
        st.image(test_file, caption="Test", use_container_width=True)

    # ----- Step 1: Detect Defects -----
    with st.spinner("🔍 Detecting defects..."):
        defects, mask = detect_defects_template(ref_img, test_img,
                                                threshold_value=30,
                                                min_contour_area=100,
                                                visualize=False)

    st.info(f"✅ Found {len(defects)} candidate defect regions.")

    if len(defects) == 0:
        st.success("🎉 No defects detected! The test image matches the reference.")
        st.stop()

    # ----- Step 2: Extract Features -----
    with st.spinner("📏 Extracting features..."):
        features_df = extract_features(defects, test_img)

    # ----- Step 3: Classify (if model exists) -----
    labels = None
    try:
        with st.spinner("🧠 Loading classifier and predicting..."):
            model, scaler = load_model()
            predicted = predict_defects(features_df, model, scaler)
            labels = list(predicted)
            features_df['predicted_label'] = labels
            st.success("✅ Classification complete!")
    except FileNotFoundError:
        st.warning("⚠️ Trained model not found. Showing defect locations only (no labels).")
        labels = [f"Defect {i}" for i in range(len(defects))]
        features_df['predicted_label'] = 'Unknown'

    # ----- Step 4: Display Results -----
    st.subheader("📊 Detection & Classification Results")

    # Draw results on the test image
    result_img = draw_results(test_img, defects, labels)

    # Show the result image
    st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB),
             caption="Defect Detection Results",
             use_container_width=True)

    # Show feature table
    st.subheader("📋 Defect Feature Details")
    st.dataframe(features_df, use_container_width=True)

    # Download button for CSV
    csv = features_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="💾 Download Results as CSV",
        data=csv,
        file_name="defect_results.csv",
        mime="text/csv",
    )

else:
    st.info("👆 Please upload both a reference image and a test image to begin.")

# ------------------ Footer ------------------
st.markdown("---")
st.caption("Built with Streamlit • AOI Defect Classification Pipeline")