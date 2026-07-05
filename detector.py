"""
detector.py
Detects potential defect regions using absolute difference + thresholding.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_image(image_path: str, color_flag=cv2.IMREAD_COLOR) -> np.ndarray:
    """Load an image; raises an error if the file is not found."""
    img = cv2.imread(image_path, color_flag)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    return img


def detect_defects_template(reference_img: np.ndarray, test_img: np.ndarray,
                            blur_kernel=(5, 5), threshold_value=30,
                            min_contour_area=100, visualize=False):
    """
    Detect defect regions by subtracting a blurred reference from a blurred test image.

    Parameters:
        reference_img : BGR image of the perfect reference.
        test_img      : BGR image under inspection.
        blur_kernel   : Gaussian kernel size for noise reduction.
        threshold_value: Lower = more sensitive; adjust per image.
        min_contour_area: Ignore regions smaller than this (pixels).
        visualize     : If True, show intermediate steps.

    Returns:
        defects      : List of contours (each a numpy array of points).
        defect_mask  : Binary mask highlighting all detected defects.
    """
    # Convert to grayscale and blur to reduce noise
    ref_gray = cv2.cvtColor(reference_img, cv2.COLOR_BGR2GRAY)
    test_gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    ref_blurred = cv2.GaussianBlur(ref_gray, blur_kernel, 0)
    test_blurred = cv2.GaussianBlur(test_gray, blur_kernel, 0)

    # Absolute pixel‑wise difference
    diff = cv2.absdiff(ref_blurred, test_blurred)

    # Threshold to get binary regions of significant change
    _, thresh = cv2.threshold(diff, threshold_value, 255, cv2.THRESH_BINARY)

    # Morphological cleaning: close gaps and remove isolated noise
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

    # Find contours (external only)
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter by minimum area
    defects = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]

    # Create a binary mask for visualisation / further processing
    defect_mask = np.zeros_like(ref_gray)
    cv2.drawContours(defect_mask, defects, -1, 255, thickness=cv2.FILLED)

    # Optional visualisation
    if visualize:
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 4, 1); plt.imshow(ref_gray, cmap='gray')
        plt.title('Reference'); plt.axis('off')
        plt.subplot(1, 4, 2); plt.imshow(test_gray, cmap='gray')
        plt.title('Test Image'); plt.axis('off')
        plt.subplot(1, 4, 3); plt.imshow(diff, cmap='hot')
        plt.title('Difference Map'); plt.axis('off')
        plt.subplot(1, 4, 4); plt.imshow(defect_mask, cmap='gray')
        plt.title(f'Defects: {len(defects)}'); plt.axis('off')
        plt.show()

    return defects, defect_mask