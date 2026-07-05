"""
run_detector.py
Main pipeline: detect defects, extract features, and (if model exists) classify them.
"""

from detector import load_image, detect_defects_template
from extractor import extract_features, visualize_features
from classifier import load_model, predict_defects

# ----------------------------------------------------------------------
# 🔧 Configuration – adjust these paths and parameters as needed
# ----------------------------------------------------------------------
REFERENCE_IMAGE = 'dataset/reference/good_sample.jpg'
TEST_IMAGE      = 'dataset/test/defective_sample.jpg'

THRESHOLD       = 30          # sensitivity; lower = more sensitive
MIN_AREA        = 100         # ignore tiny noise
BLUR_KERNEL     = (5, 5)

# ----------------------------------------------------------------------
# Step 1: Load images
# ----------------------------------------------------------------------
print("🔄 Loading images...")
ref = load_image(REFERENCE_IMAGE)
test = load_image(TEST_IMAGE)

# ----------------------------------------------------------------------
# Step 2: Detect defects
# ----------------------------------------------------------------------
print("🔍 Detecting defects...")
defects, mask = detect_defects_template(ref, test,
                                        blur_kernel=BLUR_KERNEL,
                                        threshold_value=THRESHOLD,
                                        min_contour_area=MIN_AREA,
                                        visualize=True)          # shows difference map

print(f"✅ Found {len(defects)} candidate defect regions.")

if not defects:
    print("No defects found. Pipeline finished.")
    exit()

# ----------------------------------------------------------------------
# Step 3: Extract features
# ----------------------------------------------------------------------
print("📏 Extracting features...")
features_df = extract_features(defects, test)   # use the test image for colour/texture
print(features_df.head())                       # preview

# ----------------------------------------------------------------------
# Step 4: Classify (if a trained model exists)
# ----------------------------------------------------------------------
try:
    model, scaler = load_model()
    predictions = predict_defects(features_df, model, scaler)
    features_df['predicted_label'] = predictions
    print("\n🏷️ Predicted defect types:")
    for i, lbl in enumerate(predictions):
        print(f"   Defect {i}: {lbl}")
except FileNotFoundError:
    print("\n⚠️ No trained model found. Skipping classification.")
    print("   Train one by labeling defect_features.csv and running classifier.py.")
    features_df['predicted_label'] = 'Unknown'

# ----------------------------------------------------------------------
# Step 5: Save results and visualise
# ----------------------------------------------------------------------
output_csv = 'defect_features_with_predictions.csv'
features_df.to_csv(output_csv, index=False)
print(f"\n💾 Results saved to {output_csv}")

# Show final visualisation with contours and a feature table
visualize_features(test, defects, features_df)