"""
classifier.py
Trains a Random Forest classifier on extracted features and predicts defect types.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

# Paths where the trained model and scaler will be stored
MODEL_PATH = 'defect_classifier.pkl'
SCALER_PATH = 'scaler.pkl'


def train_classifier(csv_path='defect_features.csv'):
    """
    Load labeled CSV, train a Random Forest, save model + scaler.

    The CSV must have a column named 'label' with the defect type strings.
    """
    df = pd.read_csv(csv_path)

    if 'label' not in df.columns:
        raise ValueError("CSV must contain a 'label' column with defect types.")

    # All columns except 'defect_id' and 'label' are features
    feature_cols = [col for col in df.columns if col not in ['defect_id', 'label']]
    X = df[feature_cols]
    y = df['label']

    # Replace any missing values with the column mean (simple but safe)
    X = X.fillna(X.mean())

    # Train / test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardise features (important for distance‑based algorithms)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Random Forest with 100 trees
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    # Evaluate
    y_pred = model.predict(X_test_scaled)
    print("=" * 60)
    print("📊 Model Evaluation")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("=" * 60)

    # Save
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"✅ Model saved to {MODEL_PATH}")
    print(f"✅ Scaler saved to {SCALER_PATH}")

    return model, scaler


def load_model():
    """Load the saved model and scaler."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file '{MODEL_PATH}' not found. Train first using train_classifier()."
        )
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


def predict_defects(features_df, model=None, scaler=None):
    """
    Predict defect types for a DataFrame of features (without labels).

    Returns a list of predicted strings (e.g., ['scratch', 'dent', ...]).
    """
    if model is None or scaler is None:
        model, scaler = load_model()

    # Select feature columns (exclude 'defect_id' and 'label' if they exist)
    feature_cols = [col for col in features_df.columns if col not in ['defect_id', 'label']]
    X_new = features_df[feature_cols]
    X_new = X_new.fillna(X_new.mean())

    # Scale using the same scaler used during training
    X_new_scaled = scaler.transform(X_new)

    return model.predict(X_new_scaled)


# If this script is run directly, you can uncomment the line below to train
if __name__ == "__main__":
    # train_classifier()   # <-- uncomment after labeling your CSV
    pass