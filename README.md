# 🔍 AOI Defect Classification System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.5%2B-green.svg)](https://opencv.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red.svg)](https://streamlit.io/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0%2B-orange.svg)](https://scikit-learn.org/)

An end‑to‑end **Automated Optical Inspection (AOI)** system that detects and classifies surface defects (scratch, dent, contamination) from industrial images. Built with a modular pipeline: **Detector → Feature Extractor → Classifier**, wrapped in an interactive **Streamlit** web interface.

---

## ✨ Features

- 🔎 **Defect Detection** – Template matching via image differencing (golden reference vs. test image) with morphological cleaning.
- 📊 **Feature Extraction** – Computes 24 geometric, shape (Hu moments, circularity, convexity), texture (intensity stats), and colour (RGB) features per defect.
- 🧠 **Machine Learning** – Trains a **Random Forest** classifier (Scikit‑learn) with model persistence (joblib).
- 🖥️ **Web Dashboard** – Interactive Streamlit app: upload images, visualize detected defects with bounding boxes, view feature tables, and export results as CSV.
- ⚡ **Fast & Lightweight** – Processes a pair of images in under 5 seconds.
