# 🌾 Nutrient Deficiency Detection using MobileNetV2

This project is part of the **AI for AgriTech Hackathon – Stage 1**. It focuses on detecting nutrient deficiency in wheat crops using vegetation indices — NDVI (Healthy) and GNDVI (Deficient) — through a CNN-based classification model built on MobileNetV2.

---

## 📦 Dataset

> Due to GitHub size constraints, only sample images are included. The full dataset with 8000 images can be downloaded from [Kaggle](https://www.kaggle.com/datasets/masiaslahi/rgbnir-aerial-crop-dataset).

**Structure:**

📁 Dataset/
├── raw/
│ ├── Wheat13082019/ → 3 sample RGB images
│ ├── Wheat27072019/ → 3 sample RGB images
│ └── Wheat30082019/ → 3 sample RGB images
├── processed/
│ ├── NDVI/
│ │ ├── Wheat13082019/ → 3 images
│ │ ├── Wheat27072019/ → 3 images
│ │ └── Wheat30082019/ → 3 images
│ └── GNDVI/
│ ├── Wheat13082019/ → 3 images
│ ├── Wheat27072019/ → 3 images
│ └── Wheat30082019/ → 3 images


---

## 🧠 Model Overview

- Architecture: **MobileNetV2**
- Total Parameters: ~2.2M (only 1.2K trainable)
- Training Accuracy: **98%**
- Validation Accuracy: **99%**
- Final Test Accuracy: **85%**
- Loss: `binary_crossentropy`
- Optimizer: `adam`

---

## 📁 Project Contents

📁 Models/
├── best_model_mobilenet.h5
├── class_indices_mobilenet.json
├── training_plot.png
└── confusion_matrix.png

📁 Dataset/
├── raw/
└── processed/

📄 model_training_mobilenet.py → Model training script
📄 app_mobilenet.py → Streamlit app
📄 vegetation_index.py → NDVI/GNDVI processor
📄 requirements.txt → Dependencies list

---

## 🚀 How to Run

```bash
# 1️⃣ Install requirements
pip install -r requirements.txt

# 2️⃣ Launch the app
streamlit run app_mobilenet.py
✅ Output
NDVI → 🌱 Healthy

GNDVI → 🍂 Deficient

UI is clean, responsive, and optimized for local use via Streamlit.

🏁 Final Note
All outputs belong to the Models/ folder — including the best model, class mappings, training and confusion plots.
Only sample data is uploaded; the full dataset (8000 images) must be downloaded externally from Kaggle.
This project is the result of extensive testing and optimization done entirely on a local machine (16GB RAM) to ensure performance and stability.
