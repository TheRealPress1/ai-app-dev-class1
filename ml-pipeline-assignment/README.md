# Breast Cancer Classifier — Automated ML Pipeline

**Major Assignment | OIM 3641 AI-Driven App Development | Gaspard Seuge**

## Overview

Compares manual (scikit-learn) and low-code (PyCaret) ML workflows on the
Breast Cancer Wisconsin dataset, then deploys the best model as a FastAPI service.

## Dataset

- **Source:** UCI ML Repository / sklearn.datasets
- **Rows:** 1,707 (augmented from 569 base samples)
- **Features:** 30 numeric diagnostic measurements
- **Target:** malignant (0) or benign (1)

## Results

| Model | Accuracy |
|---|---|
| **Gradient Boosting** | **99.12%** |
| Random Forest | 97.37% |
| Extra Trees | 97.37% |
| Logistic Regression | 96.49% |
| SVM | 95.61% |

## Files

- `discovery.py` — PyCaret + sklearn comparison
- `main.py` — FastAPI deployment
- `report.docx` — Screenshots and analysis

## How to Run

```bash
pip install pycaret fastapi uvicorn scikit-learn
python discovery.py
uvicorn main:app --reload
```

Visit http://localhost:8000/docs

## Sample API Input / Output

**POST /predict**

Input:
```json
{
  "mean_radius": 17.99, "mean_texture": 10.38, "mean_perimeter": 122.8,
  "mean_area": 1001.0, "mean_smoothness": 0.1184, "mean_compactness": 0.2776,
  "mean_concavity": 0.3001, "mean_concave_points": 0.1471,
  "mean_symmetry": 0.2419, "mean_fractal_dimension": 0.07871,
  "radius_se": 1.095, "texture_se": 0.9053, "perimeter_se": 8.589,
  "area_se": 153.4, "smoothness_se": 0.006399, "compactness_se": 0.04904,
  "concavity_se": 0.05373, "concave_points_se": 0.01587,
  "symmetry_se": 0.03003, "fractal_dimension_se": 0.006193,
  "worst_radius": 25.38, "worst_texture": 17.33, "worst_perimeter": 184.6,
  "worst_area": 2019.0, "worst_smoothness": 0.1622, "worst_compactness": 0.6656,
  "worst_concavity": 0.7119, "worst_concave_points": 0.2654,
  "worst_symmetry": 0.4601, "worst_fractal_dimension": 0.1189
}
```

Output:
```json
{
  "prediction": "malignant",
  "confidence": 0.9987,
  "label": 0
}
```
