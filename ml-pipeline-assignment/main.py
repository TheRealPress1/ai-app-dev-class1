import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Breast Cancer Classifier API", version="1.0.0")

COL_NAMES = [
    "mean radius", "mean texture", "mean perimeter", "mean area",
    "mean smoothness", "mean compactness", "mean concavity",
    "mean concave points", "mean symmetry", "mean fractal dimension",
    "radius error", "texture error", "perimeter error", "area error",
    "smoothness error", "compactness error", "concavity error",
    "concave points error", "symmetry error", "fractal dimension error",
    "worst radius", "worst texture", "worst perimeter", "worst area",
    "worst smoothness", "worst compactness", "worst concavity",
    "worst concave points", "worst symmetry", "worst fractal dimension",
]

class TumorFeatures(BaseModel):
    mean_radius: float; mean_texture: float; mean_perimeter: float
    mean_area: float; mean_smoothness: float; mean_compactness: float
    mean_concavity: float; mean_concave_points: float; mean_symmetry: float
    mean_fractal_dimension: float; radius_se: float; texture_se: float
    perimeter_se: float; area_se: float; smoothness_se: float
    compactness_se: float; concavity_se: float; concave_points_se: float
    symmetry_se: float; fractal_dimension_se: float; worst_radius: float
    worst_texture: float; worst_perimeter: float; worst_area: float
    worst_smoothness: float; worst_compactness: float; worst_concavity: float
    worst_concave_points: float; worst_symmetry: float; worst_fractal_dimension: float

pipeline = None
TARGET_NAMES = ["malignant", "benign"]

def load_pipeline():
    global pipeline
    from pycaret.classification import load_model
    pipeline = load_model("best_pipeline")
    print("PyCaret pipeline loaded.")

load_pipeline()

@app.get("/")
def root():
    return {"message": "API running. Visit /docs to test."}

@app.post("/predict")
def predict(features: TumorFeatures):
    try:
        vals = [features.mean_radius, features.mean_texture, features.mean_perimeter,
            features.mean_area, features.mean_smoothness, features.mean_compactness,
            features.mean_concavity, features.mean_concave_points, features.mean_symmetry,
            features.mean_fractal_dimension, features.radius_se, features.texture_se,
            features.perimeter_se, features.area_se, features.smoothness_se,
            features.compactness_se, features.concavity_se, features.concave_points_se,
            features.symmetry_se, features.fractal_dimension_se, features.worst_radius,
            features.worst_texture, features.worst_perimeter, features.worst_area,
            features.worst_smoothness, features.worst_compactness, features.worst_concavity,
            features.worst_concave_points, features.worst_symmetry, features.worst_fractal_dimension]
        input_df = pd.DataFrame([vals], columns=COL_NAMES)
        from pycaret.classification import predict_model
        result = predict_model(pipeline, data=input_df)
        prediction = int(result["prediction_label"].iloc[0])
        score = float(result["prediction_score"].iloc[0])
        return {"prediction": TARGET_NAMES[prediction], "confidence": round(score, 4), "label": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
