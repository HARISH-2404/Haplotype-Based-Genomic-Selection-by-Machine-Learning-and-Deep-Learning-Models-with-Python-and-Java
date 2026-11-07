from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict
from typing import List
import joblib
import pandas as pd
from .config import MODELS

app = FastAPI(title="Cotton Haplotype Genomic Prediction API",
              version="1.0.0")

class PredictionRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    trait: str
    feature_set: str = Field(alias="featureSet")
    model: str
    feature_names: List[str] = Field(alias="featureNames")
    genotypes: List[str]
    matrix: List[List[float]]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(req: PredictionRequest):
    if req.feature_set not in {"SNP", "Haplotype", "Integrated"}:
        raise HTTPException(400, "Invalid feature_set")
    if len(req.genotypes) != len(req.matrix):
        raise HTTPException(400, "genotypes and matrix row counts differ")
    if any(len(row) != len(req.feature_names) for row in req.matrix):
        raise HTTPException(400, "matrix width differs from feature_names")
    path = MODELS / f"{req.trait}_{req.feature_set}_{req.model}.joblib"
    try:
        bundle = joblib.load(path)
    except FileNotFoundError:
        raise HTTPException(404, f"Model not found: {path.name}")
    if req.feature_names != bundle["features"]:
        raise HTTPException(400, "feature_names/order do not match trained model")
    X = pd.DataFrame(req.matrix, columns=req.feature_names, index=req.genotypes)
    pred = bundle["estimator"].predict(X)
    return {"trait": req.trait, "model": req.model, "feature_set": req.feature_set,
            "predictions": [
                {"genotype": gid, "predicted_adjusted_performance": float(value)}
                for gid, value in zip(req.genotypes, pred)
            ]}
