from fastapi import APIRouter, Request
from loguru import logger
from auto_ml.train import train
from auto_ml.validate import validate
import pandas as pd
import json
router = APIRouter()


@router.post("/train/")
async def train_model(input: Request):
    input = await input.json()
    target_variable = input["target_variable"]
    id_column = input["id_column"]
    return_metrics = input["return_metrics"]
    data = pd.DataFrame(input["data"])
    model_path, metrics = train(data, target_variable, id_column)
    if return_metrics:
        return {"model_path": model_path, "metrics": metrics}
    return {"model_path": model_path}


@router.post("/validate/")
async def validate_performance(input: Request):
    input = await input.json()
    data = input["data"]
    model_path = input["model_path"]
    id_column = input["id_column"]
    data = pd.DataFrame(data)
    predictions = validate(data, model_path, id_column)
    return {"predictions": predictions.to_dict(orient="records")}
