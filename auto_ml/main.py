from fastapi import APIRouter, Form, File, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
from io import BytesIO, StringIO
from loguru import logger

from auto_ml.train import train
from auto_ml.validate import validate

import pandas as pd


router = APIRouter()


@router.post("/train/")
async def train_model(
    target_variable: str = Form(...),
    id_column: Optional[str] = Form(...),
    return_metrics: Optional[bool] = Form(...),
    dataframe: UploadFile = File(...),
):

    data = pd.read_csv(dataframe.file)
    logger.info(dataframe.filename)
    model_path, metrics = train(data, target_variable, id_column)
    if return_metrics:
        return {"model_path": model_path, "metrics": metrics}
    return {"model_path": model_path}


@router.post("/validate/")
def validate_performance(
    dataframe: UploadFile = File(...),
    model_path: str = Form(...),
    id_column: Optional[str] = Form(...)
):
    data = pd.read_csv(dataframe.file)
    logger.info(dataframe.filename)
    predictions = validate(data, model_path, id_column)
    return {"predictions": predictions.to_dict(orient="records")}
