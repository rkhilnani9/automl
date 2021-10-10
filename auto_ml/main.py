from fastapi import APIRouter, Form, File, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
from io import BytesIO, StringIO
from loguru import logger

from auto_ml.train import train_model
from auto_ml.validate import validate

import pandas as pd


router = APIRouter()


@router.post("/train/")
async def train(
    target_variable: str = Form(...),
    id_column: Optional[str] = Form(...),
    dataframe: UploadFile = File(...),
):
    data = pd.read_csv(dataframe.file)
    logger.info(dataframe.filename)
    model_path = train_model(data, target_variable, id_column)
    return {"model_path": model_path}


@router.post("/validate/")
def validate(
    model_path: str = Form(...),
    id_column: Optional[str] = Form(...),
    dataframe: UploadFile = File(...),
):
    data = pd.read_csv(dataframe.file)
    logger.info(dataframe.filename)
    pred_df = validate(data, model_path, id_column)
    print(pred_df)
    stream = StringIO()
    pred_df.to_csv(stream, index=False)
    response = StreamingResponse(iter([stream.getvalue()]), media_type="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=predictions.csv"
    return {"message": "success"}
