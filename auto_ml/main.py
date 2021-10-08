from fastapi import APIRouter, Form, File, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
from io import BytesIO, StringIO
from loguru import logger

from auto_ml.automl import train_model

import pandas as pd


router = APIRouter()


@router.post("/train/")
async def train(
    target_variable: str = Form(...),
    id_column: Optional[str] = Form(...),
    dataframe: UploadFile = File(...),
):
    data = pd.read_csv(BytesIO(dataframe.file.read()))
    logger.info(dataframe.filename)
    pred = train_model(data, target_variable, id_column)
    stream = StringIO()
    pred.to_csv(stream, index=False)
    response = StreamingResponse(iter([stream.getvalue()]), media_type="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=predictions.csv"
    return response
