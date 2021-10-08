from fastapi import FastAPI, Form, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from typer import Typer

from auto_ml.main import router as auto_ml_router
from auto_ml.automl import train_model

from fastapi.responses import StreamingResponse
from typing import Optional
from io import BytesIO, StringIO
import pandas as pd
from loguru import logger

app = FastAPI()
app.include_router(auto_ml_router, prefix="/auto_ml")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

cli = Typer()


@cli.command()
def validate(target_variable, id_column, data):
    return train_model(data, target_variable, id_column)


if __name__ == "__main__":
    cli()
