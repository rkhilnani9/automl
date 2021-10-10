from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from typer import Typer

from auto_ml.main import router as auto_ml_router
from auto_ml.train import train_model
from auto_ml.validate import validate


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
def train(target_variable, id_column, data):
    return train_model(data, target_variable, id_column)


@cli.command()
def validate(model_path, id_column, data):
    return validate(data, model_path, id_column)


if __name__ == "__main__":
    cli()
