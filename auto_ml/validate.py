import pickle
import pandas as pd
from loguru import logger

from auto_ml.train_utils import preprocess_data


def validate(data, model_path, id_column=None):
    logger.info(model_path)
    model = pickle.load(open(f"{model_path}.pkl", "rb"))

    df = preprocess_data(data)

    if id_column:
        df.drop(id_column, axis=1, inplace=True)

    predictions = model.predict(df.values)

    predictions = [int(pred) for pred in predictions]

    return list(predictions)
