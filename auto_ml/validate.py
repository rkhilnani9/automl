import h2o


def validate(data, model_path, id_column=None):
    model = h2o.load_model(model_path)

    h2o_df = h2o.H2OFrame(data)
    cols = h2o_df.columns
    if id_column:
        cols.remove(id_column)

    predictions = model.predict(h2o_df)

    print(predictions)
    return predictions
