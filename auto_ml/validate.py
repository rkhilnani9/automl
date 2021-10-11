import h2o
import pandas as pd

h2o.init(max_mem_size="16G")


def validate(data, model_path, id_column=None):
    print(model_path)
    model = h2o.load_model(model_path)

    if id_column:
        data.drop(id_column, axis=1, inplace=True)

    h2o_df = h2o.H2OFrame(data)

    # perf = model.model_performance(h2o_df)
    # print(perf)

    predictions = model.predict(h2o_df)

    return predictions.as_data_frame(use_pandas=True)
