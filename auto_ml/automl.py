import h2o
from h2o.automl import H2OAutoML

h2o.init(max_mem_size="16G")


def train_model(data, target_variable, id_column=None):
    # Split into train and validation
    h2o_df = h2o.H2OFrame(data)
    splits = h2o_df.split_frame(ratios=[0.8], seed=1)
    train, test = splits[0], splits[1]

    # Split into x and y
    y = target_variable
    x = h2o_df.columns
    x.remove(y)
    if id_column:
        x.remove(id_column)

    # Train the model
    aml = H2OAutoML(max_runtime_secs=5, seed=1)
    aml.train(x=x, y=y, training_frame=train)

    # Get metrics of best model
    lb = aml.leaderboard.as_data_frame(use_pandas=True)
    metrics = lb.to_dict(orient="records")[0]
    print(metrics)

    # Get predictions with best model
    pred = aml.predict(test).as_data_frame(use_pandas=True)

    # if save_model:
    #     h2o.save_model(aml.leader, path=model_save_path)

    return pred
