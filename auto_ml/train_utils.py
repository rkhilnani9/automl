import h2o
from h2o.automl import H2OAutoML
from sklearn.preprocessing import LabelEncoder


def get_best_hyper_params(data, target_variable, id_column=None):
    # Split into train and validation
    h2o_df = h2o.H2OFrame(data)

    # Classification is supported for upto 5 classes
    if h2o_df[target_variable].unique().shape[0] <= 5:
        h2o_df[target_variable] = h2o_df[target_variable].asfactor()
        problem = "classification"
    else:
        problem = "regression"
    splits = h2o_df.split_frame(ratios=[0.8], seed=1)
    train, test = splits[0], splits[1]

    # Split into x and y
    y = target_variable
    x = h2o_df.columns
    x.remove(y)
    if id_column:
        x.remove(id_column)

    # Train model
    aml = H2OAutoML(max_runtime_secs=30, seed=1)
    aml.train(x=x, y=y, training_frame=train)

    # Get model ids for all models in the AutoML Leaderboard
    model_ids = list(aml.leaderboard['model_id'].as_data_frame().iloc[:, 0])
    # Get the GBM model
    m = h2o.get_model([mid for mid in model_ids if "GBM" in mid][0])

    metrics = m.model_performance(test)
    hyper_params = m.params

    # # Save model
    # save_path = "../models"
    # model_path = h2o.save_model(model=aml.leader, path=save_path, force=True)

    # # Return metrics
    # lb = aml.leaderboard.as_data_frame(use_pandas=True)
    # metrics = lb.to_dict(orient="records")[0]

    return problem, hyper_params, metrics


def preprocess_data(df):
    object_cols = [col for col in df.columns if df[col].dtype == "O"]
    numerical_cols = [
        col
        for col in df.columns
        if (df[col].dtype == "float") | (df[col].dtype == "int")
    ]

    for col in object_cols:
        if df[col].nunique() <= 5:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            df[col].fillna(0, inplace=True)
        else:
            df.drop(col, axis=1, inplace=True)

    for col in numerical_cols:
        df[col].fillna(df[col].mean(), inplace=True)
        df[col] = df[col].round(2)

    return df

