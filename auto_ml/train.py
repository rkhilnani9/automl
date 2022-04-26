import datetime
import pickle
import xgboost as xgb

from sklearn.model_selection import train_test_split
from auto_ml.train_utils import get_best_hyper_params, preprocess_data


def train(data, target_variable, id_column=None):
    problem, hyper_params, metrics = get_best_hyper_params(
        data, target_variable, id_column
    )

    print(hyper_params["ntrees"]["actual"], hyper_params["learn_rate"]["actual"], hyper_params["max_depth"]["actual"], hyper_params["sample_rate"]["actual"])

    if problem == "classification":
        model = xgb.XGBClassifier(
            n_estimators=hyper_params["ntrees"]["actual"],
            learning_rate=hyper_params["learn_rate"]["actual"],
            max_depth=hyper_params["max_depth"]["actual"],
            subsample=hyper_params["sample_rate"]["actual"],
        )
    else:
        model = xgb.XGBRegressor(
            n_estimators=hyper_params["ntrees"]["actual"],
            learning_rate=hyper_params["learn_rate"]["actual"],
            max_depth=hyper_params["max_depth"]["actual"],
            subsample=hyper_params["sample_rate"]["actual"],
        )

    df = preprocess_data(data)

    if id_column:
        df.drop(id_column, axis=1, inplace=True)

    y = df[target_variable].values
    X = df.drop(target_variable, axis=1).values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Train sklearn model
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    # Save model
    model_path = f"models/{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"

    # Model for validating
    pickle.dump(model, open(f"{model_path}.pkl", "wb"))

    # Model for deployment
    model.save_model(model_path)

    return model_path, metrics
