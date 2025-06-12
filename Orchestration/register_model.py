import mlflow
import mlflow.sklearn
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
import pandas as pd


def register_model(dv, model):
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("yellow-taxi-linear-regression")

    with mlflow.start_run():
        mlflow.sklearn.log_model(model, "linear-regression-model")
        mlflow.sklearn.log_model(dv, "dict-vectorizer")

        model_uri = f"runs:/{mlflow.active_run().info.run_id}/linear-regression-model"
        mlflow.register_model(model_uri, "YellowTaxiLinearRegressor")


if __name__ == "__main__":
    # For standalone testing
    df = pd.read_parquet("yellow_tripdata_2023-03.parquet")
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df.duration = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)]
    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)

    dv = DictVectorizer()
    train_dicts = df[categorical + ['trip_distance']].to_dict(orient='records')
    X_train = dv.fit_transform(train_dicts)
    y_train = df['duration'].values

    lr = LinearRegression()
    lr.fit(X_train, y_train)

    register_model(dv, lr)
