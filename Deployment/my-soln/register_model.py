import mlflow
import mlflow.sklearn
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression


def register_model(dv: DictVectorizer, model: LinearRegression, experiment_name: str = "yellow-taxi-linear-regression"):
    """Register the model and DictVectorizer with MLflow."""
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():
        mlflow.sklearn.log_model(model, "linear-regression-model")
        mlflow.sklearn.log_model(dv, "dict-vectorizer")
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/linear-regression-model"
        mlflow.register_model(model_uri, "YellowTaxiLinearRegressor")
        print(
            f"Model registered in MLflow under experiment: {experiment_name}")


if __name__ == "__main__":
    import argparse
    from load_data import load_data
    from transform_data import transform_data
    from train_model import train_model

    parser = argparse.ArgumentParser(description="Register model in MLflow")
    parser.add_argument("--year", type=int, required=True,
                        help="Year of the data")
    parser.add_argument("--month", type=int, required=True,
                        help="Month of the data")
    parser.add_argument("--experiment-name", type=str,
                        default="yellow-taxi-linear-regression", help="MLflow experiment name")
    args = parser.parse_args()

    df = load_data(args.year, args.month)
    df_transformed = transform_data(df)
    dv, model = train_model(df_transformed)
    register_model(dv, model, args.experiment_name)
