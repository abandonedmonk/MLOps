import pandas as pd
import pickle
import mlflow
import mlflow.sklearn
import argparse
from mlflow.tracking import MlflowClient
from load_data import load_data
from transform_data import transform_data


def load_model(model_path: str = None, experiment_name: str = None, mlflow_model_name: str = None, mlflow_run_id: str = None):
    """Load DictVectorizer and model from MLflow or local file."""
    if model_path:
        with open(model_path, 'rb') as f_in:
            dv, model = pickle.load(f_in)
    else:
        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        client = MlflowClient()

        if experiment_name:
            # Get the latest run from the specified experiment
            experiment = client.get_experiment_by_name(experiment_name)
            if not experiment:
                raise ValueError(f"Experiment '{experiment_name}' not found")
            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=["start_time DESC"],
                max_results=1
            )
            if not runs:
                raise ValueError(
                    f"No runs found in experiment '{experiment_name}'")
            run_id = runs[0].info.run_id
            dv_uri = f"runs:/{run_id}/dict-vectorizer"
            model_uri = f"runs:/{run_id}/linear-regression-model"
        elif mlflow_run_id:
            dv_uri = f"runs:/{mlflow_run_id}/dict-vectorizer"
            model_uri = f"runs:/{mlflow_run_id}/linear-regression-model"
        elif mlflow_model_name:
            dv_uri = f"models:/{mlflow_model_name}/latest"
            model_uri = f"models:/{mlflow_model_name}/latest"
        else:
            raise ValueError(
                "Must specify model_path, experiment_name, mlflow_model_name, or mlflow_run_id")

        dv = mlflow.sklearn.load_model(dv_uri)
        model = mlflow.sklearn.load_model(model_uri)
    return dv, model


def predict(df: pd.DataFrame, dv, model) -> pd.DataFrame:
    """Make predictions on the input DataFrame."""
    categorical = ['PULocationID', 'DOLocationID']
    numerical = ['trip_distance']
    dicts = df[categorical + numerical].to_dict(orient='records')
    X = dv.transform(dicts)
    df['predicted_duration'] = model.predict(X)
    return df


def save_results(df: pd.DataFrame, year: int, month: int, output_path: str) -> None:
    """Save ride_id and predicted_duration to a parquet file."""
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
    df_result = df[['ride_id', 'predicted_duration']]
    df_result.to_parquet(
        output_path,
        engine='pyarrow',
        compression=None,
        index=False
    )
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make batch predictions")
    parser.add_argument("--year", type=int, required=True,
                        help="Year of the data")
    parser.add_argument("--month", type=int, required=True,
                        help="Month of the data")
    parser.add_argument("--model-path", type=str,
                        default=None, help="Path to local model file")
    parser.add_argument("--experiment-name", type=str,
                        default=None, help="MLflow experiment name")
    parser.add_argument("--mlflow-model-name", type=str,
                        default=None, help="MLflow model name")
    parser.add_argument("--mlflow-run-id", type=str,
                        default=None, help="MLflow run ID")
    parser.add_argument("--output-path", type=str,
                        default="predictions.parquet", help="Path to save predictions")
    args = parser.parse_args()

    # Load and transform data
    df = load_data(args.year, args.month)
    df_transformed = transform_data(df)

    # Load model and predict
    dv, model = load_model(args.model_path, args.experiment_name,
                           args.mlflow_model_name, args.mlflow_run_id)
    df_pred = predict(df_transformed, dv, model)

    # Compute metrics
    mean_pred = df_pred['predicted_duration'].mean()
    std_pred = df_pred['predicted_duration'].std()
    print(f"Mean predicted duration: {mean_pred:.2f}")
    print(f"Standard deviation of predicted duration: {std_pred:.2f}")

    # Save results
    save_results(df_pred, args.year, args.month, args.output_path)
