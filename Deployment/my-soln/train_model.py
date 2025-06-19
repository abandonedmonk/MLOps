import pandas as pd
import pickle
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error


def train_model(df: pd.DataFrame, model_path: str = "model.bin") -> tuple[DictVectorizer, LinearRegression]:
    """Train a LinearRegression model and save it with DictVectorizer."""
    categorical = ['PULocationID', 'DOLocationID']
    numerical = ['trip_distance']
    target = 'duration'

    dv = DictVectorizer()
    train_dicts = df[categorical + numerical].to_dict(orient='records')
    X_train = dv.fit_transform(train_dicts)
    y_train = df[target].values

    lr = LinearRegression()
    lr.fit(X_train, y_train)

    y_pred = lr.predict(X_train)
    rmse = root_mean_squared_error(y_train, y_pred)
    print(f"Model intercept: {lr.intercept_}, Training RMSE: {rmse:.2f}")

    with open(model_path, 'wb') as f_out:
        pickle.dump((dv, lr), f_out)
    print(f"Model saved to {model_path}")

    return dv, lr


if __name__ == "__main__":
    import argparse
    from load_data import load_data
    from transform_data import transform_data

    parser = argparse.ArgumentParser(
        description="Train a model on Yellow Taxi data")
    parser.add_argument("--year", type=int, required=True,
                        help="Year of the data")
    parser.add_argument("--month", type=int, required=True,
                        help="Month of the data")
    parser.add_argument("--model-path", type=str,
                        default="model.bin", help="Path to save the model")
    args = parser.parse_args()

    df = load_data(args.year, args.month)
    df_transformed = transform_data(df)
    dv, model = train_model(df_transformed, args.model_path)
