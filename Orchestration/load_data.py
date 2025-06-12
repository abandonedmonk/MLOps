import pandas as pd


def load_data():
    df = pd.read_parquet(
        "D:/ML Final/MLOps/Orchestration/yellow_tripdata_2023-03.parquet")
    print(f"Number of records loaded: {len(df)}")
    return df


if __name__ == "__main__":
    df = load_data()
