import pandas as pd


def load_data():
    url = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-03.parquet"
    df = pd.read_parquet(url)
    print(f"Number of records loaded: {len(df)}")
    return df


if __name__ == "__main__":
    df = load_data()
