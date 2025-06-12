import pandas as pd


def transform_data(df):
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df.duration = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)

    print(f"Size of the resulting DataFrame: {len(df)}")
    return df


if __name__ == "__main__":
    df = pd.read_parquet("yellow_tripdata_2023-03.parquet")
    df_transformed = transform_data(df)
