import pandas as pd


def transform_data(df: pd.DataFrame) -> pd.DataFrame:
    """Transform the input DataFrame by computing duration and filtering outliers."""
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df['duration'].dt.total_seconds() / 60
    df = df[(df['duration'] >= 1) & (df['duration'] <= 60)].copy()
    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    print(f"Size of the resulting DataFrame: {len(df)}")
    return df


if __name__ == "__main__":
    import argparse
    from load_data import load_data

    parser = argparse.ArgumentParser(description="Transform Yellow Taxi data")
    parser.add_argument("--year", type=int, required=True,
                        help="Year of the data")
    parser.add_argument("--month", type=int, required=True,
                        help="Month of the data")
    args = parser.parse_args()

    df = load_data(args.year, args.month)
    df_transformed = transform_data(df)
