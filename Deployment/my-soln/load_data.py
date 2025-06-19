import pandas as pd


def load_data(year: int, month: int) -> pd.DataFrame:
    """Load Yellow Taxi data from NYC TLC for a given year and month"""

    url = f"https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet"
    df = pd.read_parquet(url)
    print(f"Number of records loaded for {year}-{month:02d}: {len(df)}")
    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Load Yellow Taxi data")
    parser.add_argument("--year", type=int, required=True,
                        help="Year of the data")
    parser.add_argument("--month", type=int, required=True,
                        help="Month of the data")
    args = parser.parse_args()

    df = load_data(args.year, args.month)
