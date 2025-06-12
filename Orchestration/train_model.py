import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error


def train_model(df):
    categorical = ['PULocationID', 'DOLocationID']
    numerical = ['trip_distance']
    target = 'duration'

    dv = DictVectorizer()
    train_dicts = df[categorical + numerical].to_dict(orient='records')
    X_train = dv.fit_transform(train_dicts)
    y_train = df[target].values

    lr = LinearRegression()
    lr.fit(X_train, y_train)

    print(f"Model intercept: {lr.intercept_}")

    return dv, lr


if __name__ == "__main__":
    df = pd.read_parquet("yellow_tripdata_2023-03.parquet")
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df.duration = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)]
    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)

    dv, model = train_model(df)
