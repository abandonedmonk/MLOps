import pickle


with open("Deployment\web-service\model.pkl", 'rb') as f:
    dv, model = pickle.load(f)


# Function to prepare features for prediction
def prepare_features(ride):
    features = {}
    features['PU_DO'] = '%s_%s' % (ride['PULocationID'], ride['DOLocationID'])
    features['trip_distance'] = ride['trip_distance']
    return features


# Function to predict the fare
def predict(features):
    X = dv.trasnform(features)
    y_pred = model.predict(X)
    return y_pred
