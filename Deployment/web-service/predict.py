import pickle
from flask import Flask, request, jsonify
app = Flask('fare-predictor')

# dv = DictVectorizer()
with open("ridge_reg.bin", 'rb') as f:
    dv, model = pickle.load(f)


# Function to prepare features for prediction
def prepare_features(ride):
    features = {}
    features['PU_DO'] = '%s_%s' % (ride['PULocationID'], ride['DOLocationID'])
    features['trip_distance'] = ride['trip_distance']
    return features


# Function to predict the fare
def predict(features):
    X = dv.transform(features)
    y_pred = model.predict(X)
    return y_pred[0]


@app.route('/predict', methods=['POST'])
def predict_endpoint():
    ride = request.get_json()
    features = prepare_features(ride)
    pred = predict(features)
    res = {
        'duration': pred
    }
    return jsonify(res)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)
