import pickle


with open("Deployment\web-service\model.pkl", 'rb') as f:
    dv, model = pickle.load(f)


def predict(features):
    X = dv.trasnform(features)
    y_pred = model.predict(X)
    return y_pred
