from flask import Flask, request, jsonify
import pandas as pd
from predict import load_model, predict

app = Flask(__name__)

# Load model
MODEL_PATH = "model.bin"
dv, model = load_model(MODEL_PATH)


@app.route('/predict', methods=['POST'])
def predict_endpoint():
    """Predict ride duration based on input JSON."""
    try:
        data = request.get_json()
        # Expecting JSON with 'PULocationID', 'DOLocationID', 'trip_distance'
        df = pd.DataFrame([data])
        df['PULocationID'] = df['PULocationID'].astype(str)
        df['DOLocationID'] = df['DOLocationID'].astype(str)
        df_pred = predict(df, dv, model)
        prediction = df_pred['predicted_duration'].iloc[0]
        return jsonify({'predicted_duration': float(prediction)})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
