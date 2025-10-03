import os
import logging
import numpy as np
from flask import Flask, render_template, request, jsonify
import joblib
from flasgger import Swagger

# ======================
# Configuration
# ======================
MODEL_DIR = "models"
XGB_PATH = os.path.join(MODEL_DIR, "xgb_final_model.joblib")
ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder_kepoi.joblib")   # for kepoi_name
TARGET_ENCODER_PATH = os.path.join(MODEL_DIR, "target_encoder.joblib")  # for target label
os.makedirs(MODEL_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("model_api")

app = Flask(__name__)

# Feature order used in training
FEATURE_ORDER = [
    "koi_score",
    "koi_model_snr",
    "koi_max_mult_ev",
    "koi_count",
    "koi_prad",
    "koi_smet_err2",
    "koi_prad_err1",
    "kepoi_name",
    "koi_dicco_msky",
    "koi_dicco_msky_err"
]

# ======================
# Swagger Configuration
# ======================
swagger_config = {
    "headers": [],
    "specs": [
        {
            "endpoint": 'apispec',
            "route": '/apispec.json',
            "rule_filter": lambda rule: True,
            "model_filter": lambda tag: True,
        }
    ],
    "static_url_path": "/flasgger_static",
    "swagger_ui": True,
    "specs_route": "/docs"
}

swagger_template = {
    "swagger": "2.0",
    "info": {
        "title": "ML Model API",
        "description": "API for predicting with XGBoost model",
        "version": "1.0.0"
    },
    "schemes": ["http", "https"],
}

swagger = Swagger(app, config=swagger_config, template=swagger_template)

# ======================
# Prediction Endpoint
# ======================
@app.route("/predict", methods=["POST"])
def predict():

    """
    Predict using XGBoost model
    ---
    consumes:
      - application/json
    parameters:
      - in: body
        name: input
        required: true
        schema:
          type: object
          properties:
            koi_score: {type: number, example: 0.85}
            koi_model_snr: {type: number, example: 15.2}
            koi_max_mult_ev: {type: number, example: 3.5}
            koi_count: {type: number, example: 2}
            koi_prad: {type: number, example: 1.8}
            koi_smet_err2: {type: number, example: 0.15}
            koi_prad_err1: {type: number, example: 0.05}
            kepoi_name: {type: string, example: "K00001.01"}
            koi_dicco_msky: {type: number, example: 0.95}
            koi_dicco_msky_err: {type: number, example: 0.02}
    responses:
      200:
        description: Prediction result
      400:
        description: Invalid input
      500:
        description: Prediction error
    """
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No input data provided"}), 400

        # Load encoders and model
        le_kepoi = joblib.load(ENCODER_PATH)
        le_target = joblib.load(TARGET_ENCODER_PATH)
        model = joblib.load(XGB_PATH)

        # Check required features
        missing = [f for f in FEATURE_ORDER if f not in data]
        if missing:
            return jsonify({"error": f"Missing features: {missing}"}), 400

        # Encode kepoi_name
        features = dict(data)
        features["kepoi_name"] = le_kepoi.transform([features["kepoi_name"]])[0]

        # Build input array
        X = np.array([features[f] for f in FEATURE_ORDER]).reshape(1, -1)

        # Prediction
        pred = model.predict(X)[0]
        proba = model.predict_proba(X)[0]

        # Decode prediction
        pred_label = le_target.inverse_transform([pred])[0]

        # Decode all probabilities using LabelEncoder
        prob_dict = {}
        for i, p in enumerate(proba):
            label_name = le_target.inverse_transform([i])[0]
            prob_dict[label_name] = float(p)

        return jsonify({
            "prediction": int(pred),
            "prediction_label": pred_label,
            "confidence": float(np.max(proba)),
            "probabilities": prob_dict
        })

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 500


# ======================
# Run the API
# ======================
if __name__ == "__main__":
    logger.info("Starting Model API on http://127.0.0.1:5000")
    logger.info("Swagger docs available at http://127.0.0.1:5000/docs")
    app.run(debug=True, host="0.0.0.0", port=5000)