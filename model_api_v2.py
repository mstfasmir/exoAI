import os
import logging
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import joblib
from flasgger import Swagger   # ✅ أضفنا المكتبة

# config
MODEL_DIR = "models"
XGB_PATH = os.path.join(MODEL_DIR, "xgb_final_model.joblib")
LOG_PATH = os.path.join(MODEL_DIR, "log_reg_model.joblib")
ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder_kepoi.joblib")   # for kepoi_name
TARGET_ENCODER_PATH = os.path.join(MODEL_DIR, "target_encoder.joblib") # for target label
os.makedirs(MODEL_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ml_service")

app = Flask(__name__)
swagger = Swagger(app)   # ✅ إضافة Swagger

# features we want
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

# -------------------------
# Training endpoint
# -------------------------
@app.route("/train", methods=["POST"])
def train():
    """
    Train Logistic Regression and XGBoost models
    ---
    consumes:
      - multipart/form-data
    parameters:
      - name: file
        in: formData
        type: file
        required: true
        description: CSV file containing training data
    responses:
      200:
        description: Training results with accuracy and F1 score
    """
    file = request.files["file"]
    df = pd.read_csv(file)

    # check all required features
    missing = [f for f in FEATURE_ORDER + ["label"] if f not in df.columns]
    if missing:
        return jsonify({"error": f"Missing columns: {missing}"}), 400

    # encode kepoi_name
    le_kepoi = LabelEncoder()
    df["kepoi_name"] = le_kepoi.fit_transform(df["kepoi_name"])
    joblib.dump(le_kepoi, ENCODER_PATH)

    # encode target labels
    le_target = LabelEncoder()
    y = le_target.fit_transform(df["label"])
    joblib.dump(le_target, TARGET_ENCODER_PATH)

    # split
    X = df[FEATURE_ORDER]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # train logistic regression
    log_model = LogisticRegression(max_iter=1000, random_state=42)
    log_model.fit(X_train, y_train)

    # train xgb
    xgb_model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        random_state=42
    )
    xgb_model.fit(X_train, y_train)

    # evaluate both
    results = {}
    for name, model in [("logistic", log_model), ("xgb", xgb_model)]:
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        acc_train = accuracy_score(y_train, y_train_pred)
        acc_test = accuracy_score(y_test, y_test_pred)
        f1_train = f1_score(y_train, y_train_pred, average="macro")
        f1_test = f1_score(y_test, y_test_pred, average="macro")

        results[name] = {
            "train_accuracy": acc_train,
            "test_accuracy": acc_test,
            "train_f1": f1_train,
            "test_f1": f1_test
        }

    # save models
    joblib.dump(log_model, LOG_PATH)
    joblib.dump(xgb_model, XGB_PATH)

    return jsonify({
        "message": "Both models trained and saved!",
        "results": results
    })

# -------------------------
# Prediction endpoint
# -------------------------
@app.route("/predict/<model_type>", methods=["POST"])
def predict(model_type):
    """
    Predict using trained Logistic or XGBoost model
    ---
    parameters:
      - name: model_type
        in: path
        type: string
        enum: ["xgb", "logistic"]
        required: true
        description: Which model to use for prediction
      - name: body
        in: body
        required: true
        schema:
          type: object
          properties:
            koi_score: {type: number}
            koi_model_snr: {type: number}
            koi_max_mult_ev: {type: number}
            koi_count: {type: number}
            koi_prad: {type: number}
            koi_smet_err2: {type: number}
            koi_prad_err1: {type: number}
            kepoi_name: {type: string}
            koi_dicco_msky: {type: number}
            koi_dicco_msky_err: {type: number}
    responses:
      200:
        description: Prediction result with probabilities
    """
    data = request.json
    le_kepoi = joblib.load(ENCODER_PATH)
    le_target = joblib.load(TARGET_ENCODER_PATH)

    # load model
    if model_type == "xgb":
        model = joblib.load(XGB_PATH)
    elif model_type == "logistic":
        model = joblib.load(LOG_PATH)
    else:
        return jsonify({"error": "Invalid model type. Use 'xgb' or 'logistic'"}), 400

    # check features
    missing = [f for f in FEATURE_ORDER if f not in data]
    if missing:
        return jsonify({"error": f"Missing features: {missing}"}), 400

    # encode kepoi_name
    features = dict(data)
    features["kepoi_name"] = le_kepoi.transform([features["kepoi_name"]])[0]

    # build input array
    X = np.array([features[f] for f in FEATURE_ORDER]).reshape(1, -1)

    # predict
    pred = model.predict(X)[0]
    proba = model.predict_proba(X).tolist()[0]

    # decode prediction to original label
    pred_label = le_target.inverse_transform([pred])[0]

    return jsonify({
        "model": model_type,
        "prediction": int(pred),
        "prediction_label": pred_label,
        "probabilities": proba
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)
