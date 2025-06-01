import os
import pickle
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template
import onnxruntime as ort
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# --- Configuration and Paths ---
MODELS_DIR = "models"
MODEL_NAME_ONNX = "anemia_binary_model.onnx"
ONNX_MODEL_PATH = os.path.join(MODELS_DIR, MODEL_NAME_ONNX)
ENCODERS_DIR = os.path.join(MODELS_DIR, 'encoders')

# Expected features for the model
EXPECTED_COLUMNS = ['HGB', 'HCT', 'RBC', 'RDW', 'MCH', 'MCHC', 'MCV', 'SD', 'TSD']

# Categorical columns within EXPECTED_COLUMNS (empty based on your logs)
CATEGORICAL_COLUMNS = []

# Prediction labels mapping
PREDICTION_LABELS = {
    0: "Tidak Terkena Anemia",
    1: "Terkena Anemia"
}

# --- Load Model and Encoders on App Startup ---
ort_session = None
scaler = None
label_encoders_dict = {}

try:
    ort_session = ort.InferenceSession(ONNX_MODEL_PATH)
    print(f"[*] ONNX model '{ONNX_MODEL_PATH}' loaded successfully.")

    with open(os.path.join(ENCODERS_DIR, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    print(f"[*] Scaler '{os.path.join(ENCODERS_DIR, 'scaler.pkl')}' loaded successfully.")

    label_encoders_file = os.path.join(ENCODERS_DIR, 'label_encoders.pkl')
    if os.path.exists(label_encoders_file) and CATEGORICAL_COLUMNS:
        with open(label_encoders_file, 'rb') as f:
            label_encoders_dict = pickle.load(f)
        print(f"[*] Label Encoders '{label_encoders_file}' loaded successfully.")
    else:
        print("[*] No Label Encoders loaded (file not found or no categorical columns defined).")

    print("[*] All models and encoders are ready!")

except Exception as e:
    print(f"[!] FATAL ERROR: Failed to load model or encoders: {e}")
    print("[!] Ensure you have run the training pipeline and all files are correctly saved.")
    exit()

# --- Helper Functions ---

def safe_label_encode(le, value):
    value_str = str(value)
    if value_str in le.classes_:
        return le.transform([value_str])[0]
    else:
        print(f"Warning: Unknown categorical value '{value}'. Using 0 as default.")
        return 0

# --- API Endpoints ---

@app.route('/')
def home():
    return render_template('index.html', expected_features=EXPECTED_COLUMNS, categorical_features=CATEGORICAL_COLUMNS)

@app.route('/predict', methods=['POST'])
def predict_anemia():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()

    # 1. Validate Input
    missing_keys = [key for key in EXPECTED_COLUMNS if key not in data]
    if missing_keys:
        return jsonify({
            "error": "Missing features in JSON request.",
            "missing_features": missing_keys,
            "expected_features": EXPECTED_COLUMNS
        }), 400

    # 2. Convert to DataFrame and Preprocess
    try:
        df_input = pd.DataFrame([data], columns=EXPECTED_COLUMNS)

        for col in CATEGORICAL_COLUMNS:
            if col in df_input.columns and col in label_encoders_dict:
                df_input[col] = df_input[col].apply(lambda x: safe_label_encode(label_encoders_dict[col], x))
            elif col in EXPECTED_COLUMNS:
                df_input[col] = 0

        numerical_columns = [col for col in EXPECTED_COLUMNS if col not in CATEGORICAL_COLUMNS]

        for col in numerical_columns:
            df_input[col] = pd.to_numeric(df_input[col], errors='coerce')
            if scaler is not None:
                try:
                    col_idx_in_scaler = list(scaler.feature_names_in_).index(col)
                    df_input[col].fillna(scaler.mean_[col_idx_in_scaler], inplace=True)
                except (ValueError, AttributeError):
                    df_input[col].fillna(0, inplace=True)
            else:
                df_input[col].fillna(0, inplace=True)

        scaled_data = scaler.transform(df_input[numerical_columns])
        df_input[numerical_columns] = scaled_data

        processed_input = df_input[EXPECTED_COLUMNS].to_numpy().astype(np.float32)

    except Exception as e:
        return jsonify({"error": f"Error during data preprocessing: {str(e)}"}), 400

    # 3. Perform Inference with ONNX Runtime
    try:
        input_name = ort_session.get_inputs()[0].name
        output_name = ort_session.get_outputs()[0].name
        
        outputs = ort_session.run([output_name], {input_name: processed_input})
        y_pred_proba = outputs[0][0]

        # 4. Decode Prediction Result
        y_pred_label_idx = int(np.argmax(y_pred_proba))
        predicted_label_str = PREDICTION_LABELS.get(y_pred_label_idx, "Unknown Class")

        probabilities_dict = {
            PREDICTION_LABELS.get(i, f"Class {i}"): float(prob)
            for i, prob in enumerate(y_pred_proba)
        }

        # 5. Send JSON Response
        return jsonify({
            "status": "success",
            "prediction_label": predicted_label_str,
            "probability_no_anemia": probabilities_dict.get(PREDICTION_LABELS.get(0), 0.0),
            "probability_yes_anemia": probabilities_dict.get(PREDICTION_LABELS.get(1), 0.0),
            "all_probabilities": probabilities_dict,
            "input_data_received": data
        })

    except Exception as e:
        return jsonify({"error": f"Error during prediction: {str(e)}"}), 500

# --- Run Flask App ---
if __name__ == '__main__':
    print("\n[+] Flask app is ready.")
    print(f"    Access UI at: http://127.0.0.1:5000/")
    print(f"    API endpoint at: http://127.0.0.1:5000/predict (POST request)")
    app.run(debug=True, host='0.0.0.0', port=5000)