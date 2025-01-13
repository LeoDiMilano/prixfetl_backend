from flask import Blueprint, request, jsonify
from services.inference import ModelInference
import pandas as pd

predict_bp = Blueprint('predict_bp', __name__)

# Crée une instance unique ou dynamique
model_inference = ModelInference(model_path='model_storage/xgb_model.joblib')

@predict_bp.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint pour obtenir la prédiction de la variation de prix
    """
    input_data = request.get_json()  # ex: { "features": { "col1": ..., "col2": ...} }
    df_features = pd.DataFrame([input_data["features"]])  # transformer en DataFrame

    prediction = model_inference.predict_variation(df_features)
    # post-processing => transformer la classe 0,1,2,3,4 en '--','-','S','+','++'
    labels = ['--', '-', 'S', '+', '++']
    predicted_label = labels[prediction[0]]  # ex. 2 => 'S'

    return jsonify({"prediction": predicted_label})
