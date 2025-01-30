import os
from flask import Blueprint, jsonify
from db import fetch_all
from .decorators import require_token # Import require_token depuis routers.decorators

predict_bp = Blueprint('predict', __name__)

@predict_bp.route('/predict', methods=['GET'])
@require_token
def get_predict():
    """
    GET /api/predict
    
    Pour chaque produit listé dans liste_produit_groupe.txt, 
    renvoie la dernière date disponible et les variations prévues.
    Colonnes renvoyées :
      - "PRODUIT_GROUPE"
      - "DATE_INTERROGATION"
      - "PRIX_REEL_S"
      - "VAR_PRIX_PREV_S1"
      - "VAR_PRIX_PREV_S2"
      - "VAR_PRIX_PREV_S3"
    """

    # 1) Lecture du fichier liste_produit_groupe.txt
    file_path = os.path.join(os.path.dirname(__file__), '..', 'liste_produit_groupe.txt')
    products = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                products.append(line.strip())
    except Exception as e:
        return jsonify({"message": f"Erreur lors de la lecture du fichier: {e}"}), 500

    # 2) Récupération des données depuis la base de données
    query = """
        SELECT 
            "DATE_INTERROGATION",
            "PRODUIT_GROUPE",
            "PRIX_REEL_S",
            "VAR_PRIX_PREV_S1",
            "VAR_PRIX_PREV_S2",
            "VAR_PRIX_PREV_S3"
        FROM previsions_prix
        WHERE "PRODUIT_GROUPE" IN %s
          AND "DATE_INTERROGATION" = (SELECT MAX("DATE_INTERROGATION") FROM previsions_prix)
    """
    try:
        data = fetch_all(query, (tuple(products),))
    except Exception as e:
        return jsonify({"message": f"Erreur lors de la récupération des données: {e}"}), 500

    return jsonify(data)
