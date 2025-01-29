import os
from flask import Blueprint, jsonify
from ..db import fetch_all
from .decorators import require_token

history_bp = Blueprint('history', __name__)

@history_bp.route('/history', methods=['GET'])
@require_token
def get_history():
     """
     GET /api/history
     Lit la liste des produits depuis liste_produit_groupe.txt,
     puis renvoie pour chacun (et pour chaque date disponible) 
     les variations prévues et réelles S+1, S+2, S+3.
     Résultat trié par PRODUIT_GROUPE (asc), puis DATE_INTERROGATION (desc).
     """
     # 1) Lecture du fichier liste_produit_groupe.txt
     file_path = os.path.join(os.path.dirname(__file__), '..', 'liste_produit_groupe.txt')
     products = []
     try:
         with open(file_path, 'r', encoding='utf-8') as f:
             for line in f:
                 p = line.strip()
                 if p:
                     products.append(p)
     except Exception as e:
         return jsonify({"error": "Impossible de lire la liste de produits", "details": str(e)}), 500

     # 2) Construction de la requête SQL
     placeholders = ", ".join(["%s"] * len(products))  # ex: "%s, %s, %s"
     query = f"""
         SELECT
             "PRODUIT_GROUPE",
             "DATE_INTERROGATION",
             "VAR_PRIX_PREV_S1",
             "VAR_PRIX_REEL_S1",
             "VAR_PRIX_PREV_S2",
             "VAR_PRIX_REEL_S2",
             "VAR_PRIX_PREV_S3",
             "VAR_PRIX_REEL_S3"
         FROM previsions_prix
         WHERE "PRODUIT_GROUPE" IN ({placeholders})
         ORDER BY 1,2 DESC
     """

     # 3) Exécution de la requête (fetch_all est défini dans db.py)
     rows = fetch_all(query, tuple(products))

     # 4) Retour des résultats sous forme JSON
     return jsonify(rows), 200
