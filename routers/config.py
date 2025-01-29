import os
from flask import Blueprint, jsonify

config_bp = Blueprint('config', __name__)

@config_bp.route('/config', methods=['GET'])
def get_config():
    """
    GET /api/config → renvoie la config pour l'affichage
    (produits disponibles, etc.)
    """
    liste_produits = []
    file_path = os.path.join(os.path.dirname(__file__), '..', 'liste_produit_groupe.txt')
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line_clean = line.strip()
                if line_clean:
                    liste_produits.append(line_clean)
    except Exception as e:
        return jsonify({"error": f"Impossible de lire la liste des produits: {e}"}), 500

    # Tu peux ajouter d'autres infos de config :
    config_data = {
        "produitsDisponibles": liste_produits,
        "variations": ["S+1", "S+2", "S+3"]
    }

    return jsonify(config_data), 200
