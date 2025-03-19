import os
import sys
import subprocess
from datetime import date
from dotenv import load_dotenv
import logging
from log_utils import setup_logging

# Ajouter le chemin des services au path Python
services_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "services")
if services_path not in sys.path:
    sys.path.append(services_path)


# Import des modules nécessaires depuis services
from services.training import PriceTrainer
from services.inference import ModelInference
from services.data_preprocessing import ApplePriceDataLoader

logger = logging.getLogger(__name__)

routers_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "routers")
if routers_path not in sys.path:
    sys.path.append(routers_path)

from flask import Flask, request, jsonify
from routers.history import history_bp
from routers.config import config_bp
from routers.predict import predict_bp
from flask_cors import CORS
from routers.decorators import require_token

load_dotenv()  # charge les variables depuis .env

# Nouvelle fonction pour appliquer le décorateur aux vues.
def apply_token_auth():
    def before_request():
        # Récupérer la vue
        view = request.endpoint
        if view not in history_bp.view_functions and view not in predict_bp.view_functions:
           return
        token = request.headers.get('Authorization')
        if not token or token != f'Bearer {API_TOKEN_SECRET}':
            return jsonify({'error': 'Unauthorized'}), 401
            
    return before_request


def create_app():
    app = Flask(__name__)

    # 2) Configurer CORS pour autoriser ton/tes domaines
    #    Tu peux bien sûr en rajouter ou mettre "*" si tu préfères
    CORS(app, resources={
        r"/api/*": {
            "origins": [
                "https://intellibooster.com",
                "https://www.intellibooster.com",
                "https://www.intelliboost.fr",
                "https://api.intellibooster.com"
            ]
        }
    })

    # 3) Appliquer le décorateur "require_token" avant chaque requête
    #    sur certains blueprints (ex: history, predict).
    #    Si tu veux protéger config_bp aussi, ajoute-le. Sinon, laisse-le libre.
    app.before_request(apply_token_auth())

    # 4) Enregistrer les blueprints
    app.register_blueprint(history_bp, url_prefix='/api')
    app.register_blueprint(config_bp, url_prefix='/api')
    app.register_blueprint(predict_bp, url_prefix='/api')

    return app




db_config = {
    "host": os.getenv("POSTGRES_HOST"),
    "port": int(os.getenv("POSTGRES_PORT")),
    "database": os.getenv("POSTGRES_DB"),
    "user": os.getenv("POSTGRES_USER"),
    "password": os.getenv("POSTGRES_PASSWORD")
}

DATA_RAW_DIR = os.getenv("DATA_RAW_DIR")
DATA_OUTPUT_DIR = os.getenv("DATA_OUTPUT_DIR")

def main():
    # Étape 1) Mise à jour des données brutes (ETL / ingestion)
    logger.info("=== 1) Mise à jour des données météo ===")
    subprocess.run(["python", "/app/services/1_maj_meteo.py"], check=True)

    logger.info("=== 2) Mise à jour des cotations RNM ===")
    subprocess.run(["python", "/app/services/2_maj_COTATIONS_RNM_JOURNALIERES.py"], check=True)

    logger.info("=== 3) Mise à jour des données AGRIDATA (trade) ===")
    subprocess.run(["python", "/app/services/3_maj_AGRIDATA_EU_APPLES_trade_data.py"], check=True)

    logger.info("=== 4) Mise à jour des données de Vacances - sauté ===")
    subprocess.run(["python", "/app/services/4_maj_VACANCES.py"], check=True)

    logger.info("=== 4b) Mise à jour des données de Pologne ===")
    subprocess.run(["python", "/app/services/5_maj_COTATIONS_WIESCIROLNICZE.py"], check=True)    

    # Étape 2) Préprocessing (construction du DataFrame global)
    logger.info("=== 5) Exécution du script data_preprocessing ===")
    subprocess.run(["python", "/app/services/data_preprocessing.py"], check=True)

     # Lecture de la liste de produits depuis le fichier 'liste_produit_groupe.txt'
    list_of_products = []
    if os.path.exists("liste_produit_groupe.txt"):
        with open("liste_produit_groupe.txt", "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    list_of_products.append(line)
    else:
        logger.warning("Fichier liste_produit_groupe.txt introuvable. On utilise un exemple.")
        list_of_products = [
            "EXP POMME GOLDEN FRANCE 170/220G CAT.I PLATEAU 1RG",
            "EXP POMME GRANNY FRANCE 100-113 CAT.I BUSHEL"
        ]

    # (A) Entraînement
    logger.info("=== 6) Entraînement des modèles ===")
    trainer = PriceTrainer(db_config)
    trainer.run_full_training(list_of_products, date(2024, 8, 5))

    logger.info("=== 7) Inférence ===")
    # (B) Inférence
    inference_engine = ModelInference(db_config, trainer)

    # 1) Insertion des nouvelles prévisions (pour les semaines à venir)
    logger.info("=== 8) Insertion des prévisions pour les lundis manquants ===")
    inference_engine.fill_previsions_for_missing_mondays(list_of_products)

    # 2) Mise à jour des prix réels (pour les semaines passées)
    logger.info("=== 9) Mise à jour des prix réels (S+1, S+2, S+3) ===")
    inference_engine.update_real_prices()

    logger.info("\n=== Fin du script principal ===")


if __name__ == "__main__":
    # Configurer le logging
    setup_logging()
    
    if '--cron' in sys.argv:
        main()  # Exécute le pipeline de mise à jour
    else:
        my_app = create_app()
