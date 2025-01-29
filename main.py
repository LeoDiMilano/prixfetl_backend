
import os
import subprocess
import sys
services_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "services")
if services_path not in sys.path:
    sys.path.append(services_path)
# Import des modules nécessaires depuis services
from services.training import PriceTrainer
from services.inference import ModelInference
from services.data_preprocessing import ApplePriceDataLoader
from dotenv import load_dotenv
from log_utils import setup_logging
import logging
logger = logging.getLogger(__name__)

load_dotenv()  # charge les variables depuis .env

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
    print("=== 1) Mise à jour des données météo ===")
    #subprocess.run(["python", "/app/services/1_maj_meteo.py"], check=True)

    print("=== 2) Mise à jour des cotations RNM ===")
    #subprocess.run(["python", "/app/services/2_maj_COTATIONS_RNM_JOURNALIERES.py"], check=True)

    print("=== 3) Mise à jour des données AGRIDATA (trade) ===")
    #subprocess.run(["python", "/app/services/3_maj_AGRIDATA_EU_APPLES_trade_data.py"], check=True)

    # Étape 2) Préprocessing (construction du DataFrame global)
    print("=== 4) Exécution du script data_preprocessing ===")
    
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
        print("[AVERTISSEMENT] Fichier liste_produit_groupe.txt introuvable. On utilise un exemple.")
        list_of_products = [
            "EXP POMME GOLDEN FRANCE 170/220G CAT.I PLATEAU 1RG",
            "EXP POMME GRANNY FRANCE 100-113 CAT.I BUSHEL"
        ]

    # (A) Entraînement
    print("=== 5) Entraînement des modèles ===")
    trainer = PriceTrainer(db_config)
    trainer.run_full_training(list_of_products)

    # (B) Inférence
    inference_engine = ModelInference(db_config, trainer)

    # 1) Insertion des nouvelles prévisions (pour les semaines à venir)
    print("=== 6) Insertion des prévisions pour les lundis manquants ===")
    inference_engine.fill_previsions_for_missing_mondays(list_of_products)

    # 2) Mise à jour des prix réels (pour les semaines passées)
    print("=== 7) Mise à jour des prix réels (S+1, S+2, S+3) ===")
    inference_engine.update_real_prices()

    print("\n=== Fin du script principal ===")


if __name__ == "__main__":
    main()
