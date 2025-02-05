import os
import re
import logging
import psycopg2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from sklearn.metrics import confusion_matrix, classification_report

logger = logging.getLogger(__name__)

class ConfusionMatrixGenerator:
    def __init__(self):
        """
        Le constructeur lit immédiatement les variables d'environnement et 
        prépare la config DB, le chemin de sortie pour les PNG, et la liste de produits.
        """
        # Charger les variables d'environnement
        load_dotenv()
        # 1) Charger la config DB via les variables d'environnement
        self.db_config = {
            "host": os.getenv("POSTGRES_HOST"),
            "port": int(os.getenv("POSTGRES_PORT", 5432)),
            "database": os.getenv("POSTGRES_DB"),
            "user": os.getenv("POSTGRES_USER"),
            "password": os.getenv("POSTGRES_PASSWORD"),
        }

        # 2) Chemin où on va enregistrer les images (par défaut: /app/data/processed)
        self.output_dir = os.getenv("DATA_OUTPUT_PROCESSED_DIR", "/app/data/processed")

        # 3) Charger la liste des produits depuis /app/liste_produit_groupe.txt
        #    ou DATA_RAW_DIR si besoin, mais tu as précisé /app/liste_produit_groupe.txt
        self.product_list_path = "/app/liste_produit_groupe.txt"
        self.products = self._load_products()

    def _load_products(self):
        """
        Charge la liste de produits depuis /app/liste_produit_groupe.txt
        (un produit par ligne).
        """
        if not os.path.exists(self.product_list_path):
            logger.warning(
                f"Fichier liste_produit_groupe.txt introuvable: {self.product_list_path}. "
                "Aucun produit ne sera traité."
            )
            return []

        with open(self.product_list_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        
        logger.info(f"{len(lines)} produits chargés depuis {self.product_list_path}.")
        return lines

    def get_connection(self):
        """
        Crée une connexion psycopg2 via db_config.
        """
        try:
            conn = psycopg2.connect(**self.db_config)
            return conn
        except psycopg2.Error as e:
            logger.error(f"Erreur de connexion PostgreSQL: {e}")
            raise

    def _clean_product_name(self, product_name):
        """
        Remplace les caractères spéciaux par des underscores
        pour créer un nom de fichier sûr.
        """
        return re.sub(r"[^\w]+", "_", product_name)

    def _plot_and_save_confusions(self, cm_s1, cm_s2, cm_s3, product, season, report_s1, report_s2, report_s3):
        """
        Construit et sauvegarde un PNG avec 3 sous-graphes (S+1, S+2, S+3) et les rapports de classification.
        """
        fig, axes = plt.subplots(3, 2, figsize=(20, 15))
        all_labels = [-2, -1, 0, 1, 2]
        horizons = ["S+1", "S+2", "S+3"]
        cms = [cm_s1, cm_s2, cm_s3]
        reports = [report_s1, report_s2, report_s3]

        for i, (cm, report, horizon) in enumerate(zip(cms, reports, horizons)):
            ax = axes[i, 0]
            im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
            ax.set_title(f"Matrice de confusion {horizon}")
            ax.set_xlabel("Prévision")
            ax.set_ylabel("Réel")

            ax.set_xticks(range(len(all_labels)))
            ax.set_yticks(range(len(all_labels)))
            ax.set_xticklabels(all_labels)
            ax.set_yticklabels(all_labels)

            # Ajouter le score numérique dans chaque case
            for (j, k), val in np.ndenumerate(cm):
                ax.text(k, j, int(val), ha="center", va="center",
                        color="white" if val > cm.max()/2. else "black")

            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            # Ajouter le rapport de classification
            ax_report = axes[i, 1]
            ax_report.axis('off')
            ax_report.text(0.01, 0.5, report, fontsize=12, verticalalignment='center', family='monospace')

        fig.suptitle(f"{product} (Saison {season})", fontsize=16)

        # Nom du fichier
        clean_name = self._clean_product_name(product)
        filename = f"{season}_{clean_name}.png"
        output_path = os.path.join(self.output_dir, filename)

        os.makedirs(self.output_dir, exist_ok=True)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close(fig)

        logger.info(f"PNG généré pour {product}: {output_path}")

    def generate_confusion_matrices(self, season):
        """
        1) Récupère toutes les données de la table previsions_prix pour la SAISON demandée.
        2) Filtre sur les produits qui sont dans la liste self.products.
        3) Calcule la matrice de confusion pour S+1, S+2, S+3, et enregistre dans un PNG.
        
        Retourne un dict: 
          {
            "produit1": {"S+1": cm_s1, "S+2": cm_s2, "S+3": cm_s3},
            ...
          }
        """
        query = """
        SELECT
            "SAISON",
            "DATE_INTERROGATION",
            "PRODUIT_GROUPE",
            "VAR_PRIX_PREV_S1", "VAR_PRIX_PREV_S2", "VAR_PRIX_PREV_S3",
            "VAR_PRIX_REEL_S1", "VAR_PRIX_REEL_S2", "VAR_PRIX_REEL_S3"
        FROM previsions_prix
        WHERE "SAISON" = %s
        AND "VAR_PRIX_PREV_S1" IS NOT NULL
        AND "VAR_PRIX_REEL_S1" IS NOT NULL
        ORDER BY "PRODUIT_GROUPE", "DATE_INTERROGATION";
        """

        conn = None
        df = None
        try:
            conn = self.get_connection()
            df = pd.read_sql(query, conn, params=(season,))
        except Exception as e:
            logger.error(f"Erreur lors de la requête sur previsions_prix: {e}")
            return {}
        finally:
            if conn:
                conn.close()

        if df is None or df.empty:
            logger.warning(f"Aucune donnée pour la saison {season}.")
            return {}

        # Filtrer uniquement les produits de la liste
        df = df[df["PRODUIT_GROUPE"].isin(self.products)]
        if df.empty:
            logger.warning(f"Aucune donnée pour les produits demandés en saison {season}.")
            return {}

        all_labels = [-2, -1, 0, 1, 2]
        confusion_results = {}

        # Pour chaque produit de la liste, on calcule la matrice
        for product in self.products:
            df_prod = df[df["PRODUIT_GROUPE"] == product]
            if df_prod.empty:
                logger.info(f"Aucune ligne en base pour le produit {product}, on saute.")
                continue

            # S+1
            y_pred_s1 = df_prod["VAR_PRIX_PREV_S1"].dropna()
            y_true_s1 = df_prod["VAR_PRIX_REEL_S1"].dropna()
            common_idx_s1 = y_pred_s1.index.intersection(y_true_s1.index)
            cm_s1 = confusion_matrix(
                y_true_s1.loc[common_idx_s1],
                y_pred_s1.loc[common_idx_s1],
                labels=all_labels
            )
            report_s1 = classification_report(
                y_true_s1.loc[common_idx_s1],
                y_pred_s1.loc[common_idx_s1],
                labels=all_labels,
                target_names=[str(label) for label in all_labels]
            )

            # S+2
            y_pred_s2 = df_prod["VAR_PRIX_PREV_S2"].dropna()
            y_true_s2 = df_prod["VAR_PRIX_REEL_S2"].dropna()
            common_idx_s2 = y_pred_s2.index.intersection(y_true_s2.index)
            cm_s2 = confusion_matrix(
                y_true_s2.loc[common_idx_s2],
                y_pred_s2.loc[common_idx_s2],
                labels=all_labels
            )
            report_s2 = classification_report(
                y_true_s2.loc[common_idx_s2],
                y_pred_s2.loc[common_idx_s2],
                labels=all_labels,
                target_names=[str(label) for label in all_labels]
            )

            # S+3
            y_pred_s3 = df_prod["VAR_PRIX_PREV_S3"].dropna()
            y_true_s3 = df_prod["VAR_PRIX_REEL_S3"].dropna()
            common_idx_s3 = y_pred_s3.index.intersection(y_true_s3.index)
            cm_s3 = confusion_matrix(
                y_true_s3.loc[common_idx_s3],
                y_pred_s3.loc[common_idx_s3],
                labels=all_labels
            )
            report_s3 = classification_report(
                y_true_s3.loc[common_idx_s3],
                y_pred_s3.loc[common_idx_s3],
                labels=all_labels,
                target_names=[str(label) for label in all_labels]
            )

            confusion_results[product] = {
                "S+1": cm_s1,
                "S+2": cm_s2,
                "S+3": cm_s3
            }

            # Génère et sauvegarde un PNG
            self._plot_and_save_confusions(cm_s1, cm_s2, cm_s3, product, season, report_s1, report_s2, report_s3)

        logger.info(f"Matrices de confusion générées pour la saison {season}, total produits: {len(confusion_results)}")
        return confusion_results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # On crée l'instance
    cmg = ConfusionMatrixGenerator()

    # On choisit la saison qu'on veut analyser, ex: 2024
    season = 2024

    # On génère
    results = cmg.generate_confusion_matrices(season=season)

    # results est un dict de {produit: {"S+1": cm_s1, "S+2": cm_s2, "S+3": cm_s3}}
    print("Résultats:", results)