# price_trainer.py

import os
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
import re
import matplotlib.pyplot as plt
import sys 
from xgboost import XGBRegressor
from sklearn.feature_selection import RFE
from datetime import date
import psycopg2
# On importe la classe qui prépare et charge les données
from data_preprocessing import ApplePriceDataLoader

from dotenv import load_dotenv

load_dotenv()  # charge .env
DATA_RAW_DIR = os.getenv("DATA_RAW_DIR")
DATA_OUTPUT_DIR = os.getenv("DATA_OUTPUT_DIR")
DATA_OUTPUT_PROCESSED_DIR = os.getenv("DATA_OUTPUT_PROCESSED_DIR")

warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

from xgboost import XGBClassifier

class PriceTrainer:
    def __init__(self, db_config):
        """
        db_config : dictionnaire des paramètres de connexion à la base PostgreSQL
        Exemple :
            db_config = {
                "host": "prixfetl_postgres",
                "port": 5432,
                "database": "IAFetL",
                "user": "prixfetl",
                "password": "Leumces123"
            }
        """
        self.db_config = db_config

        # Paramètres du modèle XGB (communs à chaque horizon de prévision)
        self.model_params = {
            "n_estimators": 350000,         # Nombre d'arbres
            "learning_rate": 0.05,       # Taux d'apprentissage (plus bas = meilleur, mais plus lent)
            "max_depth": 6,              # Profondeur maximale des arbres (5-8 est souvent optimal)
            "subsample": 0.8,            # Pourcentage d'échantillons utilisés par arbre (évite l'overfitting)
            "colsample_bytree": 1,     # Pourcentage de features utilisées par arbre
            "gamma": 0,                # Prune les branches peu utiles (évite l'overfitting)
            "reg_alpha": 0.1,            # Régularisation L1 (sparse features)
            "reg_lambda": 1,             # Régularisation L2 (évite l'overfitting)
            "objective": "binary:logistic",  # Classification binaire (ajuste pour multi-classes)
            "eval_metric": "logloss",    # Métrique de classification (logloss = cross-entropy loss)
            "use_label_encoder": False   # Évite un warning inutile
        }
        # Dictionnaire où l'on stocke les modèles entraînés : self.models[(col_name, 'S+1')] = ...
        self.models = {}

        # On mémorise la liste de features sélectionnées pour chaque colonne (pour l’inférence)
        self.selected_features_for_column = {}
        
        # Pour éviter de recharger à chaque fois : un seul DataFrame complet
        self.df_complet = None

    def get_connection(self):
        """
        Établit une connexion PostgreSQL.

        Returns:
            psycopg2.Connection : Connexion à la base de données.
        """
        return psycopg2.connect(**self.db_config)


    def prepare_dataset(self):
        """
        Prépare le DataFrame complet (2018->...) via la classe ApplePriceDataLoader
        et toutes les étapes de data_preprocessing (déjà implémentées).
        
        Retourne :
            df_complet (pd.DataFrame) : le DataFrame complet prêt pour l'entraînement.
        """
        loader = ApplePriceDataLoader(self.db_config)
        df_sql = loader.load_prices_dataframe()
        print("Étape 1 : Données brutes ajoutées")
        
        df_complet = loader.complete_with_trade_data(df_sql)
        print("Étape 2 : Données commerciales ajoutées")

        df_complet = loader.complete_with_forecasts(df_complet)
        print("Étape 3 : Prévisions annuelles ajoutées")

        df_complet = loader.complete_with_meteo(df_complet)
        print("Étape 4 : Données météo ajoutées")

        df_complet = loader.handle_missing_values(df_complet)
        print("Étape 5 : Gestion des valeurs manquantes réalisées")

        df_complet = loader.add_variations(df_complet)
        print("Étape 6 : Variations ajoutées")

        df_complet = loader.add_vacations(df_complet)
        print("Étape 7 : Indicateurs de vacances ajoutés")

        # Saisons converties en int
        df_complet['SAISON'] = df_complet['SAISON'].astype(int)

        return df_complet

    def run_full_training(self, list_of_products, split_date=date(2024, 8, 5)):
        """
        Méthode globale qui :
          1) Prépare le dataset complet (2018->...)
          2) Lance l’entraînement pour chaque produit (colonne) de la liste `list_of_products`.
             (On suppose que dans le DataFrame, ces colonnes se nomment 'PRIX <produit>')
          3) Utilise `split_date` pour séparer les ensembles d'entraînement et de test.
        
        Args:
            list_of_products (list): Liste des produits pour lesquels entraîner les modèles.
            split_date (datetime.date): Date de séparation entre le train set et le test set.
        """
        print("=== Démarrage de la préparation du dataset ===")
        self.df_complet = self.prepare_dataset()
        print("=== Dataset prêt ===")

        print("\n=== Début de l'entraînement des modèles pour la liste de produits ===")
        for prod_groupe in list_of_products:
            col_name = f"PRIX {prod_groupe}"
            if col_name not in self.df_complet.columns:
                print(f"[AVERTISSEMENT] La colonne '{col_name}' est introuvable dans df_complet. On ignore.")
                continue
            self.train_models_for_column(self.df_complet, col_name, split_date)
        print("\n=== Fin de l'entraînement des modèles ===")

        # Export des features importances
        self.export_feature_importances(list_of_products)

    def generate_shifted_targets(self, df, price_col):
        """
        Crée 3 colonnes cibles (S+1, S+2, S+3) par décalage temporel sur 'price_col'.
        """
        df = df.sort_values(by=["ANNEE", "SEMAINE", "DATE_INTERROGATION"]).reset_index(drop=True)
        df[f"{price_col}_S+1"] = df[price_col].shift(-1)
        df[f"{price_col}_S+2"] = df[price_col].shift(-2)
        df[f"{price_col}_S+3"] = df[price_col].shift(-3)
        
        # Créer les classes de variation
        df[f"{price_col}_S+1_class"] = df[f"{price_col}_S+1"].pct_change().apply(self.variation_class)
        df[f"{price_col}_S+2_class"] = df[f"{price_col}_S+2"].pct_change().apply(self.variation_class)
        df[f"{price_col}_S+3_class"] = df[f"{price_col}_S+3"].pct_change().apply(self.variation_class)
        
        # Mapper les classes de [-2, -1, 0, 1, 2] à [0, 1, 2, 3, 4]
        df[f"{price_col}_S+1_class"] = df[f"{price_col}_S+1_class"].map({-2: 0, -1: 1, 0: 2, 1: 3, 2: 4})
        df[f"{price_col}_S+2_class"] = df[f"{price_col}_S+2_class"].map({-2: 0, -1: 1, 0: 2, 1: 3, 2: 4})
        df[f"{price_col}_S+3_class"] = df[f"{price_col}_S+3_class"].map({-2: 0, -1: 1, 0: 2, 1: 3, 2: 4})
        
        return df

    def train_model_s1_with_rfe(self, X_train, y_train_s1):
        """
        Lance un RFE pour sélectionner 20 features pour l'horizon S+1.
        """
        mask_s1 = y_train_s1.notnull()
        X_train_filtered = X_train[mask_s1]
        y_train_filtered = y_train_s1[mask_s1]

        os.makedirs("tests", exist_ok=True)
        with pd.ExcelWriter(os.path.join(DATA_OUTPUT_DIR, "4_train_data_filtered.xlsx")) as writer:
            X_train_filtered.T.to_excel(writer, sheet_name="X_train_filtered", header=False)
            y_train_filtered.T.to_excel(writer, sheet_name="y_train_filtered", header=False)

        print(f"Taille des données après filtrage : {X_train_filtered.shape}, {y_train_filtered.shape}")
        print(f"Nombre de features avant optimisation : {X_train_filtered.shape[1]}")

        xgb_estimator = XGBClassifier(**self.model_params)

        rfe = RFE(
            estimator=xgb_estimator,
            n_features_to_select=20,
            step=0.2
        )
        rfe.fit(X_train_filtered, y_train_filtered)

        selected_features = X_train_filtered.columns[rfe.support_]
        print(f"Nombre de features après optimisation : {len(selected_features)}")

        return rfe, selected_features.tolist()

    def train_models_for_column(self, df, price_col, split_date):
        """
        Entraîne (ou réentraîne) le modèle XGBoost pour un 'price_col' donné.
        Gère les horizons S+1, S+2, S+3.
        
        Args:
            df (pd.DataFrame): Le DataFrame contenant les données complètes.
            price_col (str): La colonne de prix pour laquelle entraîner le modèle.
            split_date (datetime.date): Date de séparation entre le train set et le test set.
        """
        # Convertir split_date en datetime64[ns]
        split_date = pd.to_datetime(split_date)

        # 1) Ajout des colonnes cibles
        df = self.generate_shifted_targets(df, price_col)

        # 2) Split train/test
        df_train = df[df['DATE_INTERROGATION'] < split_date].copy()
        df_test = df[df['DATE_INTERROGATION'] >= split_date].copy()

        target_s1 = f"{price_col}_S+1_class"
        target_s2 = f"{price_col}_S+2_class"
        target_s3 = f"{price_col}_S+3_class"

        cols_to_remove = [
            'DATE_INTERROGATION', 'ANNEE', 'SEMAINE',
            target_s1, target_s2, target_s3,
            f"{price_col}_S+1", f"{price_col}_S+2", f"{price_col}_S+3"
        ]
        X_cols = [c for c in df.columns if c not in cols_to_remove]

        X_train = df_train[X_cols].copy()
        y_train_s1 = df_train[target_s1].copy()
        y_train_s2 = df_train[target_s2].copy()
        y_train_s3 = df_train[target_s3].copy()

        # Debug : export train
        os.makedirs("tests", exist_ok=True)
        with pd.ExcelWriter(os.path.join(DATA_OUTPUT_DIR, "2_train_data.xlsx")) as writer:
            X_train.to_excel(writer, sheet_name="X_train", index=False)
            y_train_s1.to_excel(writer, sheet_name="y_train_s1", index=False)
            y_train_s2.to_excel(writer, sheet_name="y_train_s2", index=False)
            y_train_s3.to_excel(writer, sheet_name="y_train_s3", index=False)

        # Test sets
        X_test = df_test[X_cols].copy()
        y_test_s1 = df_test[target_s1].copy()
        y_test_s2 = df_test[target_s2].copy()
        y_test_s3 = df_test[target_s3].copy()

        # Debug : export test
        with pd.ExcelWriter(os.path.join(DATA_OUTPUT_DIR, "3_test_data.xlsx")) as writer:
            X_test.to_excel(writer, sheet_name="X_test", index=False)
            y_test_s1.to_excel(writer, sheet_name="y_test_s1", index=False)
            y_test_s2.to_excel(writer, sheet_name="y_test_s2", index=False)
            y_test_s3.to_excel(writer, sheet_name="y_test_s3", index=False)

        # (A) Sélection RFE sur S+1
        print(f"\n[{price_col}] - Lancement RFE pour S+1...")
        rfecv_s1, selected_features_s1 = self.train_model_s1_with_rfe(X_train, y_train_s1)

        # Modèle final S+1 (le RFE a déjà entraîné son estimator_)
        best_model_s1 = rfecv_s1.estimator_
        self.models[(price_col, 'S+1')] = best_model_s1

        # Évaluation sur test
        if not X_test.empty:
            pred_s1 = best_model_s1.predict(X_test[selected_features_s1])
            rmse_s1 = np.sqrt(np.mean((pred_s1 - y_test_s1) ** 2))
            print(f"[{price_col}] - RMSE test S+1 après RFE : {rmse_s1:.4f}")
        else:
            pred_s1 = []

        # (B) S+2
        model_s2 = XGBClassifier(**self.model_params)
        mask_s2 = y_train_s2.notnull()
        model_s2.fit(X_train[selected_features_s1][mask_s2], y_train_s2[mask_s2])
        self.models[(price_col, 'S+2')] = model_s2
        
        if not X_test.empty:
            pred_s2 = model_s2.predict(X_test[selected_features_s1])
            rmse_s2 = np.sqrt(np.mean((pred_s2 - y_test_s2) ** 2))
            print(f"[{price_col}] - RMSE test S+2 : {rmse_s2:.4f}")
        else:
            pred_s2 = []

        # (C) S+3
        model_s3 = XGBClassifier(**self.model_params)
        mask_s3 = y_train_s3.notnull()
        model_s3.fit(X_train[selected_features_s1][mask_s3], y_train_s3[mask_s3])
        self.models[(price_col, 'S+3')] = model_s3

        if not X_test.empty:
            pred_s3 = model_s3.predict(X_test[selected_features_s1])
            rmse_s3 = np.sqrt(np.mean((pred_s3 - y_test_s3) ** 2))
            print(f"[{price_col}] - RMSE test S+3 : {rmse_s3:.4f}")
        else:
            pred_s3 = []

        # Sauvegarde de la liste de features
        self.selected_features_for_column[price_col] = selected_features_s1

        sanitized_price_col = re.sub(r'\W+', '_', price_col)
        # Mapper les prédictions de [0, 1, 2, 3, 4] à [-2, -1, 0, 1, 2]
        df_test_copy = df_test.copy()
        df_test_copy['PRED_S1'] = [self.map_prediction(p) for p in pred_s1]
        df_test_copy['PRED_S2'] = [self.map_prediction(p) for p in pred_s2]
        df_test_copy['PRED_S3'] = [self.map_prediction(p) for p in pred_s3]
        
        with pd.ExcelWriter(os.path.join(DATA_OUTPUT_DIR, f"5_predictions_{sanitized_price_col}.xlsx")) as writer:
            df_test_copy[['DATE_INTERROGATION', price_col, target_s1, target_s2, target_s3]] \
                .to_excel(writer, sheet_name="Valeurs réelles", index=False)
            df_test_copy[['DATE_INTERROGATION', 'PRED_S1', 'PRED_S2', 'PRED_S3']] \
                .to_excel(writer, sheet_name="Prédictions", index=False)

        # Graphiques
        if not X_test.empty:
            fig, ax = plt.subplots(3, 1, figsize=(12, 12))
            
            ax[0].plot(df_test_copy['DATE_INTERROGATION'], df_test_copy[price_col], label='Valeurs réelles')
            ax[0].plot(df_test_copy['DATE_INTERROGATION'], df_test_copy['PRED_S1'], label='Prédictions S+1')
            ax[0].set_title(f"Prédictions pour {price_col} - S+1")
            ax[0].legend()

            ax[1].plot(df_test_copy['DATE_INTERROGATION'], df_test_copy[price_col], label='Valeurs réelles')
            ax[1].plot(df_test_copy['DATE_INTERROGATION'], df_test_copy['PRED_S2'], label='Prédictions S+2')
            ax[1].set_title(f"Prédictions pour {price_col} - S+2")
            ax[1].legend()

            ax[2].plot(df_test_copy['DATE_INTERROGATION'], df_test_copy[price_col], label='Valeurs réelles')
            ax[2].plot(df_test_copy['DATE_INTERROGATION'], df_test_copy['PRED_S3'], label='Prédictions S+3')
            ax[2].set_title(f"Prédictions pour {price_col} - S+3")
            ax[2].legend()

            plt.tight_layout()
            plt.savefig(f"tests/6_predictions_{sanitized_price_col}.png")
            plt.close()
    def map_prediction(self, prediction):
        """
        Mappe les prédictions de [0, 1, 2, 3, 4] à [-2, -1, 0, 1, 2].
        """
        mapping = {0: -2, 1: -1, 2: 0, 3: 1, 4: 2}
        return mapping.get(prediction, 0)

    def export_feature_importances(self, list_of_products):
        """
        Exporte les importances des features pour chaque modèle dans un fichier Excel.
        """
        feature_importances = {}

        for prod_groupe in list_of_products:
            col_name = f"PRIX {prod_groupe}"
            if (col_name, 'S+1') in self.models:
                model = self.models[(col_name, 'S+1')]
                booster = model.get_booster()
                importance = booster.get_score(importance_type='weight')
                sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
                feature_importances[prod_groupe] = sorted_importance

        # Création du fichier Excel
        output_dir = DATA_OUTPUT_PROCESSED_DIR
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "feature_importances.xlsx")

        with pd.ExcelWriter(output_path) as writer:
            for product, importances in feature_importances.items():
                # Nettoyer le nom du produit pour qu'il soit valide en tant que titre de feuille Excel
                clean_product_name = re.sub(r'[\\/*?:\[\]]', '_', product)
                # Mapper les indices des features aux noms des colonnes d'origine
                feature_names = [self.selected_features_for_column[f"PRIX {product}"][int(f[1:])] for f, _ in importances]
                df_importances = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': [imp for _, imp in importances]
                })
                df_importances.to_excel(writer, sheet_name=clean_product_name, index=False)

        print(f"Feature importances exported to {output_path}")

    def variation_class(self, variation):
        """
        Classe une variation de prix en une des 5 classes : -2, -1, 0, 1, 2.
        """
        if variation <= -0.03:
            return -2
        elif variation <= -0.01:
            return -1
        elif variation <= 0.01:
            return 0
        elif variation <= 0.03:
            return 1
        else:
            return 2