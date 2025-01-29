# price_trainer.py

import os
import warnings
import numpy as np
import pandas as pd
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

warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)


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
            'n_estimators': 500,
            'learning_rate': 0.3,
            'max_depth': 6,
            'min_child_weight': 1,
            'gamma': 0,
            'subsample': 0.8,
            'colsample_bytree': 1,
            'objective': 'reg:squarederror',
            'nthread': 4,
            'scale_pos_weight': 2,
            'seed': 0
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

    def run_full_training(self, list_of_products):
        """
        Méthode globale qui :
          1) Prépare le dataset complet (2018->...)
          2) Lance l’entraînement pour chaque produit (colonne) de la liste `list_of_products`.
             (On suppose que dans le DataFrame, ces colonnes se nomment 'PRIX <produit>')
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
            self.train_models_for_column(self.df_complet, col_name)
        print("\n=== Fin de l'entraînement des modèles ===")

    def generate_shifted_targets(self, df, price_col):
        """
        Crée 3 colonnes cibles (S+1, S+2, S+3) par décalage temporel sur 'price_col'.
        """
        df = df.sort_values(by=["ANNEE", "SEMAINE", "DATE_INTERROGATION"]).reset_index(drop=True)
        df[f"{price_col}_S+1"] = df[price_col].shift(-1)
        df[f"{price_col}_S+2"] = df[price_col].shift(-2)
        df[f"{price_col}_S+3"] = df[price_col].shift(-3)
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

        xgb_estimator = XGBRegressor(**self.model_params)

        rfe = RFE(
            estimator=xgb_estimator,
            n_features_to_select=20,
            step=0.2
        )
        rfe.fit(X_train_filtered, y_train_filtered)

        selected_features = X_train_filtered.columns[rfe.support_]
        print(f"Nombre de features après optimisation : {len(selected_features)}")

        return rfe, selected_features.tolist()

    def train_models_for_column(self, df, price_col):
        """
        Entraîne (ou réentraîne) le modèle XGBoost pour un 'price_col' donné.
        Gère les horizons S+1, S+2, S+3.
        """
        # 1) Ajout des colonnes cibles
        df = self.generate_shifted_targets(df, price_col)

        # 2) Split train/test
        df_train = df[df['SAISON'].between(2018, 2023)].copy()
        df_test = df[df['SAISON'] >= 2024].copy()

        target_s1 = f"{price_col}_S+1"
        target_s2 = f"{price_col}_S+2"
        target_s3 = f"{price_col}_S+3"

        cols_to_remove = [
            'DATE_INTERROGATION', 'ANNEE', 'SEMAINE',
            target_s1, target_s2, target_s3
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
        model_s2 = XGBRegressor(**self.model_params)
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
        model_s3 = XGBRegressor(**self.model_params)
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

        # Exporter les prédictions vs réelles
        sanitized_price_col = re.sub(r'\W+', '_', price_col)
        df_test_copy = df_test.copy()
        df_test_copy['PRED_S1'] = pred_s1
        df_test_copy['PRED_S2'] = pred_s2
        df_test_copy['PRED_S3'] = pred_s3
        
        with pd.ExcelWriter(os.path.join(DATA_OUTPUT_DIR, f"5_predictions_{sanitized_price_col}.xlsx")) as writer:
            df_test_copy[['DATE_INTERROGATION', price_col, target_s1, target_s2, target_s3]] \
                .to_excel(writer, sheet_name="Valeurs réelles", index=False)
            df_test_copy[['DATE_INTERROGATION', 'PRED_S1', 'PRED_S2', 'PRED_S3']] \
                .to_excel(writer, sheet_name="Prédictions", index=False)

        # Graphiques
        if not X_test.empty:
            fig, ax = plt.subplots(3, 1, figsize=(12, 12))
            
            ax[0].plot(df_test_copy['DATE_INTERROGATION'], df_test_copy[price_col], label='Valeurs réelles')
            ax[0].plot(df_test_copy['DATE_INTERROGATION'], pred_s1, label='Prédictions S+1')
            ax[0].set_title(f"Prédictions pour {price_col} - S+1")
            ax[0].legend()

            ax[1].plot(df_test_copy['DATE_INTERROGATION'], df_test_copy[price_col], label='Valeurs réelles')
            ax[1].plot(df_test_copy['DATE_INTERROGATION'], pred_s2, label='Prédictions S+2')
            ax[1].set_title(f"Prédictions pour {price_col} - S+2")
            ax[1].legend()

            ax[2].plot(df_test_copy['DATE_INTERROGATION'], df_test_copy[price_col], label='Valeurs réelles')
            ax[2].plot(df_test_copy['DATE_INTERROGATION'], pred_s3, label='Prédictions S+3')
            ax[2].set_title(f"Prédictions pour {price_col} - S+3")
            ax[2].legend()

            plt.tight_layout()
            plt.savefig(f"tests/6_predictions_{sanitized_price_col}.png")
            plt.close()

    def variation_class(self, variation):
        """
        Calcule la classe de variation basée sur la différence entre les prix réels et prévus.

        Args:
            variation (float): La différence entre le prix réel et le prix prévu.

        Returns:
            int: La classe de variation, selon les règles suivantes :
                - variation > +0.03 => 2
                - variation > +0.01 => 1
                - -0.01 <= variation <= +0.01 => 0
                - variation < -0.03 => -2
                - sinon => -1
        """
        if variation is None:
            return None  # Si la variation est absente, on ne peut pas calculer de classe.

        if variation > 0.03:
            return 2
        elif variation > 0.01:
            return 1
        elif -0.01 <= variation <= 0.01:
            return 0
        elif variation < -0.03:
            return -2
        else:
            return -1
