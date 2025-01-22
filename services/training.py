import os
import sys
import warnings
import psycopg2
import numpy as np
import pandas as pd
from datetime import date
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import re

from sklearn.feature_selection import RFECV
from sklearn.model_selection import KFold  # ou autre type de CV (StratifiedKFold, etc.)
from sklearn.feature_selection import RFE

# On importe la classe qui prépare et charge les données
from data_preprocessing import ApplePriceDataLoader

warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

from sklearn.base import BaseEstimator, RegressorMixin

class SklearnCompatibleXGBRegressor(XGBRegressor, BaseEstimator, RegressorMixin):
    def __sklearn_tags__(self):
        return {
            "estimator_type": "regressor",
            "requires_y": True,
        }


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
            'subsample': 1,
            'colsample_bytree': 1,
            'objective': 'reg:squarederror',
            'nthread': 4,
            'scale_pos_weight': 1,
            'seed': 0
        }

        # On stockera ici les modèles entraînés par (produit, horizon)
        # Exemple: self.models[(col, 'S1')] = XGBRegressor(...)
        self.models = {}

    def prepare_dataset(self):
        """
        Prépare le DataFrame complet (2018->2024...) via la classe ApplePriceDataLoader
        et toutes les étapes de data_preprocessing (déjà implémentées).
        
        Retourne :
            df_complet (pd.DataFrame) : le DataFrame complet prêt pour l'entraînement.
        """
        # Charger les données brutes et les enrichir
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
        #print(df_complet.columns)  # Vérifie si ANNEE est encore présente

        #print("Dataset complet préparé :", df_complet.head())
        return df_complet

    def get_connection(self):
        """
        Retourne une connexion PostgreSQL en utilisant psycopg2.
        """
        return psycopg2.connect(**self.db_config)

    def insert_forecasts_into_db(self, date_interrogation, produit_groupe, prix_s1, prix_s2, prix_s3):
        # Supprime le préfixe 'PRIX ' si présent dans le produit_groupe
        produit_groupe = produit_groupe.replace("PRIX ", "") if produit_groupe.startswith("PRIX ") else produit_groupe

        """
        Insère la prévision dans la table previsions_prix
        sans mise à jour si la ligne existe déjà (DO NOTHING).
        """
        query = """
        INSERT INTO public.previsions_prix(
            "DATE_INTERROGATION", 
            "PRODUIT_GROUPE", 
            "PRIX_PREV_S1", 
            "PRIX_PREV_S2", 
            "PRIV_PREV_S3"
        )
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT ("DATE_INTERROGATION", "PRODUIT_GROUPE")
        DO NOTHING;
        """
        conn = None
        try:
            conn = self.get_connection()
            cur = conn.cursor()
            cur.execute(
                query, 
                (date_interrogation, 
                produit_groupe,
                float(prix_s1) if prix_s1 is not None else None,
                float(prix_s2) if prix_s2 is not None else None,
                float(prix_s3) if prix_s3 is not None else None)
            )
            conn.commit()
            cur.close()
        except Exception as e:
            print("Erreur lors de l'insertion dans la table previsions_prix:", e)
            if conn:
                conn.rollback()
        finally:
            if conn:
                conn.close()


        def prepare_dataset(self):
            """
            Prépare le DataFrame complet (2018->2024...) via la classe ApplePriceDataLoader
            et toutes les étapes de data_preprocessing (déjà implémentées).
            
            Retourne :
                df_complet (pd.DataFrame) : le DataFrame complet prêt pour l'entraînement.
            """
            # 1. Charger les données brutes et les enrichir
            loader = ApplePriceDataLoader(self.db_config)
            df_sql = loader.load_prices_dataframe()
            df_complet = loader.complete_with_trade_data(df_sql)
            df_complet = loader.complete_with_forecasts(df_complet)
            df_complet = loader.complete_with_meteo(df_complet)
            df_complet = loader.handle_missing_values(df_complet)
            df_complet = loader.add_variations(df_complet)
            df_complet = loader.add_vacations(df_complet)

            # Exemple : Saisons en int
            df_complet['SAISON'] = df_complet['SAISON'].astype(int)

            return df_complet

    def generate_shifted_targets(self, df, price_col):
        """
        Crée 3 colonnes cibles (S+1, S+2, S+3) par décalage temporel.
        On part du principe que df est indexé chronologiquement ou par un ordre croissant de semaines.

        Retourne :
            df (pd.DataFrame) : Le DataFrame d'origine avec 3 nouvelles colonnes
                                [f"{price_col}_S+1", f"{price_col}_S+2", f"{price_col}_S+3"].
        """
        # Assurons-nous que le DataFrame est trié par date (ou au moins dans l'ordre chrono)
        df = df.sort_values(by=["ANNEE", "SEMAINE", "DATE_INTERROGATION"]).reset_index(drop=True)

        # On crée trois colonnes cibles
        df[f"{price_col}_S+1"] = df[price_col].shift(-1)  # la semaine suivante
        df[f"{price_col}_S+2"] = df[price_col].shift(-2)  # dans 2 semaines
        df[f"{price_col}_S+3"] = df[price_col].shift(-3)  # dans 3 semaines

        return df
# ------------------------------------------------------------------
# NOUVELLE METHODE : RFE / RFECV SUR S+1
# ------------------------------------------------------------------
    from sklearn.feature_selection import RFE

    def train_model_s1_with_rfe(self, X_train, y_train_s1):
        # Filtrer les données
        mask_s1 = y_train_s1.notnull()
        X_train_filtered = X_train[mask_s1]
        y_train_filtered = y_train_s1[mask_s1]

        print(f"Taille des données après filtrage : {X_train_filtered.shape}, {y_train_filtered.shape}")
        print(f"Nombre de features avant optimisation : {X_train_filtered.shape[1]}")

        # Initialisation de l'estimateur
        xgb_estimator = SklearnCompatibleXGBRegressor(**self.model_params)

        # Configuration de RFE (sans validation croisée)
        rfe = RFE(
            estimator=xgb_estimator,
            n_features_to_select=20,  # Nombre final de features à garder
            step=0.5  # On supprime 50 % des features à chaque étape
        )

        # Entraîner RFE
        rfe.fit(X_train_filtered, y_train_filtered)

        # Récupérer les features sélectionnées
        selected_features = X_train_filtered.columns[rfe.support_]
        print(f"Features sélectionnées : {selected_features.tolist()}")
        print(f"Nombre de features après optimisation : {len(selected_features)}")

        return rfe, selected_features.tolist()



    def train_models_for_column(self, df, price_col):
        """
        Version modifiée : On applique la sélection par RFECV pour S+1,
        puis on entraîne S+2, S+3 normalement (ou éventuellement tu peux
        faire aussi une RFECV pour S+2, S+3 si tu veux).
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

        X_test = df_test[X_cols].copy()
        y_test_s1 = df_test[target_s1].copy()
        y_test_s2 = df_test[target_s2].copy()
        y_test_s3 = df_test[target_s3].copy()


        # ----------------------------------------------------------------
        # (A) Sélection RFE pour S+1
        # ----------------------------------------------------------------
        print(f"[{price_col}] - Lancement RFECV pour S+1...")
        rfecv_s1, selected_features_s1 = self.train_model_s1_with_rfe(X_train, y_train_s1)

        # On entraîne le "meilleur" modèle final
        best_model_s1 = rfecv_s1.estimator_
        # Ou, dans les versions récentes, rfecv_s1.estimator_ est déjà entraîné
        # Sinon, on peut faire : best_model_s1.fit(X_train[selected_features_s1], y_train_s1)
        
        # On évalue sur le test
        if not X_test.empty:
            pred_s1 = best_model_s1.predict(X_test[selected_features_s1])
            rmse_s1 = np.sqrt(np.mean((pred_s1 - y_test_s1) ** 2))
            print(f"[{price_col}] - RMSE test S+1 après RFE : {rmse_s1:.4f}")
        else:
            rmse_s1 = None
            print(f"[{price_col}] - Pas de test set pour S+1")

        # On stocke ce modèle
        self.models[(price_col, 'S+1')] = best_model_s1

        # ----------------------------------------------------------------
        # (B) Entraîner S+2 et S+3 : 
        #     Tu peux réutiliser les mêmes features 
        #     ou relancer un RFE séparé (au choix).
        # ----------------------------------------------------------------
        # Exemple : on fait un entraînement "naïf" sur le sous-ensemble de features
        # sélectionnées pour S+1. (Ça te fait gagner du temps.)
        # Si tu préfères, tu peux coder train_model_s2_with_rfe, etc.

        model_s2 = XGBRegressor(**self.model_params)
        mask_s2 = y_train_s2.notnull()
        model_s2.fit(X_train[selected_features_s1][mask_s2], y_train_s2[mask_s2])
        self.models[(price_col, 'S+2')] = model_s2

        model_s3 = XGBRegressor(**self.model_params)
        mask_s3 = y_train_s3.notnull()
        model_s3.fit(X_train[selected_features_s1][mask_s3], y_train_s3[mask_s3])
        self.models[(price_col, 'S+3')] = model_s3

        # Evaluation basique
        if not X_test.empty:
            pred_s2 = model_s2.predict(X_test[selected_features_s1])
            pred_s3 = model_s3.predict(X_test[selected_features_s1])
            rmse_s2 = np.sqrt(np.mean((pred_s2 - y_test_s2) ** 2))
            rmse_s3 = np.sqrt(np.mean((pred_s3 - y_test_s3) ** 2))

            print(f"[{price_col}] - RMSE test S+2 : {rmse_s2:.4f}")
            print(f"[{price_col}] - RMSE test S+3 : {rmse_s3:.4f}")

        # Tu peux aussi exporter un Excel avec la liste finalisée de features 
        # et leur importance, si tu le souhaites.

    def generate_future_forecast(self, df, price_col):
        """
        Utilise le dernier point connu dans df pour prévoir S+1, S+2, S+3
        à l'aide des modèles XGBoost entraînés.

        Retourne (s1, s2, s3) = (float, float, float)
        """
        # On prend la dernière ligne disponible (ou la ligne correspondant à la date du jour, etc.)
        df_sorted = df.sort_values(by=["ANNEE", "SEMAINE", "DATE_INTERROGATION"]).reset_index(drop=True)
        last_row = df_sorted.iloc[[-1]]  # DataFrame d'une seule ligne

        # Retirer les mêmes colonnes inutiles qu’au training
        cols_to_remove = [
            'DATE_INTERROGATION', 'ANNEE', 'SEMAINE', 'SAISON', 'MOIS_SAISON', 'SEMAINE_SAISON',
            f"{price_col}_S+1", f"{price_col}_S+2", f"{price_col}_S+3"
        ]
        X_cols = [c for c in df.columns if c not in cols_to_remove]

        # Récupérer les modèles
        model_s1 = self.models.get((price_col, 'S+1'), None)
        model_s2 = self.models.get((price_col, 'S+2'), None)
        model_s3 = self.models.get((price_col, 'S+3'), None)

        if (model_s1 is None) or (model_s2 is None) or (model_s3 is None):
            print(f"Erreur : un ou plusieurs modèles manquants pour {price_col}.")
            return None, None, None

        X_future = last_row[X_cols]
        # Prévisions
        s1 = model_s1.predict(X_future)[0]
        s2 = model_s2.predict(X_future)[0]
        s3 = model_s3.predict(X_future)[0]

        return (s1, s2, s3)

    def run_training_and_insertion(self):
        """
        Méthode principale :
          1. Prépare le dataset
          2. Pour chaque colonne de prix "PRIX EXP POMME...", entraîne les modèles
          3. Génère la prévision pour S+1, S+2, S+3
          4. Insère dans la table 'previsions_prix'
        """
        # 1) Construction du dataset complet
        df_complet = self.prepare_dataset()

        # 2) Identifie les colonnes de prix "PRIX EXP POMME..."
        all_price_cols = [c for c in df_complet.columns if c.startswith("PRIX EXP POMME")]

        if not all_price_cols:
            print("Aucune colonne ne correspond à 'PRIX EXP POMME'. Vérifiez vos données.")
            return

        # 3) Entraînement des modèles pour chaque colonne
        for col in all_price_cols:
            print(f"\n=== Entraînement pour la colonne : {col} ===")
            self.train_models_for_column(df_complet, col)

        # 4) Générer la date d'interrogation = date du Lundi (par exemple aujourd'hui si on est lundi)
        #    ou un sys.argv paramétrable, etc.
        date_interrogation = date.today()  # ou tout autre logique de date

        # Ensure the tests directory exists
        os.makedirs("tests", exist_ok=True)
        # Exporter les prévisions futures
        future_forecasts = []
        for col in all_price_cols:
            s1, s2, s3 = self.generate_future_forecast(df_complet, col)
            if s1 is None:
                # Problème de modèle manquant, on skip
                continue

            future_forecasts.append({
                "produit_groupe": col,
                "date_interrogation": date_interrogation,
                "prix_s1": s1,
                "prix_s2": s2,
                "prix_s3": s3
            })

            # On insère
            self.insert_forecasts_into_db(
                date_interrogation=date_interrogation,
                produit_groupe=col,  # On peut mettre le nom exact de la colonne comme "PRODUIT_GROUPE"
                prix_s1=s1,
                prix_s2=s2,
                prix_s3=s3
            )
            print(f"Insertion OK pour {col} (S+1={s1:.2f}, S+2={s2:.2f}, S+3={s3:.2f})")

        # Convertir les prévisions futures en DataFrame et exporter
        df_forecasts = pd.DataFrame(future_forecasts)
        df_forecasts.to_csv(os.path.join("tests", "future_forecasts.csv"), index=False)

        print("\n=== FIN DE L'ENTRAINEMENT ET DE L'INSERTION DES PREVISIONS ===")


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings(
        "ignore",
        message="pandas only supports SQLAlchemy connectable",
        category=UserWarning
    )

    db_config = {
        "host": "prixfetl_postgres",
        "port": 5432,
        "database": "IAFetL",
        "user": "prixfetl",
        "password": "Leumces123"
    }

    trainer = PriceTrainer(db_config)

    # Pour tester un seul produit avant de faire un run global
    df_complet = trainer.prepare_dataset()
    price_col = "PRIX EXP POMME GALA FRANCE 170/220G CAT.I PLATEAU 1RG"

    print(f"\n** Test RFE sur {price_col} **")
    trainer.train_models_for_column(df_complet, price_col)