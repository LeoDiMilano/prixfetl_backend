import os
import sys
import warnings
import psycopg2
import numpy as np
import pandas as pd
from datetime import date, timedelta, datetime
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import re

from sklearn.model_selection import KFold  # ou autre type de CV
from sklearn.feature_selection import RFE
from sklearn.base import BaseEstimator, RegressorMixin

# On importe la classe qui prépare et charge les données
from data_preprocessing import ApplePriceDataLoader


warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
warnings.filterwarnings(
    "ignore",
    message="pandas only supports SQLAlchemy connectable",
    category=UserWarning
)

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
        # On stockera ici les modèles entraînés par (produit_groupe, horizon)
        self.models = {}
        
        # Pour gagner du temps : on stocke ici le DataFrame complet préparé une seule fois
        self.df_complet = None

    def get_connection(self):
        """
        Retourne une connexion PostgreSQL en utilisant psycopg2.
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

        # Saisons converties en entiers
        df_complet['SAISON'] = df_complet['SAISON'].astype(int)

        return df_complet

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

        # Debug : exporter en Excel pour contrôle
        os.makedirs("tests", exist_ok=True)
        with pd.ExcelWriter("tests/4_train_data_filtered.xlsx") as writer:
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
        with pd.ExcelWriter("tests/2_train_data.xlsx") as writer:
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
        with pd.ExcelWriter("tests/3_test_data.xlsx") as writer:
            X_test.to_excel(writer, sheet_name="X_test", index=False)
            y_test_s1.to_excel(writer, sheet_name="y_test_s1", index=False)
            y_test_s2.to_excel(writer, sheet_name="y_test_s2", index=False)
            y_test_s3.to_excel(writer, sheet_name="y_test_s3", index=False)

        # (A) Sélection RFE sur S+1
        print(f"[{price_col}] - Lancement RFE pour S+1...")
        rfecv_s1, selected_features_s1 = self.train_model_s1_with_rfe(X_train, y_train_s1)

        # Modèle final S+1 (déjà entraîné par RFE)
        best_model_s1 = rfecv_s1.estimator_
        self.models[(price_col, 'S+1')] = best_model_s1

        # Évaluation sur test
        if not X_test.empty:
            pred_s1 = best_model_s1.predict(X_test[selected_features_s1])
            rmse_s1 = np.sqrt(np.mean((pred_s1 - y_test_s1) ** 2))
            print(f"[{price_col}] - RMSE test S+1 après RFE : {rmse_s1:.4f}")
        else:
            pred_s1 = []
            print(f"[{price_col}] - Pas de test set pour S+1")

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

        # Exporter les prédictions vs réelles
        sanitized_price_col = re.sub(r'\W+', '_', price_col)
        df_test_copy = df_test.copy()
        df_test_copy['PRED_S1'] = pred_s1
        df_test_copy['PRED_S2'] = pred_s2
        df_test_copy['PRED_S3'] = pred_s3
        
        with pd.ExcelWriter(f"tests/5_predictions_{sanitized_price_col}.xlsx") as writer:
            df_test_copy[[
                'DATE_INTERROGATION', price_col, target_s1, target_s2, target_s3
            ]].to_excel(writer, sheet_name="Valeurs réelles", index=False)
            
            df_test_copy[[
                'DATE_INTERROGATION', 'PRED_S1', 'PRED_S2', 'PRED_S3'
            ]].to_excel(writer, sheet_name="Prédictions", index=False)

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

        # On mémorise aussi la liste finale des features (S+1) pour le scoring
        self.models[(price_col, 'selected_features')] = selected_features_s1

    def insert_forecasts_into_db(self, date_interrogation, produit_groupe,
                                 prix_s1, prix_s2, prix_s3,
                                 var_s1, var_s2, var_s3):
        """
        Insère la prévision dans la table previsions_prix
        sans mise à jour si la ligne existe déjà (ON CONFLICT DO NOTHING).
        
        - 'var_s1', 'var_s2', 'var_s3' correspondent aux classes de variation
        """
        # Supprime le préfixe 'PRIX ' si présent (selon la convention que vous avez adoptée)
        if produit_groupe.startswith("PRIX "):
            produit_groupe = produit_groupe.replace("PRIX ", "")
        # Convertir les valeurs numpy.float32 en types Python natifs
        prix_s1 = float(prix_s1) if prix_s1 is not None else None
        prix_s2 = float(prix_s2) if prix_s2 is not None else None
        prix_s3 = float(prix_s3) if prix_s3 is not None else None

        query = """
        INSERT INTO public.previsions_prix(
            "DATE_INTERROGATION", 
            "PRODUIT_GROUPE", 
            "PRIX_PREV_S1", 
            "PRIX_PREV_S2", 
            "PRIV_PREV_S3",
            "VAR_PRIX_PREV_S1",
            "VAR_PRIX_PREV_S2",
            "VAR_PRIX_PREV_S3"
            -- on laisse PRIX_REEL_Sx à NULL
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT ("DATE_INTERROGATION", "PRODUIT_GROUPE")
        DO NOTHING;
        """
        conn = None
        try:
            conn = self.get_connection()
            cur = conn.cursor()
            cur.execute(query, (
                date_interrogation,
                produit_groupe,
                prix_s1 if prix_s1 is not None else None,
                prix_s2 if prix_s2 is not None else None,
                prix_s3 if prix_s3 is not None else None,
                var_s1,
                var_s2,
                var_s3
            ))
            conn.commit()
            cur.close()
        except Exception as e:
            print("Erreur lors de l'insertion dans la table previsions_prix:", e)
            if conn:
                conn.rollback()
        finally:
            if conn:
                conn.close()

    @staticmethod
    def variation_class(diff):
        """
        Retourne la classe de variation en fonction du diff = prix_prevu - prix_actuel
           - diff > +0.03 => 2
           - diff > +0.01 => 1
           - diff entre -0.01 et +0.01 => 0
           - diff < -0.03 => -2
           - sinon => -1
        """
        if diff > 0.03:
            return 2
        elif diff > 0.01:
            return 1
        elif diff >= -0.01 and diff <= 0.01:
            return 0
        elif diff < -0.03:
            return -2
        else:
            return -1

    def predict_for_one_date(self, df, date_lundi, price_col):
        """
        Fait la prédiction S+1, S+2, S+3 pour un produit ('price_col') à une date donnée.
        On va chercher dans df la ligne correspondante à 'date_lundi'.
        """
        selected_features = self.models.get((price_col, 'selected_features'), [])
        model_s1 = self.models.get((price_col, 'S+1'), None)
        model_s2 = self.models.get((price_col, 'S+2'), None)
        model_s3 = self.models.get((price_col, 'S+3'), None)

        if None in [model_s1, model_s2, model_s3]:
            print(f"[ERREUR] Modèles non entraînés pour {price_col}!")
            return None, None, None

        # Filtrer dans df la ligne du lundi
        row_df = df.loc[df['DATE_INTERROGATION'] == pd.to_datetime(date_lundi)]

        if row_df.empty:
            # Si on n'a pas de ligne exacte pour ce lundi dans df, on ne peut pas prédire
            # (ou alors on prend la dernière connue, à adapter selon vos besoins)
            return None, None, None
        
        # On prend X = row_df[selected_features]
        X = row_df[selected_features]
        prix_s1 = model_s1.predict(X)[0]
        prix_s2 = model_s2.predict(X)[0]
        prix_s3 = model_s3.predict(X)[0]

        return prix_s1, prix_s2, prix_s3

    def run_full_training(self):
        """
        Entraîne tous les modèles (pour toutes les colonnes PRIX ... ou tous les produits).
        """
        print("=== Démarrage de la préparation du dataset ===")
        self.df_complet = self.prepare_dataset()
        print("=== Dataset prêt ===")

        # Vous pouvez filtrer les colonnes PRIX si vous souhaitez entraîner sur toutes,
        # ou bien laisser tel quel si vous allez appeler .train_models_for_column() pour chaque produit.
        # Ici, on ne fait rien de plus. On entraîne selon votre besoin dans 'train_models_for_column'.
        pass

    def train_for_list_of_products(self, list_of_products):
        """
        Entraîne le modèle pour chaque produit de la liste, 
        c'est-à-dire pour chaque colonne 'PRIX <produit>'.
        
        Exemple si list_of_products = [
           'EXP POMME GOLDEN FRANCE 170/220G CAT.I PLATEAU 1RG',
           ...
        ]
        alors la colonne dans df_complet s'appelle 'PRIX EXP POMME GOLDEN FRANCE 170/220G CAT.I PLATEAU 1RG'
        (selon votre convention).
        """
        if self.df_complet is None:
            self.df_complet = self.prepare_dataset()

        for prod in list_of_products:
            col_name = f"PRIX {prod}"  # si c'est ainsi que la colonne est nommée dans df
            if col_name not in self.df_complet.columns:
                print(f"[WARN] La colonne '{col_name}' n'existe pas dans df_complet.")
                continue

            print(f"\n=== Entraînement du modèle pour la colonne : {col_name} ===")
            self.train_models_for_column(self.df_complet, col_name)

    def get_last_date_in_db(self):
        """
        Cherche la dernière date présente dans la table previsions_prix.
        Si aucune ligne n'existe, on renvoie la date du Lundi de la Semaine 1 de 2024 (ex: 2024-01-01).
        """
        sql = """SELECT MAX("DATE_INTERROGATION") FROM public.previsions_prix;"""
        conn = None
        last_date = None
        try:
            conn = self.get_connection()
            cur = conn.cursor()
            cur.execute(sql)
            row = cur.fetchone()
            cur.close()
            if row and row[0] is not None:
                last_date = row[0]  # type: datetime.date
        except Exception as e:
            print("Erreur lors de la récupération de la dernière date:", e)
            if conn:
                conn.rollback()
        finally:
            if conn:
                conn.close()

        if last_date is None:
            # Si aucune date en base, on prend le lundi de la Semaine 1 de 2024
            # 2024-01-01 est un lundi, donc parfait
            # Sinon, vous pouvez calculer le lundi de la semaine 1 autrement.
            return date(2024, 1, 1)
        else:
            return last_date

    @staticmethod
    def generate_mondays_between(start_date, end_date):
        """
        Génère tous les lundis entre start_date (inclus) et end_date (inclus),
        sous forme de date (datetime.date).
        """
        # on s'assure que start_date <= end_date
        if start_date > end_date:
            return

        current = start_date
        # On doit avancer jusqu'au lundi suivant si start_date n'est pas un lundi
        # Pour simplifier, on considère qu'on commence déjà un lundi dans ce cas d'usage
        # ou on force : 
        while current.weekday() != 0:  # 0 = lundi
            current += timedelta(days=1)

        while current <= end_date:
            yield current
            current += timedelta(days=7)

    def fill_previsions_for_missing_mondays(self, list_of_products):
        """
        1) Récupère la dernière date en base,
        2) Pour chaque produit de list_of_products,
        3) Pour chaque lundi entre last_date et aujourd'hui (inclus),
        4) Fait la prévision, calcule la variation et insère dans la table.
        """
        if self.df_complet is None:
            self.df_complet = self.prepare_dataset()

        # 1) Dernière date
        last_dt = self.get_last_date_in_db()
        today = date.today()

        print(f"Date la plus récente en base : {last_dt}. On va jusqu'au {today}.")

        # 2) Pour chaque produit
        for prod in list_of_products:
            col_name = f"PRIX {prod}"
            if col_name not in self.df_complet.columns:
                print(f"[WARN] La colonne '{col_name}' n'existe pas dans df_complet. Skipping.")
                continue

            print(f"\n--- Insertion prévisions pour : {col_name} ---")
            # 3) Pour chaque lundi
            for monday in self.generate_mondays_between(last_dt, today):
                # Vérifie si déjà présent en base ? 
                # => on pourrait le faire, mais le ON CONFLICT DO NOTHING gère déjà le doublon.

                # 4) Fait la prévision
                prix_s1, prix_s2, prix_s3 = self.predict_for_one_date(self.df_complet, monday, col_name)
                if prix_s1 is None:
                    # Pas de row => skip
                    continue

                # Prix "actuel" (celui du Monday dans df) pour calculer la variation
                row_monday = self.df_complet.loc[self.df_complet['DATE_INTERROGATION'] == pd.to_datetime(monday)]
                if not row_monday.empty:
                    current_price = row_monday[col_name].values[0]
                else:
                    # Pas de prix actuel => on ne peut pas calculer la variation => skip
                    continue

                diff_s1 = prix_s1 - current_price
                diff_s2 = prix_s2 - current_price
                diff_s3 = prix_s3 - current_price

                var_s1 = self.variation_class(diff_s1)
                var_s2 = self.variation_class(diff_s2)
                var_s3 = self.variation_class(diff_s3)

                # 5) Insert
                self.insert_forecasts_into_db(
                    date_interrogation=monday,
                    produit_groupe=prod,  # Dans la table, on stocke "PRODUIT_GROUPE" sans "PRIX "
                    prix_s1=prix_s1,
                    prix_s2=prix_s2,
                    prix_s3=prix_s3,
                    var_s1=var_s1,
                    var_s2=var_s2,
                    var_s3=var_s3
                )
                print(f"[OK] {monday} - {prod} : "
                      f"S1={prix_s1:.3f}, S2={prix_s2:.3f}, S3={prix_s3:.3f} "
                      f"(class var: {var_s1}, {var_s2}, {var_s3})")


if __name__ == "__main__":
    # Exemple de configuration de la base
    db_config = {
        "host": "prixfetl_postgres",
        "port": 5432,
        "database": "IAFetL",
        "user": "prixfetl",
        "password": "Leumces123"
    }
    
    # 1) Instancie le trainer
    trainer = PriceTrainer(db_config)

    # 2) Entraîne globalement sur tout l'historique (Saisons 2018->2023) pour chaque produit.
    #    - On prépare d'abord le dataset
    trainer.run_full_training()

    # 3) Lit la liste de produits depuis le fichier 'liste_produit_groupe.txt'
    #    Exemple de contenu du fichier:
    #       EXP POMME GOLDEN FRANCE 170/220G CAT.I PLATEAU 1RG
    #       EXP POMME GOLDEN FRANCE 201/270G CAT.I PLATEAU 1RG
    #       EXP POMME GRANNY FRANCE 100-113 CAT.I BUSHEL
    list_of_products = []
    with open("liste_produit_groupe.txt", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                list_of_products.append(line)

    # 4) Entraîne un modèle XGBoost pour chaque produit de la liste
    trainer.train_for_list_of_products(list_of_products)

    # 5) Maintenant, alimente la table previsions_prix pour tous les lundis manquants.
    trainer.fill_previsions_for_missing_mondays(list_of_products)

    print("\n=== Fin de l'exécution ===")
