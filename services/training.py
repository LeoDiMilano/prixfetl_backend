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

# On importe la classe qui prépare et charge les données
from data_preprocessing import ApplePriceDataLoader

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

    def train_models_for_column(self, df, price_col):
        """
        Pour une seule colonne de prix (ex: "PRIX EXP POMME ..."), 
        on crée 3 modèles XGBoost (pour S+1, S+2, S+3).
        
        - On sépare la période de train (SAISON 2018->2023) et la période de test (SAISON >= 2024)
        - On sélectionne un ensemble de features X (toute la table sauf colonnes cibles + info date)
        - On entraîne 3 modèles et on les stocke dans self.models[(price_col, 'S+1')] etc.
        
        Retourne : 
            None. (mais self.models est mis à jour)
        """
        # 1) Ajout des colonnes cibles par décalage
        df = self.generate_shifted_targets(df, price_col)

        # 2) Filtrage sur la période d'entraînement
        df_train = df[df['SAISON'].between(2018, 2023)].copy()
        df_test = df[df['SAISON'] >= 2024].copy()

        # 3) Construction des X et y
        #    On enlève les colonnes cibles futures et éventuellement les colonnes date/ID qu'on ne veut pas comme features
        target_s1 = f"{price_col}_S+1"
        target_s2 = f"{price_col}_S+2"
        target_s3 = f"{price_col}_S+3"

        # Exemple: on retire les colonnes cibles + quelques colonnes de type "str" ou date
        cols_to_remove = [
            'DATE_INTERROGATION', 'ANNEE', 'SEMAINE', 
            target_s1, target_s2, target_s3
        ]
        # Retirer également toute colonne contenant 'PRIX ' si on veut éviter la "fuite de données"
        # Mais c'est discutable selon la logique métier. À affiner !
        # Ici, on montre un exemple minimaliste:
        X_cols = [c for c in df.columns if c not in cols_to_remove]

        # On retire aussi toutes les colonnes cibles des features
        X_train = df_train[X_cols].copy()
        y_train_s1 = df_train[target_s1].copy()
        y_train_s2 = df_train[target_s2].copy()
        y_train_s3 = df_train[target_s3].copy()

        # Idem pour la partie test (à titre d'évaluation interne)
        X_test = df_test[X_cols].copy()
        y_test_s1 = df_test[target_s1].copy()
        y_test_s2 = df_test[target_s2].copy()
        y_test_s3 = df_test[target_s3].copy()

        # 4) Création des modèles
        model_s1 = XGBRegressor(**self.model_params)
        model_s2 = XGBRegressor(**self.model_params)
        model_s3 = XGBRegressor(**self.model_params)

        # 5) Entraînement (on ignore les NaN dans y)
        #    Noter qu'il faut éventuellement filtrer X_train / y_train pour enlever les lignes vides
        mask_s1 = y_train_s1.notnull()
        model_s1.fit(X_train[mask_s1], y_train_s1[mask_s1])

        mask_s2 = y_train_s2.notnull()
        model_s2.fit(X_train[mask_s2], y_train_s2[mask_s2])

        mask_s3 = y_train_s3.notnull()
        model_s3.fit(X_train[mask_s3], y_train_s3[mask_s3])

        # 6) Stockage des modèles
        self.models[(price_col, 'S+1')] = model_s1
        self.models[(price_col, 'S+2')] = model_s2
        self.models[(price_col, 'S+3')] = model_s3

        # 7) (Optionnel) Évaluation basique sur la période test (2024+)
        if not X_test.empty:
            # Prévoir sur la partie test
            pred_s1 = model_s1.predict(X_test)
            pred_s2 = model_s2.predict(X_test)
            pred_s3 = model_s3.predict(X_test)

            # Calcul d’un RMSE moyen (à titre indicatif)
            rmse_s1 = np.sqrt(np.mean((pred_s1 - y_test_s1) ** 2))
            rmse_s2 = np.sqrt(np.mean((pred_s2 - y_test_s2) ** 2))
            rmse_s3 = np.sqrt(np.mean((pred_s3 - y_test_s3) ** 2))

            print(f"[{price_col}] Test RMSE (S+1 / S+2 / S+3) = ({rmse_s1:.2f}, {rmse_s2:.2f}, {rmse_s3:.2f})")

            # Créer un DataFrame pour l'export
            export_df = pd.DataFrame({
                "PRODUIT_GROUPE": price_col.replace("PRIX ", ""),
                "SAISON": df_test["SAISON"].values,
                "SEMAINE_SAISON": df_test["SEMAINE_SAISON"].values,
                "PRIX_S1_PREVU": pred_s1,
                "PRIX_S1_REEL": y_test_s1.values,
                "PRIX_S2_PREVU": pred_s2,
                "PRIX_S2_REEL": y_test_s2.values,
                "PRIX_S3_PREVU": pred_s3,
                "PRIX_S3_REEL": y_test_s3.values,
            })

            # Exporter les prédictions et les valeurs réelles dans un fichier Excel
            output_dir = "/app/tests"

            # Nettoyage avancé du nom pour éviter les caractères problématiques
            clean_price_col = re.sub(r"[^\w\s-]", "_", price_col.replace(" ", "_"))
            output_path = os.path.join(output_dir, f"evaluation_{clean_price_col}.xlsx")

            # Vérifier et créer le répertoire si nécessaire
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                print(f"Répertoire créé : {output_dir}")

            # Assurer la création du chemin complet du fichier
            parent_dir = os.path.dirname(output_path)
            if not os.path.exists(parent_dir):
                os.makedirs(parent_dir)
                print(f"Répertoire parent créé : {parent_dir}")

            export_df.to_excel(output_path, index=False)
            print(f"Fichier exporté : {output_path}")

            # Générer un graphique
            plt.figure(figsize=(12, 6))

            # Tracer les courbes pour S+1
            plt.plot(export_df["SEMAINE_SAISON"], export_df["PRIX_S1_PREVU"], label="PRIX_S1_PREVU", marker="o")
            plt.plot(export_df["SEMAINE_SAISON"], export_df["PRIX_S1_REEL"], label="PRIX_S1_REEL", linestyle="--", marker="x")

            # Tracer les courbes pour S+2
            plt.plot(export_df["SEMAINE_SAISON"], export_df["PRIX_S2_PREVU"], label="PRIX_S2_PREVU", marker="o")
            plt.plot(export_df["SEMAINE_SAISON"], export_df["PRIX_S2_REEL"], label="PRIX_S2_REEL", linestyle="--", marker="x")

            # Tracer les courbes pour S+3
            plt.plot(export_df["SEMAINE_SAISON"], export_df["PRIX_S3_PREVU"], label="PRIX_S3_PREVU", marker="o")
            plt.plot(export_df["SEMAINE_SAISON"], export_df["PRIX_S3_REEL"], label="PRIX_S3_REEL", linestyle="--", marker="x")

            # Ajouter des légendes et des titres
            plt.title(f"Comparaison des prix prévus et réels ({price_col.replace('PRIX ', '')})", fontsize=14)
            plt.xlabel("Semaine Saison", fontsize=12)
            plt.ylabel("Prix (€)", fontsize=12)
            plt.legend()
            plt.grid(True)

            # Afficher et sauvegarder le graphique
            plt.tight_layout()
            graph_path = os.path.join(output_dir, f"graph_{clean_price_col}.png")
            plt.savefig(graph_path)
            plt.show()

            print(f"Graphique sauvegardé dans : {graph_path}")

            # Exporter les données de test X_test pour analyse
            x_test_path = os.path.join(output_dir, f"x_test_{clean_price_col}.xlsx")

            # Vérifier et créer le répertoire si nécessaire
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # Exporter X_test
            X_test.to_excel(x_test_path, index=False)
            print(f"X_test exporté : {x_test_path}")


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

        # 5) Pour chaque colonne "PRIX EXP POMME", on fait la prévision future et on insère dans la table
        for col in all_price_cols:
            s1, s2, s3 = self.generate_future_forecast(df_complet, col)
            if s1 is None:
                # Problème de modèle manquant, on skip
                continue

            # On insère
            self.insert_forecasts_into_db(
                date_interrogation=date_interrogation,
                produit_groupe=col,  # On peut mettre le nom exact de la colonne comme "PRODUIT_GROUPE"
                prix_s1=s1,
                prix_s2=s2,
                prix_s3=s3
            )
            print(f"Insertion OK pour {col} (S+1={s1:.2f}, S+2={s2:.2f}, S+3={s3:.2f})")

        print("\n=== FIN DE L'ENTRAINEMENT ET DE L'INSERTION DES PREVISIONS ===")


    def train_model_s1_and_set_features(
        self,
        X_train,
        y_train_s1,
        X_test,
        y_test_s1,
        initial_features=None,
        max_iter=20
    ):
        """
        Entraîne un model_s1 en testant une approche 'dichotomie + ajustements unitaires'
        pour minimiser le RMSE.

        Étapes principales :
        1. Entraîner un modèle sur le set courant de features.
        2. Retirer la moitié des features les moins importantes => nouveau set de features.
        3. Tester le RMSE :
        - si amélioration => on continue avec ce set réduit.
        - sinon => on revient en arrière et on tente des suppressions unitaires ou des ajouts unitaires.
            * On supprime 1 colonne : si ça améliore => on continue, sinon => on annule et on essaye d'ajouter 1 feature.
            * On ajoute 1 colonne : si ça améliore => on continue, sinon => on arrête la boucle.
        4. Répéter jusqu'à max_iter ou plus aucune amélioration.

        Parameters
        ----------
        X_train : pd.DataFrame
            Données d'entraînement
        y_train_s1 : pd.Series
            Label pour S+1
        X_test : pd.DataFrame
            Données de test
        y_test_s1 : pd.Series
            Label de test pour S+1
        initial_features : list
            Liste initiale de features à considérer.
        max_iter : int
            Nombre maximum d'itérations.

        Returns
        -------
        best_model_s1 : XGBRegressor
            Le meilleur modèle trouvé (RMSE minimal)
        best_features : list
            La liste de features associée à ce meilleur modèle
        best_rmse : float
            Le RMSE correspondant
        rmse_history : list of (iter, nb_features, rmse)
            Historique des itérations
        """

        import numpy as np
        import pandas as pd
        from xgboost import XGBRegressor

        # --- 0) Initialisation des variables
        if initial_features is None:
            current_features = list(X_train.columns)
        else:
            current_features = list(initial_features)

        # Dans le cas où on souhaite garder en mémoire les features qu'on a retirées
        # (pour les réintroduire potentiellement dans le mode 'ajout')
        # on peut considérer que "excluded_features" = tout - current_features
        excluded_features = [f for f in X_train.columns if f not in current_features]

        # Paramètres du modèle XGB
        xgb_params = self.model_params.copy()

        # On entraîne un premier modèle sur la config initiale
        best_model_s1 = None
        best_rmse = float('inf')
        best_features = current_features[:]
        rmse_history = []

        # Petite fonction utilitaire pour entraîner + calculer le rmse
        def train_and_get_rmse(features):
            model = XGBRegressor(**xgb_params)
            mask_s1 = y_train_s1.notnull()
            model.fit(X_train[features][mask_s1], y_train_s1[mask_s1])
            if not X_test.empty and not y_test_s1.empty:
                pred_test = model.predict(X_test[features])
                rmse_val = np.sqrt(np.mean((pred_test - y_test_s1) ** 2))
            else:
                # fallback sur le train si pas de test
                print("ATTENTION Pas de données de test, on utilise le train pour le RMSE.")
                pred_train = model.predict(X_train[features][mask_s1])
                rmse_val = np.sqrt(np.mean((pred_train - y_train_s1[mask_s1]) ** 2))
            return model, rmse_val

        # On commence par entraîner sur le set initial
        model_s1, rmse_s1 = train_and_get_rmse(current_features)
        best_model_s1 = model_s1
        best_rmse = rmse_s1
        print(f"Initial RMSE S1 : {rmse_s1:.4f}")
        best_features = current_features[:]
        rmse_history.append((0, len(current_features), rmse_s1))

        # fonction pour obtenir l'importance et trier
        def get_importance_order(model, feature_list):
            importances = model.feature_importances_
            df_imp = pd.DataFrame({
                "feature": feature_list,
                "importance": importances
            }).sort_values("importance", ascending=False)
            return df_imp

        iteration = 0
        while iteration < max_iter:
            iteration += 1
            # 1) Entraîner un modèle sur current_features pour connaître l'ordre d'importance
            model_s1, rmse_s1 = train_and_get_rmse(current_features)
            # 2) Récupérer l'ordre d'importance
            df_imp = get_importance_order(model_s1, current_features)

            # 3) Retirer la moitié (arrondie vers le bas) la moins importante
            half = len(df_imp) // 2
            reduced_features = list(df_imp.iloc[:half]["feature"].values)

            # on entraîne avec ce "reduced_features"
            model_reduced, rmse_reduced = train_and_get_rmse(reduced_features)
            print(f"Iteration {iteration} : RMSE réduit = {rmse_reduced:.4f} (vs {rmse_s1:.4f})")

            if rmse_reduced <= rmse_s1:
                # => On garde la réduction (ça s'est amélioré)
                current_features = reduced_features
                # On voit si c'est notre best
                if rmse_reduced <= best_rmse:
                    best_rmse = rmse_reduced
                    best_model_s1 = model_reduced
                    best_features = current_features[:]
                rmse_history.append((iteration, len(current_features), rmse_reduced))
                # et on continue la boucle
                continue
            else:
                # => Ça empire => on revient à l'ancien set => on tente des modifs unitaires
                # "Revert"
                # On va travailler sur un clone local = old_features
                old_features = current_features[:]

                # --- MODE SUPPRESSION 1 à 1 ---
                # On prend la moins importante sur old_features
                least_important_feat = df_imp.iloc[-1]["feature"]  # tout en bas
                try_features_minus1 = [f for f in old_features if f != least_important_feat]

                model_minus1, rmse_minus1 = train_and_get_rmse(try_features_minus1)

                if rmse_minus1 <= rmse_s1:
                    # => On a amélioré en supprimant 1 feature => 
                    # on va continuer à supprimer 1 par 1 dans l'ordre d'importance croissante
                    # (cad du bas de df_imp vers le haut)
                    print(f"Amélioration en retirant la feature {least_important_feat}, on continue sur mode -1")
                    current_features = try_features_minus1[:]
                    # On prend la feature la moins importante suivante
                    # On fait une boucle tant que l'amélioration est positive
                    # (on pourrait faire un while True, mais limitons nous)
                    for i in range(1, len(df_imp)):
                        # Recalculer l’ordre d’importance sur le set current_features
                        # (pas forcément le même ordre que df_imp précédent)
                        model_tmp, rmse_tmp = train_and_get_rmse(current_features)
                        new_imp = get_importance_order(model_tmp, current_features)
                        if len(new_imp) < 2:
                            break
                        # on teste la suppression de la moins importante
                        feat_to_remove = new_imp.iloc[-1]["feature"]
                        next_try_features = [f for f in current_features if f != feat_to_remove]
                        model_next, rmse_next = train_and_get_rmse(next_try_features)
                        if rmse_next < rmse_tmp:
                            # ok, amélioration => on garde
                            current_features = next_try_features
                            if rmse_next < best_rmse:
                                best_rmse = rmse_next
                                best_model_s1 = model_next
                                best_features = current_features[:]
                        else:
                            # pas d'amélioration => on stoppe la suppression
                            break

                    # on sort du mode "supprimer 1 par 1"
                    rmse_history.append((iteration, len(current_features), best_rmse))
                    continue

                else:
                    # => On n'a pas amélioré en retirant 1 feature => on revert
                    print("Suppression unitaire ne marche pas, on annule et on essaye l'ajout.")

                    # --- MODE AJOUT 1 à 1 ---
                    # On a old_features => on essaye d'ajouter 1 feature dans excluded_features
                    # Il faut un ordre pour les features qu'on n'a pas.
                    # on va utiliser un "super modèle" entraîné sur toutes les colonnes
                    # pour connaître l'importance "globale".
                    model_full, rmse_full = train_and_get_rmse(list(X_train.columns))
                    full_imp = get_importance_order(model_full, list(X_train.columns))

                    # On trie excluded_features par importance décroissante
                    # (tu peux choisir l'ordre que tu veux, c'est un choix)
                    df_excluded = full_imp[full_imp["feature"].isin(excluded_features)].copy()
                    # tri desc sur importance
                    df_excluded.sort_values("importance", ascending=False, inplace=True)
                    df_excluded_feats = list(df_excluded["feature"].values)

                    # On va tester l'ajout 1 par 1 dans l'ordre décroissant d'importance
                    improved_once = False
                    temp_current = old_features[:]
                    rmse_temp = rmse_s1

                    for feat_add in df_excluded_feats:
                        try_features_add1 = temp_current + [feat_add]
                        model_add1, rmse_add1 = train_and_get_rmse(try_features_add1)
                        if rmse_add1 < rmse_temp:
                            # ça améliore => on garde
                            temp_current = try_features_add1
                            rmse_temp = rmse_add1
                            # On retire feat_add de excluded_features
                            if feat_add in excluded_features:
                                excluded_features.remove(feat_add)
                            improved_once = True
                        else:
                            # pas d'amélioration => on n'ajoute pas
                            pass

                    if improved_once:
                        # On a trouvé un set en ajoutant des features
                        current_features = temp_current
                        if rmse_temp < best_rmse:
                            best_rmse = rmse_temp
                            best_model_s1 = model_add1
                            best_features = current_features[:]
                        rmse_history.append((iteration, len(current_features), best_rmse))
                        continue
                    else:
                        # On n'a pas réussi à améliorer en ajoutant => on arrête
                        print("Aucune amélioration ni en -1 ni en +1, on s'arrête.")
                        rmse_history.append((iteration, len(current_features), best_rmse))
                        break

        return best_model_s1, best_features, best_rmse, rmse_history



if __name__ == "__main__":
    import warnings
    warnings.filterwarnings(
        "ignore",
        message="pandas only supports SQLAlchemy connectable",
        category=UserWarning
    )

    # 1) Exemple de config de connexion
    db_config = {
        "host": "prixfetl_postgres",
        "port": 5432,
        "database": "IAFetL",
        "user": "prixfetl",
        "password": "Leumces123"
    }

    # 2) Instancier notre PriceTrainer
    trainer = PriceTrainer(db_config)

    # 3) Préparer le dataset
    df_complet = trainer.prepare_dataset()

    # 4) Sélectionner la colonne à tester
    price_col = "PRIX EXP POMME GALA FRANCE 170/220G CAT.I PLATEAU 1RG"
    df = trainer.generate_shifted_targets(df_complet, price_col)

    # 5) Filtrer sur la période d'entraînement (2018->2023) et test (>=2024)
    df_train = df[df['SAISON'].between(2018, 2023)].copy()
    df_test = df[df['SAISON'] >= 2024].copy()

    # 6) Définir target_s1
    target_s1 = f"{price_col}_S+1"

    # 7) Retirer les colonnes cibles et les colonnes date/ID
    cols_to_remove = [
        'DATE_INTERROGATION', 'ANNEE', 'SEMAINE', 
        f"{price_col}_S+1", f"{price_col}_S+2", f"{price_col}_S+3"
    ]
    X_cols = [c for c in df.columns if c not in cols_to_remove]

    X_train = df_train[X_cols].copy()
    y_train_s1 = df_train[target_s1].copy()
    X_test = df_test[X_cols].copy()
    y_test_s1 = df_test[target_s1].copy()

    # 8) Appeler la méthode d’entraînement par dichotomie + ajustements
    best_model_s1, best_features, best_rmse, rmse_history = trainer.train_model_s1_and_set_features(
        X_train=X_train,
        y_train_s1=y_train_s1,
        X_test=X_test,
        y_test_s1=y_test_s1,
        initial_features=None,  # ou bien une liste précise de features
        max_iter=20
    )

    # 9) Afficher les résultats
    #print(f"\nMeilleure liste de features : {best_features}")
    print(f"Meilleur RMSE obtenu       : {best_rmse:.4f}")
    print("Historique des itérations :")
    for (it, nbf, rmse) in rmse_history:
        print(f"  -> Iteration={it}, nb_features={nbf}, RMSE={rmse:.4f}")

    # 10) Si tu veux, tu peux maintenant entraîner model_s2 et model_s3 
    #     avec la liste best_features, ou tu peux simplement stocker best_model_s1
    #     dans trainer.models[(price_col, 'S+1')] si tu veux le réutiliser plus tard.
    trainer.models[(price_col, 'S+1')] = best_model_s1
    print("\nEntraînement finalisé pour model_s1. Tu peux maintenant lancer model_s2 et model_s3 au besoin.")
