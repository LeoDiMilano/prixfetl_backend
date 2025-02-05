# inference.py

import psycopg2
import pandas as pd
from datetime import date, timedelta
from data_preprocessing import ApplePriceDataLoader  # pour recharger le dernier df_sql

class ModelInference:
    """
    Gère l’inférence et la mise à jour des prix réels dans la table previsions_prix.
    """

    def __init__(self, db_config, trainer):
        self.db_config = db_config
        self.trainer = trainer  # Pour accéder aux modèles et, si besoin, au df_complet déjà construit

    def get_connection(self):
        return psycopg2.connect(**self.db_config)

    def get_last_date_in_db(self):
        """
        Récupère la dernière date présente dans la table `previsions_prix`. Si aucune date n'est trouvée,
        retourne le lundi de la première semaine_saison de 2024.

        Returns:
            datetime.date : La dernière date trouvée ou la date par défaut (2024-01-01).
        """
        query = """
        SELECT MAX("DATE_INTERROGATION")
        FROM public.previsions_prix;
        """
        conn = None
        last_date = None
        try:
            conn = self.get_connection()
            cur = conn.cursor()
            cur.execute(query)
            row = cur.fetchone()
            cur.close()

            if row and row[0] is not None:
                last_date = row[0]  # La dernière date trouvée
        except Exception as e:
            print(f"Erreur lors de la récupération de la dernière date : {e}")
            if conn:
                conn.rollback()
        finally:
            if conn:
                conn.close()

        # Retourne la date par défaut si aucune date n'est trouvée
        return last_date if last_date else date(2024, 8, 5)

    @staticmethod
    def generate_mondays_between(start_date, end_date):
        """
        Génère tous les lundis entre deux dates données (incluses).

        Args:
            start_date (datetime.date): Date de début.
            end_date (datetime.date): Date de fin.

        Yields:
            datetime.date: Chaque lundi dans l'intervalle.
        """
        current_date = start_date
        # Ajuste pour commencer au prochain lundi si la date de début n'est pas un lundi
        while current_date.weekday() != 0:  # 0 correspond à lundi
            current_date += timedelta(days=1)

        # Génère tous les lundis jusqu'à la date de fin
        while current_date <= end_date:
            yield current_date
            current_date += timedelta(days=7)


    def update_real_prices(self):
        """
        Met à jour les colonnes PRIX_REEL_Sx et VAR_PRIX_REEL_Sx dans la table previsions_prix
        en utilisant uniquement les données déjà présentes dans la table.
        """

        # SQL pour récupérer les lignes nécessitant une mise à jour
        select_sql = """
            SELECT
                "DATE_INTERROGATION",
                "PRODUIT_GROUPE",
                "PRIX_REEL_S",
                "PRIX_PREV_S1",
                "PRIX_PREV_S2",
                "PRIX_PREV_S3",
                "PRIX_REEL_S1",
                "PRIX_REEL_S2",
                "PRIX_REEL_S3"
            FROM previsions_prix
            ORDER BY "PRODUIT_GROUPE", "DATE_INTERROGATION"
        """

        rows_to_update = []
        conn = None
        try:
            conn = self.get_connection()
            cur = conn.cursor()
            cur.execute(select_sql)
            results = cur.fetchall()
            colnames = [desc[0] for desc in cur.description]

            for row in results:
                row_dict = dict(zip(colnames, row))
                rows_to_update.append(row_dict)

            cur.close()
        except Exception as e:
            print("Erreur SELECT previsions_prix:", e)
            if conn:
                conn.rollback()
            return
        finally:
            if conn:
                conn.close()

        if not rows_to_update:
            print("[update_real_prices] Rien à mettre à jour.")
            return

        # Préparer les mises à jour
        updates = []
        for i, row in enumerate(rows_to_update):
            date_lundi = row["DATE_INTERROGATION"]
            produit_groupe = row["PRODUIT_GROUPE"]
            real_price_s = row["PRIX_REEL_S"]
        
            # Obtenir les prix réels S+1, S+2, S+3 à partir des lignes suivantes
            real_price_s1 = rows_to_update[i + 1]["PRIX_REEL_S"] if i + 1 < len(rows_to_update) and rows_to_update[i + 1]["PRODUIT_GROUPE"] == produit_groupe else None
            real_price_s2 = rows_to_update[i + 2]["PRIX_REEL_S"] if i + 2 < len(rows_to_update) and rows_to_update[i + 2]["PRODUIT_GROUPE"] == produit_groupe else None
            real_price_s3 = rows_to_update[i + 3]["PRIX_REEL_S"] if i + 3 < len(rows_to_update) and rows_to_update[i + 3]["PRODUIT_GROUPE"] == produit_groupe else None

            # Calcul des variations
            var_s1 = self._calc_var_class_reelle(real_price_s1, real_price_s) if real_price_s1 else None
            var_s2 = self._calc_var_class_reelle(real_price_s2, real_price_s) if real_price_s2 else None
            var_s3 = self._calc_var_class_reelle(real_price_s3, real_price_s) if real_price_s3 else None

            updates.append({
                "DATE_INTERROGATION": date_lundi,
                "PRODUIT_GROUPE": produit_groupe,
                "PRIX_REEL_S1": real_price_s1,
                "PRIX_REEL_S2": real_price_s2,
                "PRIX_REEL_S3": real_price_s3,
                "VAR_PRIX_REEL_S1": var_s1,
                "VAR_PRIX_REEL_S2": var_s2,
                "VAR_PRIX_REEL_S3": var_s3
            })

        # Effectuer les mises à jour en base
        self._perform_updates_in_db(updates)
        print(f"[update_real_prices] {len(updates)} lignes mises à jour.")

    def _calc_var_class_reelle(self, real_price, prev_price):
        """
        Compare le prix réel (real_price) avec le prix prévu (prev_price)
        pour définir la classe de variation :
            - diff > +0.03 => 2
            - diff > +0.01 => 1
            - -0.01 <= diff <= +0.01 => 0
            - diff < -0.03 => -2
            - sinon => -1
        """
        if real_price is None or prev_price is None:
            return None

        diff = real_price - prev_price
        if diff > 0.03:
            return 2
        elif diff > 0.01:
            return 1
        elif -0.01 <= diff <= 0.01:
            return 0
        elif diff < -0.03:
            return -2
        else:
            return -1

    def _perform_updates_in_db(self, updates):
        """
        Met à jour la table previsions_prix pour chaque ligne de la liste `updates`.
        """
        if not updates:
            return

        query = """
        UPDATE previsions_prix
        SET
            "PRIX_REEL_S1" = %s,
            "PRIX_REEL_S2" = %s,
            "PRIX_REEL_S3" = %s,
            "VAR_PRIX_REEL_S1" = %s,
            "VAR_PRIX_REEL_S2" = %s,
            "VAR_PRIX_REEL_S3" = %s
        WHERE "DATE_INTERROGATION" = %s
          AND "PRODUIT_GROUPE" = %s
        """

        conn = None
        try:
            conn = self.get_connection()
            cur = conn.cursor()
            for row in updates:
                cur.execute(query, (
                    row["PRIX_REEL_S1"],
                    row["PRIX_REEL_S2"],
                    row["PRIX_REEL_S3"],
                    row["VAR_PRIX_REEL_S1"],
                    row["VAR_PRIX_REEL_S2"],
                    row["VAR_PRIX_REEL_S3"],
                    row["DATE_INTERROGATION"],
                    row["PRODUIT_GROUPE"]
                ))
            conn.commit()
            cur.close()
        except Exception as e:
            print("Erreur lors de la mise à jour des prix réels:", e)
            if conn:
                conn.rollback()
        finally:
            if conn:
                conn.close()

    def fill_previsions_for_missing_mondays(self, list_of_products):
        """
        Remplit les prévisions manquantes dans la table `previsions_prix` pour tous les lundis
        entre la dernière date en base et aujourd'hui (date actuelle).

        Arguments :
        - list_of_products (list) : liste des produits pour lesquels générer les prévisions.
        """
        # Vérifie si les modèles sont déjà entraînés
        if not self.trainer.models:
            raise ValueError("Les modèles doivent être entraînés avant de remplir les prévisions manquantes.")

        # Étape 1 : Récupérer la dernière date présente dans la table
        last_date = self.get_last_date_in_db()
        today = date.today()

        print(f"Dernière date en base : {last_date}. Génération des prévisions jusqu'à {today}.")

        # Étape 2 : Générer les dates des lundis entre last_date et today
        mondays = list(self.generate_mondays_between(last_date, today))

        if not mondays:
            print("Aucun lundi à compléter.")
            return

        print(f"Lundis à compléter : {[str(monday) for monday in mondays]}")

        # Étape 3 : Charger le DataFrame complet si ce n'est pas déjà fait
        if self.trainer.df_complet is None:
            print("Chargement des données complètes pour l'inférence.")
            self.trainer.df_complet = self.trainer.prepare_dataset()

        # Étape 4 : Pour chaque produit, générer les prévisions manquantes
        for product in list_of_products:
            col_name = f"PRIX {product}"

            if col_name not in self.trainer.df_complet.columns:
                print(f"[AVERTISSEMENT] La colonne '{col_name}' n'existe pas dans les données. Ignorée.")
                continue

            print(f"Génération des prévisions pour le produit : {product}")

            for monday in mondays:
                # Vérifie si une prévision existe déjà pour ce lundi et ce produit
                if self.check_existing_forecast(monday, product):
                    print(f"Prévision déjà existante pour {product} au {monday}. Ignorée.")
                    continue
                #si prix_reel_s est null, on ne fait pas de prévision
                current_price = self.get_current_price(self.trainer.df_complet, monday, col_name)

                if current_price is None or current_price == '':
                    print(f"Prix actuel introuvable pour {product} au {monday}. Prévisions non générées.")
                    continue

                # Étape 5 : Faire la prévision pour ce lundi
                prix_s1, prix_s2, prix_s3 = self.predict_for_one_date(self.trainer.df_complet, monday, col_name)

                if prix_s1 is None:
                    print(f"Impossible de générer des prévisions pour {product} au {monday}. Ignoré.")
                    continue

                # Étape 6 : Calculer les variations
                var_s1 = self.trainer.variation_class(prix_s1 - current_price)
                var_s2 = self.trainer.variation_class(prix_s2 - current_price)
                var_s3 = self.trainer.variation_class(prix_s3 - current_price)

                # Étape 7 : Insérer la prévision dans la table
                saison = self.trainer.df_complet.loc[self.trainer.df_complet['DATE_INTERROGATION'] == pd.to_datetime(monday), 'SAISON'].values[0]
                semaine_saison = self.trainer.df_complet.loc[self.trainer.df_complet['DATE_INTERROGATION'] == pd.to_datetime(monday), 'SEMAINE_SAISON'].values[0]
                prix_reel_s = self.trainer.df_complet.loc[self.trainer.df_complet['DATE_INTERROGATION'] == pd.to_datetime(monday), col_name].values[0]
                                
                self.insert_forecasts_into_db(
                    date_interrogation=monday,
                    produit_groupe=product,
                    prix_s1=prix_s1,
                    prix_s2=prix_s2,
                    prix_s3=prix_s3,
                    var_s1=var_s1,
                    var_s2=var_s2,
                    var_s3=var_s3,
                    saison=saison,
                    semaine_saison=semaine_saison,
                    prix_reel_s=prix_reel_s

                )

                print(f"Prévisions insérées pour {product} au {monday} : S+1={prix_s1:.2f}, S+2={prix_s2:.2f}, S+3={prix_s3:.2f}.")

        print("Remplissage des prévisions terminé.")

    def check_existing_forecast(self, monday, product):
        """
        Vérifie si une prévision existe déjà dans la table `previsions_prix`
        pour un lundi donné et un produit donné.

        Arguments :
        - monday (date) : date du lundi à vérifier.
        - product (str) : nom du produit à vérifier.

        Retourne :
        - bool : True si une prévision existe déjà, False sinon.
        """
        query = """
        SELECT 1 FROM public.previsions_prix
        WHERE "DATE_INTERROGATION" = %s AND "PRODUIT_GROUPE" = %s
        LIMIT 1;
        """
        conn = self.trainer.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(query, (monday, product))
                return cur.fetchone() is not None
        finally:
            conn.close()

    def get_current_price(self, df, monday, col_name):
        """
        Récupère le prix actuel pour un produit donné à une date donnée.

        Arguments :
        - df (DataFrame) : DataFrame contenant les données complètes.
        - monday (date) : date du lundi à vérifier.
        - col_name (str) : nom de la colonne du produit.

        Retourne :
        - float : le prix actuel ou None si introuvable.
        """
        
        row = df.loc[df['DATE_INTERROGATION'] == pd.to_datetime(monday)]
        # aide moi a debugger en affichant la date et le produit
        #print(f"monday={monday}, col_name={col_name}")
        #print(f"row={row}")

        if not row.empty:
            value = row[col_name].values[0]
            if pd.isna(value) or value == '':
                return None
            return value
        return None

    def predict_for_one_date(self, df, date_target, col_name):
        """
        Fait une prédiction pour une date cible donnée et une colonne spécifique en utilisant les modèles entraînés.

        Args:
            df (pd.DataFrame): Le DataFrame contenant les données complètes pour l'inférence.
            date_target (datetime.date): La date cible pour laquelle prédire.
            col_name (str): La colonne contenant les prix du produit.

        Returns:
            tuple: Les prédictions pour S+1, S+2 et S+3 (ou None si la prédiction échoue).
        """
        try:
            # Vérifier si le modèle pour la colonne et l'horizon existe
            if (col_name, 'S+1') not in self.trainer.models or \
            (col_name, 'S+2') not in self.trainer.models or \
            (col_name, 'S+3') not in self.trainer.models:
                print(f"[Erreur] Modèles manquants pour la colonne {col_name}.")
                return None, None, None

            # Obtenir les features sélectionnées pour la colonne
            selected_features = self.trainer.selected_features_for_column.get(col_name, None)
            if not selected_features:
                print(f"[Erreur] Aucune feature sélectionnée pour la colonne {col_name}.")
                return None, None, None

            # Filtrer les données pertinentes pour la date cible
            subset = df[df['DATE_INTERROGATION'] <= pd.to_datetime(date_target)].copy()
            subset.sort_values(by='DATE_INTERROGATION', inplace=True)

            # Préparer les features pour l'inférence
            X = subset[selected_features].tail(1)
            if X.empty:
                print(f"[Erreur] Pas de données disponibles pour la prédiction au {date_target}.")
                return None, None, None

            # Prédire pour les horizons S+1, S+2 et S+3
            pred_s1 = self.trainer.models[(col_name, 'S+1')].predict(X)[0]
            pred_s2 = self.trainer.models[(col_name, 'S+2')].predict(X)[0]
            pred_s3 = self.trainer.models[(col_name, 'S+3')].predict(X)[0]

            # Mapper les prédictions de [0, 1, 2, 3, 4] à [-2, -1, 0, 1, 2]
            pred_s1 = self.map_prediction(pred_s1)
            pred_s2 = self.map_prediction(pred_s2)
            pred_s3 = self.map_prediction(pred_s3)

            # Calculer les prix prévus en utilisant les valeurs centrales des classes de variation
            current_price = X[col_name].values[0]
            prix_s1 = self.calculate_predicted_price(current_price, pred_s1)
            prix_s2 = self.calculate_predicted_price(current_price, pred_s2)
            prix_s3 = self.calculate_predicted_price(current_price, pred_s3)

            return prix_s1, prix_s2, prix_s3
        except Exception as e:
            print(f"[Erreur] Échec de la prédiction pour {col_name} au {date_target} : {e}")
            return None, None, None

    def map_prediction(self, prediction):
        """
        Mappe les prédictions de [0, 1, 2, 3, 4] à [-2, -1, 0, 1, 2].
        """
        mapping = {0: -2, 1: -1, 2: 0, 3: 1, 4: 2}
        return mapping.get(prediction, 0)

    def calculate_predicted_price(self, current_price, prediction):
        """
        Calcule le prix prévu en utilisant la valeur centrale de la classe de variation.

        Args:
            current_price (float): Le prix actuel.
            prediction (int): La classe de variation prédite.

        Returns:
            float: Le prix prévu.
        """
        if prediction == 2:
            return current_price + 0.05
        elif prediction == 1:
            return current_price + 0.02
        elif prediction == 0:
            return current_price
        elif prediction == -1:
            return current_price - 0.02
        elif prediction == -2:
            return current_price - 0.05
        else:
            return current_price


    def insert_forecasts_into_db(self, date_interrogation, produit_groupe, prix_s1, prix_s2, prix_s3, var_s1, var_s2, var_s3, saison, semaine_saison,prix_reel_s):
        """
        Insère les prévisions de prix et leurs variations dans la table `previsions_prix`.

        Args:
            date_interrogation (datetime.date): Date de la prévision.
            produit_groupe (str): Nom du produit pour lequel la prévision est effectuée.
            prix_s1 (float): Prévision de prix pour S+1.
            prix_s2 (float): Prévision de prix pour S+2.
            prix_s3 (float): Prévision de prix pour S+3.
            var_s1 (int): Variation de prix pour S+1.
            var_s2 (int): Variation de prix pour S+2.
            var_s3 (int): Variation de prix pour S+3.
            saison (int): Saison de la prévision.
            semaine_saison (int): Semaine de la saison de la prévision.
            prix_reel_s (float): Prix actuel du produit.
        """
        query = """
        INSERT INTO previsions_prix (
            "DATE_INTERROGATION", 
            "PRODUIT_GROUPE", 
            "PRIX_PREV_S1", 
            "PRIX_PREV_S2", 
            "PRIX_PREV_S3", 
            "VAR_PRIX_PREV_S1", 
            "VAR_PRIX_PREV_S2", 
            "VAR_PRIX_PREV_S3",
            "SAISON",
            "SEMAINE_SAISON",
            "PRIX_REEL_S"
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT ("DATE_INTERROGATION", "PRODUIT_GROUPE") DO UPDATE SET
            "PRIX_PREV_S1" = EXCLUDED."PRIX_PREV_S1",
            "PRIX_PREV_S2" = EXCLUDED."PRIX_PREV_S2",
            "PRIX_PREV_S3" = EXCLUDED."PRIX_PREV_S3",
            "VAR_PRIX_PREV_S1" = EXCLUDED."VAR_PRIX_PREV_S1",
            "VAR_PRIX_PREV_S2" = EXCLUDED."VAR_PRIX_PREV_S2",
            "VAR_PRIX_PREV_S3" = EXCLUDED."VAR_PRIX_PREV_S3",
            "SAISON" = EXCLUDED."SAISON",
            "SEMAINE_SAISON" = EXCLUDED."SEMAINE_SAISON",
            "PRIX_REEL_S" = EXCLUDED."PRIX_REEL_S";
        """
        conn = None
        try:
            conn = self.get_connection()
            with conn.cursor() as cur:
                cur.execute(query, (
                    date_interrogation, 
                    produit_groupe, 
                    float(prix_s1) if prix_s1 is not None else None, 
                    float(prix_s2) if prix_s2 is not None else None, 
                    float(prix_s3) if prix_s3 is not None else None, 
                    int(var_s1) if var_s1 is not None else None, 
                    int(var_s2) if var_s2 is not None else None, 
                    int(var_s3) if var_s3 is not None else None,
                    int(saison) if saison is not None else None,
                    int(semaine_saison) if semaine_saison is not None else None,
                    float(prix_reel_s) if prix_reel_s is not None and not pd.isna(prix_reel_s) else None
                ))
            conn.commit()
            print(f"[SUCCESS] Prévisions insérées/actualisées pour {produit_groupe} au {date_interrogation}.")
        except Exception as e:
            if conn:
                conn.rollback()
            print(f"[ERROR] Échec de l'insertion des prévisions pour {produit_groupe} au {date_interrogation} : {e}")
        finally:
            if conn:
                conn.close()
