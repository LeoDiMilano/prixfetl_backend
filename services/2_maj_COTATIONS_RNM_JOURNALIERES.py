import sqlite3
import time
import os
import csv
import sys
import re
import pandas as pd
from datetime import datetime, timedelta

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import TimeoutException


class CotationRnmScraper:
    """
    Cette classe gère la mise à jour des cotations RNM journalières :
      1) Récupère la dernière date présente dans la BDD
      2) Scrape le site rnm.franceagrimer.fr pour toutes les dates manquantes
      3) Convertit les fichiers .slk en .csv, puis insère dans la table
    """

    def __init__(self, db_path: str, driver_path: str):
        """
        Initialise l'instance :
          - db_path : chemin vers la BDD SQLite
          - driver_path : chemin vers le chromedriver
        """
        self.db_path = db_path
        self.driver_path = driver_path
        self.browser = None
        self.init_driver()

    def init_driver(self):
        """
        Configure et instancie le webdriver Chrome,
        stocke l'instance dans self.browser.
        """
        chrome_options = webdriver.ChromeOptions()
        chrome_options.add_experimental_option("prefs", {
            "download.default_directory": os.getcwd(),
            "download.prompt_for_download": False,
            "download.directory_upgrade": True,
            "safebrowsing.enabled": True
        })

        # Options de stabilité
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-background-networking")
        chrome_options.add_argument("--disable-extensions")
        chrome_options.add_argument("--disable-sync")
        chrome_options.add_argument("--disable-translate")
        chrome_options.add_argument("--disable-gpu")  # Désactiver le GPU pour headless
        chrome_options.add_argument("--disable-software-rasterizer")
        chrome_options.add_argument("--disable-crash-reporter")
        chrome_options.add_argument("--disable-logging")
        chrome_options.add_argument("--disable-infobars")
        chrome_options.add_argument("--disable-dev-tools")
        chrome_options.add_argument("--remote-debugging-port=9222")
        chrome_options.add_argument("--headless")  # Si problème, teste sans le mode headless
        chrome_options.add_argument("--window-size=1920x1080")  # Taille par défaut pour éviter les problèmes

        service = Service(self.driver_path)
        self.browser = webdriver.Chrome(service=service, options=chrome_options)
        def get_last_date_in_db_for_product(self, product: str):
            """
            Retourne la dernière date_interrogation (YYYY-MM-DD)
            pour la table COTATIONS_RNM_JOURNALIERES
            en filtrant sur LIBELLE_PRODUIT LIKE 'PRODUIT%',
            ou None si aucune date n'est trouvée.
            """
            if not os.path.exists(self.db_path):
                raise FileNotFoundError(f"Base de données introuvable : {self.db_path}")

            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                # On force le produit en majuscules et on ajoute le wildcard '%'
                sql_query = """
                    SELECT MAX(DATE_INTERROGATION)
                    FROM COTATIONS_RNM_JOURNALIERES
                    WHERE LIBELLE_PRODUIT LIKE ?
                """
                # On suppose que dans la base, LIBELLE_PRODUIT est stocké en majuscule.
                # Par sécurité, on force la majuscule côté Python aussi.
                like_value = product.upper() + '%'
                cursor.execute(sql_query, (like_value,))
                result = cursor.fetchone()
                conn.close()

                if result and result[0]:
                    # result[0] est du style 'YYYY-MM-DD'
                    return datetime.strptime(result[0], '%Y-%m-%d')
                else:
                    return None

            except sqlite3.Error as e:
                raise sqlite3.Error(f"Erreur lors de l'accès à la base de données : {e}")

    def get_last_date_in_db_for_product(self, product: str):
        """
        Retourne la dernière date_interrogation (YYYY-MM-DD)
        pour la table COTATIONS_RNM_JOURNALIERES
        en filtrant sur LIBELLE_PRODUIT LIKE 'PRODUIT%',
        ou None si aucune date n'est trouvée.
        """
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(f"Base de données introuvable : {self.db_path}")

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            # On force le produit en majuscules et on ajoute le wildcard '%'
            sql_query = """
                SELECT MAX(DATE_INTERROGATION)
                FROM COTATIONS_RNM_JOURNALIERES
                WHERE LIBELLE_PRODUIT LIKE ?
            """
            like_value = product.upper() + '%'
            cursor.execute(sql_query, (like_value,))
            result = cursor.fetchone()
            conn.close()

            if result and result[0]:
                # result[0] est du style 'YYYY-MM-DD'
                return datetime.strptime(result[0], '%Y-%m-%d')
            else:
                return None

        except sqlite3.Error as e:
            raise sqlite3.Error(f"Erreur lors de l'accès à la base de données : {e}")


    @staticmethod
    def slk_to_csv(slk_filename, csv_filename):
        """
        Convertit un fichier .slk en .csv
        (Méthode quasi identique à celle fournie précédemment).
        """
        data = {}
        max_row = 0
        max_col = 0

        cell_re = re.compile(r'^(?P<type>[FC]);(?P<params>.*)')
        param_re = re.compile(r'(?P<key>[A-Z])(?P<value>[^;]*)')

        current_x = None
        current_y = None

        with open(slk_filename, 'r', encoding='ISO-8859-1') as slk_file:
            for line in slk_file:
                line = line.strip()
                if not line:
                    continue
                match = cell_re.match(line)
                if match:
                    line_type = match.group('type')
                    params_str = match.group('params')
                    params = params_str.split(';')
                    x = None
                    y = None
                    k = None
                    for param in params:
                        param_match = param_re.match(param)
                        if param_match:
                            key = param_match.group('key')
                            value = param_match.group('value')
                            if key == 'X':
                                x = int(value)
                            elif key == 'Y':
                                y = int(value)
                            elif key == 'K':
                                k_value = value
                                if k_value.startswith('"') and k_value.endswith('"'):
                                    k = k_value[1:-1].replace('""', '"')
                                else:
                                    k = k_value
                    if x is not None:
                        current_x = x
                    if y is not None:
                        current_y = y

                    if line_type in ['F', 'C']:
                        if line_type == 'F':
                            continue
                        if line_type == 'C':
                            if current_x is None or current_y is None:
                                continue
                            if k is not None:
                                data[(current_y, current_x)] = k
                                if current_y > max_row:
                                    max_row = current_y
                                if current_x > max_col:
                                    max_col = current_x

        table = [['' for _ in range(max_col)] for _ in range(max_row)]
        for (y, x), value in data.items():
            table[y - 1][x - 1] = value

        with open(csv_filename, 'w', newline='', encoding='ISO-8859-1') as csv_file:
            writer = csv.writer(csv_file)
            for row in table:
                writer.writerow(row)

    def insert_data(self, date_interrogation: str, data: pd.DataFrame):
        """
        Insère les données dans la table COTATIONS_RNM_JOURNALIERES
        de la BDD SQLite.
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            for _, row in data.iterrows():
                if not row.isnull().all():
                    cursor.execute('''
                        INSERT INTO COTATIONS_RNM_JOURNALIERES (
                            DATE_INTERROGATION, DATE, MARCHE, STADE, LIBELLE_PRODUIT,
                            UNITE, PRIX_JOUR, VARIATION, MINI, MAXI
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        date_interrogation,
                        row.iloc[0], row.iloc[1], row.iloc[2], row.iloc[3],
                        row.iloc[4], row.iloc[5], row.iloc[6], row.iloc[7], row.iloc[8]

                    ))
            conn.commit()
            conn.close()

        except sqlite3.Error as e:
            raise sqlite3.Error(f"Erreur d'insertion dans la BDD : {e}")

    def scrape_and_insert_product_data(self, product: str, start_date: datetime, end_date: datetime):
        """
        Accède au site FranceAgriMer, saisit le 'product' donné,
        puis boucle sur la période [start_date, end_date],
        télécharge le .slk, le convertit, insère dans la BDD.
        """
        try:
            wait = WebDriverWait(self.browser, 20)

            # Recharger la page pour le produit
            print(f"[INFO] Accès au site pour le produit : {product}")
            self.browser.get('https://rnm.franceagrimer.fr/prix')

            # Saisir le produit
            try:
                produit_field = wait.until(EC.presence_of_element_located((By.ID, "produit")))
                produit_field.clear()
                produit_field.send_keys(product)
                produit_field.send_keys(Keys.ENTER)
            except TimeoutException:
                print(f"[ERREUR] Champ produit non trouvé pour {product}.")
                return

            # Boucle sur les dates
            dates = pd.date_range(start=start_date, end=end_date)
            for date in dates:
                formatted_date = date.strftime('%d%m%y')
                date_interrogation = date.strftime('%Y-%m-%d')

                try:
                    # Saisir la date
                    date_field = wait.until(EC.presence_of_element_located((By.ID, "val1")))
                    date_field.clear()
                    date_field.send_keys(formatted_date)

                    # Cliquer sur OK
                    ok_button = wait.until(EC.element_to_be_clickable((By.XPATH, "//input[@value='OK']")))
                    ok_button.click()

                    # Cliquer sur "Voir dans un tableur"
                    tableur_link = wait.until(EC.element_to_be_clickable(
                        (By.XPATH, "//a[@class='droit' and contains(@onclick, 'tab')]/img[@alt='lien']"))
                    )
                    tableur_link.click()

                    # Attendre le téléchargement du fichier
                    time.sleep(5)

                    # Conversion slk -> csv + insertion dans la BDD
                    downloaded_file = None
                    for file in os.listdir(os.getcwd()):
                        if file.endswith(".slk"):
                            downloaded_file = file
                            break

                    if downloaded_file:
                        csv_file = downloaded_file.replace('.slk', '.csv')
                        CotationRnmScraper.slk_to_csv(downloaded_file, csv_file)

                        # Charger et insérer les données
                        data_df = pd.read_csv(csv_file, skiprows=3, encoding='ISO-8859-1')
                        data_df = data_df[data_df.iloc[:, 5].notna()]
                        self.insert_data(date_interrogation, data_df)

                        # Supprimer les fichiers temporaires
                        os.remove(downloaded_file)
                        os.remove(csv_file)

                except TimeoutException:
                    print(f"[ERREUR] Échec du traitement pour {date_interrogation} | Produit : {product}")
                    continue

        except Exception as e:
            print(f"[ERREUR] Problème lors du traitement du produit {product}: {e}")

        finally:
            # Relancer le navigateur si nécessaire
            if not self.browser.service.is_connectable():
                print("[INFO] Redémarrage du navigateur.")
                self.init_driver()


    def run_update(self, products=None):
        """
        Pour chaque produit de la liste 'products',
        on cherche la dernière date en base où LIBELLE_PRODUIT LIKE product.upper()+'%',
        puis on scrape et insère les données à partir de cette date +1 jusqu'à aujourd'hui.
        """
        if products is None:
            products = ["pomme", "banane", "orange"]  # Ex. valeurs par défaut

        for product in products:
            print(f"\n=== Traitement du produit: {product} ===")
            last_date_for_prod = self.get_last_date_in_db_for_product(product)

            if last_date_for_prod:
                start_date = last_date_for_prod + timedelta(days=1)
            else:
                # Si aucune date en base pour ce produit, on choisit une date de départ par défaut
                start_date = datetime(2023, 8, 1)

            end_date = datetime.today()
            if start_date > end_date:
                print(f"Aucune mise à jour nécessaire pour {product} (déjà à jour).")
                continue

            # Scraper et insérer pour ce produit (cf. méthode qu’on peut isoler)
            self.scrape_and_insert_product_data(product, start_date, end_date)

        # Une fois terminé pour tous les produits, on ferme le navigateur
        self.browser.quit()
        # (optionnel) affichage debug
        self._debug_afficher_contenu_table()


    def _debug_afficher_contenu_table(self):
        """
        Petite méthode de debug pour afficher le contenu
        de la table (facultatif).
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM COTATIONS_RNM_JOURNALIERES LIMIT 10')
            rows = cursor.fetchall()
            print("Quelques lignes dans COTATIONS_RNM_JOURNALIERES :")
            for row in rows:
                print(row)
            conn.close()
        except sqlite3.Error as e:
            print(f"Erreur lors de l'affichage : {e}")


# ------------------------------------------------------------------------
# Point d’entrée principal (si on exécute directement ce script)
# ------------------------------------------------------------------------
if __name__ == "__main__":
    DB_PATH = "/app/data/IAFetL.db"
    DRIVER_PATH = "/app/driver/chromedriver/chromedriver"

    # Créer l’instance de scraper
    scraper = CotationRnmScraper(DB_PATH, DRIVER_PATH)

    # Passer la liste de produits à run_update
    PRODUCTS = ["pomme", "banane", "orange"]
    try:
        scraper.run_update(products=PRODUCTS)
    except Exception as e:
        print(f"[ERREUR] Une erreur est survenue : {e}")
    finally:
        # Assure-toi que le navigateur est fermé même en cas d'erreur
        if scraper.browser:
            scraper.browser.quit()


