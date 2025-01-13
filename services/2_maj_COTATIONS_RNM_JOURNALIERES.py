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

        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--headless=new")  # Mode headless nouvelle génération
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--disable-software-rasterizer")
        chrome_options.add_argument("--window-size=1920x1080")

        service = Service(self.driver_path)
        self.browser = webdriver.Chrome(service=service, options=chrome_options)

    def get_last_date_in_db(self):
        """
        Se connecte à la base de données,
        retourne la dernière date (colonne DATE)
        de la table COTATIONS_RNM_JOURNALIERES
        sous forme d'objet datetime, ou None si la table est vide.
        """
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(f"Base de données introuvable : {self.db_path}")

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            query = "SELECT MAX(DATE) FROM COTATIONS_RNM_JOURNALIERES"
            cursor.execute(query)
            result = cursor.fetchone()
            conn.close()

            if result and result[0]:
                # result[0] est probablement au format 'YYYY-MM-DD'
                return datetime.strptime(result[0], '%Y-%m-%d')
            else:
                return None  # table vide

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
                        row[0], row[1], row[2], row[3],
                        row[4], row[5], row[6], row[7], row[8]
                    ))
            conn.commit()
            conn.close()

        except sqlite3.Error as e:
            raise sqlite3.Error(f"Erreur d'insertion dans la BDD : {e}")

    def run_update(self):
        """
        1) Récupère la dernière date de la table,
        2) Parcourt les dates manquantes (jusqu'à aujourd'hui),
        3) Scrape le site rnm.franceagrimer.fr,
        4) Convertit, insère en BDD,
        5) Ferme le webdriver (et affiche le contenu de la table en debug).
        """
        # 1) Dernière date en base
        last_date_in_db = self.get_last_date_in_db()
        if last_date_in_db:
            start_date = last_date_in_db + timedelta(days=1)
        else:
            # S’il n’y a aucune date en base, on commence au 2023-08-01
            start_date = datetime(2023, 8, 1)

        end_date = datetime.today()
        if start_date > end_date:
            print("Aucune mise à jour n'est nécessaire. La base est déjà à jour.")
            self.browser.quit()
            return

        # 2) Accéder au site web
        self.browser.get('https://rnm.franceagrimer.fr/prix')

        wait = WebDriverWait(self.browser, 20)

        # 3) Saisir "pomme" dans le champ "produit" et appuyer sur Entrée
        try:
            produit_field = wait.until(EC.presence_of_element_located((By.ID, "produit")))
            produit_field.clear()
            produit_field.send_keys("pomme")
            produit_field.send_keys(Keys.ENTER)
        except TimeoutException:
            print("Champ produit non trouvé lors de l'ouverture du site.")
            self.browser.quit()
            return

        # 4) Boucle sur les dates souhaitées
        dates = pd.date_range(start=start_date, end=end_date)
        for date in dates:
            formatted_date = date.strftime('%d%m%y')
            date_interrogation = date.strftime('%Y-%m-%d')

            # Saisir la date
            try:
                date_field = wait.until(EC.presence_of_element_located((By.ID, "val1")))
                date_field.clear()
                date_field.send_keys(formatted_date)
            except TimeoutException:
                print(f"Élément de date non trouvé pour : {formatted_date}")
                continue

            # Cliquer sur OK
            try:
                ok_button = wait.until(EC.element_to_be_clickable((By.XPATH, "//input[@value='OK']")))
                ok_button.click()
            except TimeoutException:
                print(f"Bouton OK non trouvé pour : {formatted_date}")
                continue

            # Cliquer sur "Voir dans un tableur"
            try:
                tableur_link = wait.until(EC.element_to_be_clickable(
                    (By.XPATH, "//a[@class='droit' and contains(@onclick, 'tab')]/img[@alt='lien']"))
                )
                tableur_link.click()
            except TimeoutException:
                print(f"Lien tableur non trouvé pour : {formatted_date}")
                continue

            # Attendre le téléchargement du fichier
            time.sleep(5)  # Ajuster ce délai si nécessaire

            # Rechercher le fichier .slk
            downloaded_file = None
            for file in os.listdir(os.getcwd()):
                if file.endswith(".slk"):
                    downloaded_file = file
                    break

            if downloaded_file:
                # Conversion slk -> csv
                csv_file = downloaded_file.replace('.slk', '.csv')
                CotationRnmScraper.slk_to_csv(downloaded_file, csv_file)

                # Lire le CSV en DataFrame, en ignorant les 4 premières lignes
                data_df = pd.read_csv(csv_file, skiprows=3, encoding='ISO-8859-1')
                # On ne garde que les lignes dont la 6ème colonne n’est pas vide
                data_df = data_df[data_df.iloc[:, 5].notna()]

                # Insert DB
                self.insert_data(date_interrogation, data_df)

                # Supprimer les fichiers téléchargés
                os.remove(downloaded_file)
                os.remove(csv_file)

        # Fermer le navigateur
        self.browser.quit()

        # (Optionnel) Afficher le contenu de la table pour vérif
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
    # Adapte les chemins selon ta configuration
    DB_PATH = "/app/data/IAFetL.db"
    DRIVER_PATH = "/home/IAFetL/driver/chromedriver/chromedriver"

    scraper = CotationRnmScraper(DB_PATH, DRIVER_PATH)
    scraper.run_update()
