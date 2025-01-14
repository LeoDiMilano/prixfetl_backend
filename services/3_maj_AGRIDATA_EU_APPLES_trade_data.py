import os
import time
import sqlite3
import pandas as pd
from datetime import datetime

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException


class AgridataApplesTradeScraper:
    """
    Cette classe gère la mise à jour des données "Apples trade" depuis
    https://agridata.ec.europa.eu/extensions/DashboardApples/ApplesTrade.html#
    """

    def __init__(self, driver_path: str):
        """
        Initialise l'instance avec le chemin du chromedriver.
        """
        self.driver_path = driver_path
        self.browser = None
        self.init_driver()

    def init_driver(self):
        """
        Configure et instancie le webdriver Chrome,
        stocke l'instance dans self.browser.
        """
        chrome_options = webdriver.ChromeOptions()
        download_dir = "/app/data/raw"  # Répertoire de téléchargement
        prefs = {
            "download.default_directory": download_dir,
            "download.prompt_for_download": False,
            "download.directory_upgrade": True,
            "safebrowsing.enabled": True,
        }
        chrome_options.add_experimental_option("prefs", prefs)

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

    def run(self):
        """
        Méthode principale pour exécuter le scraping.
        """
        try:
            url = "https://agridata.ec.europa.eu/extensions/DashboardApples/ApplesTrade.html#"
            print(f"[INFO] Accès à l’URL : {url}")
            self.browser.get(url)

            # Attendre que la page charge complètement
            print("[INFO] Attente pour que la page charge complètement...")
            time.sleep(5)

            # Étape 1 : cliquer sur l'en-tête du menu déroulant
            try:
                menu_dropdown = WebDriverWait(self.browser, 20).until(
                    EC.element_to_be_clickable((By.XPATH, "//span[@style='color:white;']"))
                )
                print("[INFO] Clique sur l'en-tête du menu déroulant...")
                menu_dropdown.click()
                time.sleep(2)
            except TimeoutException:
                print("[ERREUR] Impossible de cliquer sur l'en-tête du menu déroulant.")
                return

            # Étape 2 : cliquer sur "Data Explorer"
            try:
                data_explorer_link = WebDriverWait(self.browser, 20).until(
                    EC.element_to_be_clickable((By.XPATH, "//a[contains(text(),'Data Explorer')]"))
                )
                print("[INFO] Clique sur 'Data Explorer' dans le menu déroulant...")
                data_explorer_link.click()
                time.sleep(5)
            except TimeoutException:
                print("[ERREUR] Impossible de cliquer sur 'Data Explorer'.")
                return

            # Étape 3 : passer dans l'iframe 'iFrm5'
            try:
                iframe = WebDriverWait(self.browser, 20).until(
                    EC.presence_of_element_located((By.ID, "iFrm5"))
                )
                print("[INFO] Passage dans l'iframe 'iFrm5'...")
                self.browser.switch_to.frame(iframe)
                time.sleep(5)
            except TimeoutException:
                print("[ERREUR] Impossible de passer dans l'iframe 'iFrm5'.")
                return

            # Étape 4 : cliquer sur “Bulk download ALL data”
            try:
                bulk_btn = WebDriverWait(self.browser, 20).until(
                    EC.element_to_be_clickable((By.XPATH, "//button[@title='Bulk download ALL data']"))
                )
                print("[INFO] Clique sur 'Bulk Download ALL data'...")
                bulk_btn.click()
            except TimeoutException:
                print("[ERREUR] Impossible de cliquer sur 'Bulk download ALL data'.")
                return

            # Attendre la fin du téléchargement
            print("[INFO] Attente pour le téléchargement...")
            time.sleep(10)

        except Exception as e:
            print(f"[ERREUR] Une erreur est survenue : {e}")

        finally:
            # Toujours fermer le navigateur à la fin
            if self.browser:
                print("[INFO] Fermeture du navigateur.")
                self.browser.quit()


if __name__ == "__main__":
    # Exemple : adapter le chemin vers le driver
    DRIVER_PATH = "/app/driver/chromedriver/chromedriver"
    scraper = AgridataApplesTradeScraper(DRIVER_PATH)
    scraper.run()
