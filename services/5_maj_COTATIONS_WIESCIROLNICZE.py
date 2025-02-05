import os
import psycopg2
import requests
import pandas as pd
from datetime import datetime, timedelta
from bs4 import BeautifulSoup


class CotationWiescirolniczeScraper:
    """
    Cette classe gère la mise à jour des cotations de wiescirolnicze.pl :
    1) Récupère la dernière date en base pour chaque produit.
    2) Trouve tous les lundis et vendredis manquants.
    3) Scrape les données du site et insère dans la base.
    """

    BASE_URL = "https://wiescirolnicze.pl/ceny-rolnicze/{}/?data={}"

    PRODUCTS = [
        "jablka-gala",
        "jablka",
        "jablka-golden-delicius",
        "jablka-jonagold",
        "jablka-jonagored",
        "jablka-ligol",
        "jablka-szampion"
    ]

    def __init__(self):
        self.conn = self.get_postgres_connection()

    @staticmethod
    def get_postgres_connection():
        """
        Retourne une connexion PostgreSQL.
        """
        try:
            conn = psycopg2.connect(
                dbname="IAFetL",
                user="prixfetl",
                password="Leumces123",
                host="prixfetl_postgres",
                port="5432"
            )
            return conn
        except psycopg2.Error as e:
            print(f"Erreur PostgreSQL : {e}")
            raise

    def get_last_date_for_product(self, product):
        """
        Récupère la dernière date enregistrée pour un produit donné.
        """
        with self.conn.cursor() as cursor:
            query = """
            SELECT MAX(date_interrogation) 
            FROM cotations_wiescirolnicze 
            WHERE libelle_produit LIKE %s
            """
            cursor.execute(query, (f"{product}\_%",))
            result = cursor.fetchone()
            #imprime la requete jouée et la date trouvée
            #print(f"SELECT MAX(date_interrogation) FROM cotations_wiescirolnicze WHERE libelle_produit LIKE '{product}\_%'")
            #print(f"Date trouvée : {result[0]}")

            return result[0] if result and result[0] else datetime(2018, 8, 1).date()

    def get_missing_dates(self, last_date):
        """
        Génère les lundis et vendredis manquants entre la dernière date et aujourd'hui.
        """
        today = datetime.today().date()
        dates = []
        current_date = last_date + timedelta(days=1)

        while current_date <= today:
            if current_date.weekday() == 0:  # Lundi
                friday = current_date - timedelta(days=3)
                dates.append((current_date, friday))
            current_date += timedelta(days=1)

        return dates

    def scrape_product_data(self, product, date_vendredi, date_lundi):
        """
        Scrape les données de cotation pour un produit et une date donnée.
        """
        url = self.BASE_URL.format(product, date_vendredi.strftime("%Y-%m-%d"))
        print(f"Scraping {url}...")

        response = requests.get(url)
        
        if response.status_code == 404:
            date_vendredi_alt = date_vendredi - timedelta(days=1)
            url = self.BASE_URL.format(product, date_vendredi_alt.strftime("%Y-%m-%d"))
            response = requests.get(url)
            if response.status_code == 404:
                date_vendredi_alt = date_vendredi - timedelta(days=2)
                url = self.BASE_URL.format(product, date_vendredi_alt.strftime("%Y-%m-%d"))
                response = requests.get(url)
                if response.status_code == 404:
                    date_vendredi_alt = date_vendredi - timedelta(days=3)
                    url = self.BASE_URL.format(product, date_vendredi_alt.strftime("%Y-%m-%d"))
                    response = requests.get(url)
                    if response.status_code == 404:
                        print(f"Aucune donnée trouvée pour {product} à la date {date_vendredi.strftime('%Y-%m-%d')} et {date_vendredi_alt.strftime('%Y-%m-%d')}")
                        return []
            else:
                date_vendredi = date_vendredi_alt
       
        if response.status_code != 200:
            print(f"Erreur {response.status_code} pour {url}")
            return []

        soup = BeautifulSoup(response.text, "html.parser")

        table = soup.find("table", {"data-name": "exchange-prices"})
        if not table:
            print(f"Aucune table trouvée pour {url}")
            return []

        data_rows = []
        for row in table.find("tbody").find_all("tr"):
            columns = row.find_all("td")
            if len(columns) < 6:
                continue

            marche = columns[0].text.strip()
            origine = columns[1].text.strip()
            unite = columns[2].text.strip()
            prix_jour = float(columns[3].text.replace(",", "."))
            date_cotation = datetime.strptime(columns[5].text.strip(), "%d.%m.%Y").date()

            data_rows.append({
                "marche": marche,
                "stade": "Ceny na giełdach",
                "libelle_produit": f"{product}_{origine}",
                "unite": unite,
                "prix_jour": prix_jour,
                "mini": None,
                "maxi": None,
                "date_interrogation": date_lundi,
                "date": date_cotation
            })

        return data_rows

    def insert_data(self, data_rows):
        """
        Insère les données scrappées dans la base de données.
        """
        with self.conn.cursor() as cursor:
            query = """
                INSERT INTO cotations_wiescirolnicze (
                    marche, stade, libelle_produit, unite, prix_jour, mini, maxi,
                    date_interrogation, date
                ) VALUES (%(marche)s, %(stade)s, %(libelle_produit)s, %(unite)s,
                          %(prix_jour)s, %(mini)s, %(maxi)s, 
                          %(date_interrogation)s, %(date)s)
                ON CONFLICT (marche, stade, libelle_produit, date_interrogation)
                DO NOTHING;
            """
            for row in data_rows:
                cursor.execute(query, row)

            self.conn.commit()
            print(f"Insertion de {len(data_rows)} lignes.")

    def run(self):
        """
        Exécute le script complet de mise à jour.
        """
        for product in self.PRODUCTS:
            print(f"\n[INFO] Mise à jour pour {product}")
            last_date = self.get_last_date_for_product(product)
            missing_dates = self.get_missing_dates(last_date)

            for date_lundi, date_vendredi in missing_dates:
                print(f"Scraping {product} pour {date_vendredi} (maj {date_lundi})")
                data_rows = self.scrape_product_data(product, date_vendredi, date_lundi)
                if data_rows:
                    self.insert_data(data_rows)
                else:
                    print(f"Aucune donnée trouvée pour {product} à la date {date_vendredi}")

        self.conn.close()


if __name__ == "__main__":
    scraper = CotationWiescirolniczeScraper()
    scraper.run()
