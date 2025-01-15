import pandas as pd
from datetime import datetime
import psycopg2
from sqlalchemy import create_engine
import requests

def DATE_TO_SEMAINE_SAISON(date):
    date = pd.to_datetime(date)
    if date.week > 31:
        saison = date.year
        semaine = date.week - 31
    else:
        saison = date.year - 1
        semaine = date.week  - 31 + 52        

    return saison, semaine

class VacancesJoursFeries:
    def __init__(self, db_config):
        self.db_config = db_config
        self.conn = self.get_postgres_connection()

    def get_postgres_connection(self):
        try:
            conn = psycopg2.connect(
                host=self.db_config["host"],
                port=self.db_config["port"],
                database=self.db_config["database"],
                user=self.db_config["user"],
                password=self.db_config["password"]
            )
            return conn
        except psycopg2.Error as e:
            print(f"Erreur lors de la connexion à PostgreSQL : {e}")
            raise

    def get_vacances_jours_feries(self, annee):
        # utiliser l'api https://www.data.gouv.fr/fr/dataservices/api-calendrier-scolaire/
        # pour récupérer les données de vacances scolaires
        # convertir annee en annee scolaire
        annee_scolaire = f"{annee}-{annee+1}"
        #  pour chaque zone A, B, C
        # récupérer les données de vacances scolaires
        zones = ["Zone A", "Zone B", "Zone C"]
        vacances = []
        for zone in zones:
            url = f'https://data.education.gouv.fr/api/v2/catalog/datasets/fr-en-calendrier-scolaire/records?where=annee_scolaire=%27{annee_scolaire}%27%20AND%20zones=%27{zone.replace(" ", "%20")}%27'
            response = requests.get(url)
            
            if response.status_code == 200:
                data = response.json()
                for record in data.get("records", []):
                    fields = record.get("record", {}).get("fields", {})
                    start_date = fields.get("start_date", "N/A")
                    end_date = fields.get("end_date", "N/A")
                    annee_scolaire = fields.get("annee_scolaire", "N/A")
                    # Generate all dates between start_date and end_date
                    date_range = pd.date_range(start=start_date, end=end_date)
                    for date in date_range:
                        vacances.append({'DATE': date.strftime('%Y-%m-%d'), 'VACANCES_SCOLAIRES': 1, 'Zone': zone})
            else:
                print(f"Erreur lors de la récupération des vacances scolaires pour la zone {zone} et l'année {annee}")
        vacances = pd.DataFrame(vacances)

        # utiliser l'api https://calendrier.api.gouv.fr/jours-feries/{zone}/{annee}.json
        # pour récupérer les jours fériés
        url = f'https://calendrier.api.gouv.fr/jours-feries/metropole/{annee}.json'
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            jours_feries = []
            for record in data:
                date = record
                nom = data[record]
                # ajouter les données dans jours_feries
                jours_feries.append({'DATE': pd.to_datetime(date), 'JOURS_FERIES': 1})
            jours_feries = pd.DataFrame(jours_feries)
        else:
            print(f"Erreur lors de la récupération des jours fériés pour l'année {annee}")
            return None
        
        # Ajouter les colonnes manquantes dans jours_feries
        jours_feries['VACANCES_SCOLAIRES'] = 0
        jours_feries['Zone'] = None

        # fusionner les deux dataframes
        df = pd.concat([vacances, jours_feries]).drop_duplicates().reset_index(drop=True)
        # remplir les valeurs manquantes
        df['VACANCES_SCOLAIRES'] = df['VACANCES_SCOLAIRES'].fillna(0)
        df['JOURS_FERIES'] = df['JOURS_FERIES'].fillna(0)
        return df

    def create_vacances_jours_feries_dataset(self, start_year, end_year):
        all_data = []
        for year in range(start_year, end_year + 1):
            df_vacances_jours_feries = self.get_vacances_jours_feries(year)
            all_data.append(df_vacances_jours_feries)
        
        # Concaténer toutes les données
        df_vacances_jours_feries = pd.concat(all_data).drop_duplicates().reset_index(drop=True)
        df_vacances_jours_feries['DATE'] = pd.to_datetime(df_vacances_jours_feries['DATE'])
        # supprimer les samedi et dimanches
        df_vacances_jours_feries = df_vacances_jours_feries[df_vacances_jours_feries['DATE'].dt.dayofweek < 5]

        # Ajouter les colonnes SAISON et SEMAINE_SAISON
        df_vacances_jours_feries[['SAISON', 'SEMAINE_SAISON']] = df_vacances_jours_feries['DATE'].apply(lambda x: pd.Series(DATE_TO_SEMAINE_SAISON(x)))

        # Grouper par SAISON et SEMAINE_SAISON pour obtenir le nombre de jours fériés et de jours de vacances scolaires par semaine
        df_vacances_jours_feries_grouped = df_vacances_jours_feries.groupby(['SAISON', 'SEMAINE_SAISON']).agg(
            DATE_INTERROGATION=('DATE', 'min'),
            NB_JOURS_FERIES=('JOURS_FERIES', 'sum'),
            NB_VACANCES_SCOLAIRES=('VACANCES_SCOLAIRES', 'sum')
        ).reset_index()
        
        # Ajouter une colonne calculée INDICATEUR_VACANCES = à NB_JOURS_FERIES / 10 + NB_VACANCES_SCOLAIRES /30 
        df_vacances_jours_feries_grouped['INDICATEUR_VACANCES'] = df_vacances_jours_feries_grouped['NB_JOURS_FERIES'] / 10 + df_vacances_jours_feries_grouped['NB_VACANCES_SCOLAIRES'] / 30
        return df_vacances_jours_feries_grouped

    def insert_data(self, df):
        try:
            engine = create_engine(f"postgresql+psycopg2://{self.db_config['user']}:{self.db_config['password']}@{self.db_config['host']}:{self.db_config['port']}/{self.db_config['database']}")
            df.to_sql('VACANCES_JOURS_FERIES', engine, if_exists='replace', index=False)
        except Exception as e:
            print(f"Erreur lors de l'insertion des données dans PostgreSQL : {e}")
            raise

    def export_to_excel(self, df, filename):
        df.to_excel(filename, index=False)

if __name__ == "__main__":
    db_config = {
        "host": "prixfetl_postgres",
        "port": 5432,
        "database": "IAFetL",
        "user": "prixfetl",
        "password": "Leumces123"
    }

    vacances = VacancesJoursFeries(db_config)
    df_vacances_jours_feries_grouped = vacances.create_vacances_jours_feries_dataset(2019, datetime.now().year)
    vacances.insert_data(df_vacances_jours_feries_grouped)
    vacances.export_to_excel(df_vacances_jours_feries_grouped, 'output_vacances_jours_feries.xlsx')