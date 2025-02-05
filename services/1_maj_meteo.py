import psycopg2
import pandas as pd
import requests
import time
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

# Utilise l'API Données Climatologiques
# https://portail-api.meteofrance.fr/web/fr/api/test/a5935def-80ae-4e7e-83bc-3ef622f0438d/a3f36565-73cf-4dc5-ba41-3d38f9eaac4b

def get_postgres_connection():
    """
    Crée une connexion à la base PostgreSQL.
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
        raise RuntimeError(f"Erreur de connexion à la base PostgreSQL : {e}")

def get_date_range_for_year(annee):
    """
    Retourne les dates de début et de fin pour une année donnée,
    en s'assurant que la date de fin n'est pas dans le futur.
    """
    date_debut = f"{annee}-01-01"
    date_fin = f"{annee}-12-31"

    # Obtenir la date actuelle
    today = datetime.utcnow().strftime('%Y-%m-%d')

    # Si la date de fin dépasse aujourd'hui, utiliser la date actuelle
    if date_fin >= today:
        date_fin = (datetime.utcnow() - timedelta(days=2)).strftime('%Y-%m-%d')

    return date_debut, date_fin


def get_last_meteo_date():
    """
    Se connecte à la base PostgreSQL et retourne la dernière date météo.
    """
    try:
        conn = get_postgres_connection()
        cursor = conn.cursor()

        query = "SELECT MAX(DATE_METEO) FROM METEO_DETAIL"
        cursor.execute(query)
        result = cursor.fetchone()
        
        conn.close()

        if result and result[0]:
            return result[0]  # PostgreSQL retourne directement un objet datetime
        return None
    except psycopg2.Error as e:
        raise RuntimeError(f"Erreur lors de l'accès à la base PostgreSQL : {e}")


def get_meteo_data(id_station, date_debut, date_fin, cle_api):
    """
    Effectue deux appels (commande + récupération) pour obtenir un fichier CSV.
    Retourne le contenu texte du CSV ou None en cas d'erreur.
    """
    headers = {
        'accept': '*/*',
        'apikey': cle_api
    }

    # 1) Commander les données
    url = (
        "https://public-api.meteofrance.fr/public/DPClim/v1/commande-station/quotidienne"
        f"?id-station={id_station}"
        f"&date-deb-periode={date_debut}T00%3A00%3A00Z"
        f"&date-fin-periode={date_fin}T23%3A00%3A00Z"
    )

    response = requests.get(url, headers=headers)
    if response.status_code == 202:
        id_commande = response.json()['elaboreProduitAvecDemandeResponse']['return']
    else:
        print("Erreur lors de la requête (commande) :", response.status_code, response.text)
        return None

    # Attendre que la commande soit prête
    time.sleep(2)

    # 2) Récupérer les résultats
    url_resultat = f"https://public-api.meteofrance.fr/public/DPClim/v1/commande/fichier?id-cmde={id_commande}"
    response = requests.get(url_resultat, headers=headers)
    if response.status_code == 201:
        return response.text
    else:
        print("Erreur lors de la requête (récupération) :", response.status_code, response.text)
        return None


def insert_meteo_data_ville(id_station, annee, direction, cle_api):
    """
    Récupère les données météo pour une station et une année,
    puis les insère dans la base PostgreSQL.
    """
    date_debut, date_fin = get_date_range_for_year(annee)


    data_csv = get_meteo_data(id_station, date_debut, date_fin, cle_api)
    if not data_csv:
        print(f"Erreur lors de la récupération des données météo (station {id_station}, année {annee}).")
        return

    # Convertir le CSV en DataFrame
    data_lines = data_csv.strip().split('\n')
    data_split = [line.split(';') for line in data_lines if line.strip()]
    df = pd.DataFrame(data_split[1:], columns=data_split[0])

    df['DATE'] = pd.to_datetime(df['DATE'], format='%Y%m%d', errors='coerce')
    df['ANNEE'] = df['DATE'].dt.year
    df['SEMAINE'] = df['DATE'].dt.isocalendar().week
    df['DIRECTION'] = direction

    df = df.rename(columns={
        'POSTE': 'ID_STATION',
        'DATE': 'DATE_METEO',
        'rr': 'RR',
        'tn': 'TN',
        'tx': 'TX',
        'tm': 'TM',
        'ffm': 'FFM',
        'un': 'UN',
        'glot': 'GLOT'
    })

    # Convertir les colonnes numériques
    for col in ['RR', 'TM', 'TN', 'TX', 'UN', 'FFM', 'GLOT']:
        if col in df.columns:
            df[col] = df[col].str.replace(',', '.', regex=False).astype(float, errors='ignore')

    # Remplacer les chaînes vides par None
    df = df.replace('', None)

    columns = [
        'DATE_METEO', 'ANNEE', 'SEMAINE', 'DIRECTION',
        'ID_STATION', 'RR', 'TM', 'TN', 'TX', 'UN', 'FFM', 'GLOT'
    ]
    df = df[columns]

    try:
        conn = get_postgres_connection()
        cursor = conn.cursor()

        for _, row in df.iterrows():
            query = """
                INSERT INTO METEO_DETAIL (
                    DATE_METEO, ANNEE, SEMAINE, DIRECTION,
                    ID_STATION, RR, TM, TN, TX, UN, FFM, GLOT
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (DATE_METEO, ID_STATION) DO UPDATE SET
                        RR = EXCLUDED.RR,
                        TM = EXCLUDED.TM,
                        TN = EXCLUDED.TN,
                        TX = EXCLUDED.TX,
                        UN = EXCLUDED.UN,
                        FFM = EXCLUDED.FFM,
                        GLOT = EXCLUDED.GLOT
            """
            cursor.execute(query, (
                row['DATE_METEO'], row['ANNEE'], row['SEMAINE'], row['DIRECTION'],
                row['ID_STATION'], row['RR'], row['TM'], row['TN'], row['TX'],
                row['UN'], row['FFM'], row['GLOT']
            ))

        conn.commit()
        conn.close()
        print(f"Météo insérée/MAJ pour station {id_station}, année {annee}, direction {direction}")
    except psycopg2.Error as e:
        raise RuntimeError(f"Erreur lors de l'insertion dans la base PostgreSQL : {e}")

def maj_meteo_data_ville(annee, cle_api):
    """
    Met à jour la table METEO_DETAIL pour les directions et leurs stations.
    """
    directions = {
        'NE': ['67124001', '69029001'],
        'SE': ['06088001', '13054001'],
        'SO': ['31069001', '33281001'],
        'NO': ['75114001', '35281001']
    }

    for direction, stations in directions.items():
        for station_id in stations:
            insert_meteo_data_ville(station_id, annee, direction, cle_api)
            time.sleep(5)  # Petite pause pour éviter de surcharger l'API


if __name__ == "__main__":
    load_dotenv()
    # récupération de la METEO_API_KEY depuis la variable d'environnement
    METEO_API_KEY = os.getenv("METEO_API_KEY")
    maj_meteo_data_ville(2025, METEO_API_KEY)
