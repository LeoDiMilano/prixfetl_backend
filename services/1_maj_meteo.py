import psycopg2
import pandas as pd
import requests
import time
from datetime import datetime, timedelta
import os


def get_postgres_connection():
    """
    Crée une connexion à la base PostgreSQL.
    """
    try:
        conn = psycopg2.connect(
            dbname="IAFetL",
            user="prixfetl",
            password="Leumces123",
            host="51.83.76.57",
            port="5012"
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
        date_fin = (datetime.utcnow() - timedelta(days=1)).strftime('%Y-%m-%d')

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
                    ANNEE = EXCLUDED.ANNEE,
                    SEMAINE = EXCLUDED.SEMAINE,
                    DIRECTION = EXCLUDED.DIRECTION,
                    RR = EXCLUDED.RR,
                    TM = EXCLUDED.TM,
                    TN = EXCLUDED.TN,
                    TX = EXCLUDED.TX,
                    UN = EXCLUDED.UN,
                    FFM = EXCLUDED.FFM,
                    GLOT = EXCLUDED.GLOT;
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
    API_KEY = "eyJ4NXQiOiJZV0kxTTJZNE1qWTNOemsyTkRZeU5XTTRPV014TXpjek1UVmhNbU14T1RSa09ETXlOVEE0Tnc9PSIsImtpZCI6ImdhdGV3YXlfY2VydGlmaWNhdGVfYWxpYXMiLCJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.eyJzdWIiOiJMR0FMTE9ORUBjYXJib24uc3VwZXIiLCJhcHBsaWNhdGlvbiI6eyJvd25lciI6IkxHQUxMT05FIiwidGllclF1b3RhVHlwZSI6bnVsbCwidGllciI6IlVubGltaXRlZCIsIm5hbWUiOiJEZWZhdWx0QXBwbGljYXRpb24iLCJpZCI6MTk3OTgsInV1aWQiOiJhM2YzNjU2NS03M2NmLTRkYzUtYmE0MS0zZDM4ZjllYWFjNGIifSwiaXNzIjoiaHR0cHM6XC9cL3BvcnRhaWwtYXBpLm1ldGVvZnJhbmNlLmZyOjQ0M1wvb2F1dGgyXC90b2tlbiIsInRpZXJJbmZvIjp7IjUwUGVyTWluIjp7InRpZXJRdW90YVR5cGUiOiJyZXF1ZXN0Q291bnQiLCJncmFwaFFMTWF4Q29tcGxleGl0eSI6MCwiZ3JhcGhRTE1heERlcHRoIjowLCJzdG9wT25RdW90YVJlYWNoIjp0cnVlLCJzcGlrZUFycmVzdExpbWl0IjowLCJzcGlrZUFycmVzdFVuaXQiOiJzZWMifX0sImtleXR5cGUiOiJQUk9EVUNUSU9OIiwic3Vic2NyaWJlZEFQSXMiOlt7InN1YnNjcmliZXJUZW5hbnREb21haW4iOiJjYXJib24uc3VwZXIiLCJuYW1lIjoiQVJPTUUiLCJjb250ZXh0IjoiXC9wdWJsaWNcL2Fyb21lXC8xLjAiLCJwdWJsaXNoZXIiOiJhZG1pbl9tZiIsInZlcnNpb24iOiIxLjAiLCJzdWJzY3JpcHRpb25UaWVyIjoiNTBQZXJNaW4ifSx7InN1YnNjcmliZXJUZW5hbnREb21haW4iOiJjYXJib24uc3VwZXIiLCJuYW1lIjoiQVJQRUdFIiwiY29udGV4dCI6IlwvcHVibGljXC9hcnBlZ2VcLzEuMCIsInB1Ymxpc2hlciI6ImFkbWluX21mIiwidmVyc2lvbiI6IjEuMCIsInN1YnNjcmlwdGlvblRpZXIiOiI1MFBlck1pbiJ9LHsic3Vic2NyaWJlclRlbmFudERvbWFpbiI6ImNhcmJvbi5zdXBlciIsIm5hbWUiOiJEb25uZWVzUHVibGlxdWVzQ2xpbWF0b2xvZ2llIiwiY29udGV4dCI6IlwvcHVibGljXC9EUENsaW1cL3YxIiwicHVibGlzaGVyIjoiYWRtaW5fbWYiLCJ2ZXJzaW9uIjoidjEiLCJzdWJzY3JpcHRpb25UaWVyIjoiNTBQZXJNaW4ifSx7InN1YnNjcmliZXJUZW5hbnREb21haW4iOiJjYXJib24uc3VwZXIiLCJuYW1lIjoiRG9ubmVlc1B1YmxpcXVlc1BhcXVldE9ic2VydmF0aW9uIiwiY29udGV4dCI6IlwvcHVibGljXC9EUFBhcXVldE9ic1wvdjEiLCJwdWJsaXNoZXIiOiJiYXN0aWVuZyIsInZlcnNpb24iOiJ2MSIsInN1YnNjcmlwdGlvblRpZXIiOiI1MFBlck1pbiJ9LHsic3Vic2NyaWJlclRlbmFudERvbWFpbiI6ImNhcmJvbi5zdXBlciIsIm5hbWUiOiJQYXF1ZXRBUk9NRSIsImNvbnRleHQiOiJcL3ByZXZpbnVtXC9EUFBhcXVldEFST01FXC92MSIsInB1Ymxpc2hlciI6ImZyaXNib3VyZyIsInZlcnNpb24iOiJ2MSIsInN1YnNjcmlwdGlvblRpZXIiOiI1MFBlck1pbiJ9LHsic3Vic2NyaWJlclRlbmFudERvbWFpbiI6ImNhcmJvbi5zdXBlciIsIm5hbWUiOiJEb25uZWVzUHVibGlxdWVzT2JzZXJ2YXRpb24iLCJjb250ZXh0IjoiXC9wdWJsaWNcL0RQT2JzXC92MSIsInB1Ymxpc2hlciI6ImJhc3RpZW5nIiwidmVyc2lvbiI6InYxIiwic3Vic2NyaXB0aW9uVGllciI6IjUwUGVyTWluIn1dLCJleHAiOjE3NTIzMDk4NTYsInRva2VuX3R5cGUiOiJhcGlLZXkiLCJpYXQiOjE3MzY3NTc4NTYsImp0aSI6IjQzYjRmNDA1LTA0MzUtNDM1Ny05OGE3LTEyMjIwNzczMDgzMSJ9.HOUsAR5neX0SHyul5OTgxbnor9BAfVrSLHTRMEvJLFhHdhjGcGJ4cN65_Qvpsrr3_ePzT7E9qcTTt9vLV1bMjSJ7UNNEgRTj3JlWShSHG7YrmvlHhqeLRWbOH8_99V_6nuwGQtMcTkZ9kyBuiKPrIR-JcjHlFFNkB6mYCx2Eq8W3evSTCTuckR76wRMaK4UsiWKpZts37nApwzoS7Lnm-ALezKcGO2C7yQqKBmySwP0hI0X9gyFIPEIDlWJdb8xzCBWJRUmlIY7Asq8_2Effq3GWbm2ptTae5nSbTqKInWGqPCzBR5AFjdoe8r6VI7MSYoGPsj8kal2pOQWpbakMkA=="
    maj_meteo_data_ville(2021, API_KEY)
