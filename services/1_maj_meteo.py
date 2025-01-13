import sqlite3
import pandas as pd
import requests
import time
from datetime import datetime, timedelta
import os


# ------------------------------------------------
# 1) Déterminer la dernière date stockée
# ------------------------------------------------
def get_last_meteo_date(db_path: str) -> datetime:
    """
    Se connecte à la base de données SQLite,
    retourne la dernière date (DATE_METEO) présente
    dans la table METEO_DETAIL.
    
    :param db_path: Chemin vers la base IAFetL.db
    :return: Un objet datetime représentant la dernière date dans METEO_DETAIL
             ou None si la table est vide.
    :raises FileNotFoundError: Si le fichier de base de données est introuvable.
    :raises sqlite3.Error: Si la table METEO_DETAIL n'existe pas ou autre erreur SQL.
    """
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Base de données introuvable : {db_path}")

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        query = """SELECT MAX(DATE_METEO) FROM METEO_DETAIL"""
        cursor.execute(query)
        result = cursor.fetchone()
        conn.close()

        if result and result[0]:
            # result[0] est du type string "YYYY-MM-DD" en SQLite
            return datetime.strptime(result[0], '%Y-%m-%d')
        else:
            # Table vide, aucune date trouvée
            return None

    except sqlite3.Error as e:
        raise sqlite3.Error(f"Erreur lors de l'accès à la base de données : {e}")


# ------------------------------------------------
# 2) Obtenir les données via l'API Météo France
# ------------------------------------------------
def get_meteo_data(id_station, date_debut, date_fin, cle_api):
    """
    Effectue deux appels (commande + récupération) pour obtenir un fichier CSV.
    Retourne le contenu texte du CSV ou None en cas d'erreur.
    """
    headers = {
        'accept': '*/*',
        'apikey': cle_api  # <-- Au lieu de 'Authorization': f'Bearer {cle_api}'
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
        # Récupérer l'id_commande
        id_commande = response.json()['elaboreProduitAvecDemandeResponse']['return']
    else:
        print("Erreur lors de la requête (commande) :", response.status_code, response.text)
        return None

    # Attendre 1 seconde pour que la commande soit prête
    time.sleep(2)

    # 2) Récupérer les résultats
    url_resultat = f"https://public-api.meteofrance.fr/public/DPClim/v1/commande/fichier?id-cmde={id_commande}"
    response = requests.get(url_resultat, headers=headers)
    
    if response.status_code == 201:
        return response.text  # Contenu CSV
    else:
        print("Erreur lors de la requête (récupération) :", response.status_code, response.text)
        return None


# ------------------------------------------------
# 3) Insérer les données CSV dans METEO_DETAIL
# ------------------------------------------------
def insert_meteo_data_ville(db_path, id_station, annee, direction, cle_api):
    """
    Récupère les données météo CSV (pour la station 'id_station' et l'année 'annee')
    via l'API (cle_api), puis insère/actualise ces données dans METEO_DETAIL.
    """
    date_debut = f'{annee}-01-01'
    date_fin = f'{annee}-12-31'
    # Vérifier l'heure actuelle
    current_time = datetime.utcnow()
    if current_time.hour > 11 or (current_time.hour == 11 and current_time.minute >= 30):
        # Si l'heure actuelle est après 11h30 UTC
        if pd.to_datetime(date_fin) >= pd.to_datetime('today'):
            date_fin = (pd.to_datetime('today') - pd.DateOffset(days=1)).strftime('%Y-%m-%d')
    else:
        # Si l'heure actuelle est avant 11h30 UTC
        if pd.to_datetime(date_fin) >= (pd.to_datetime('today') - pd.DateOffset(days=1)):
            date_fin = (pd.to_datetime('today') - pd.DateOffset(days=2)).strftime('%Y-%m-%d')

    data_csv = get_meteo_data(id_station, date_debut, date_fin, cle_api)
    if data_csv is None:
        print(f"Erreur lors de la récupération des données météo (station {id_station}, année {annee}).")
        return None

    # Le CSV a un entête sur la première ligne, et des lignes séparées par ';'
    # Sauvegarder temporairement dans un fichier (optionnel) ou parser directement
    data_lines = data_csv.strip().split('\n')
    data_split = [line.split(';') for line in data_lines if line.strip()]

    if len(data_split) < 2:
        print("Aucune donnée détectée dans le CSV.")
        return None

    # Convertir en DataFrame
    df = pd.DataFrame(data_split[1:], columns=data_split[0])

    # Convertir la colonne 'DATE' en datetime
    df['DATE'] = pd.to_datetime(df['DATE'], format='%Y%m%d', errors='coerce')
    df['ANNEE'] = df['DATE'].dt.year
    df['SEMAINE'] = df['DATE'].dt.isocalendar().week
    df['DIRECTION'] = direction

    # Renommer les colonnes
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

    # Sélectionner uniquement les colonnes présentes dans METEO_DETAIL
    columns = [
        'DATE_METEO', 'ANNEE', 'SEMAINE', 'DIRECTION',
        'ID_STATION', 'RR', 'TM', 'TN', 'TX', 'UN', 'FFM', 'GLOT'
    ]
    df = df[columns]

    # Connexion à la base
    conn = sqlite3.connect(db_path)

    # Insérer/Mettre à jour les données
    for _, row in df.iterrows():
        query = """
            INSERT INTO METEO_DETAIL (
                DATE_METEO, ANNEE, SEMAINE, DIRECTION,
                ID_STATION, RR, TM, TN, TX, UN, FFM, GLOT
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(DATE_METEO, ID_STATION) DO UPDATE SET
                ANNEE=excluded.ANNEE,
                SEMAINE=excluded.SEMAINE,
                DIRECTION=excluded.DIRECTION,
                RR=excluded.RR,
                TM=excluded.TM,
                TN=excluded.TN,
                TX=excluded.TX,
                UN=excluded.UN,
                FFM=excluded.FFM,
                GLOT=excluded.GLOT;
        """
        conn.execute(query, (
            row['DATE_METEO'].strftime('%Y-%m-%d') if pd.notnull(row['DATE_METEO']) else None,
            int(row['ANNEE']) if pd.notnull(row['ANNEE']) else None,
            int(row['SEMAINE']) if pd.notnull(row['SEMAINE']) else None,
            str(row['DIRECTION']),
            int(float(row['ID_STATION'])) if pd.notnull(row['ID_STATION']) else None,
            float(str(row['RR']).replace(',', '.')) if row['RR'] not in [None, ''] else None,
            float(str(row['TM']).replace(',', '.')) if row['TM'] not in [None, ''] else None,
            float(str(row['TN']).replace(',', '.')) if row['TN'] not in [None, ''] else None,
            float(str(row['TX']).replace(',', '.')) if row['TX'] not in [None, ''] else None,
            float(str(row['UN']).replace(',', '.')) if row['UN'] not in [None, ''] else None,
            float(str(row['FFM']).replace(',', '.')) if row['FFM'] not in [None, ''] else None,
            float(str(row['GLOT']).replace(',', '.')) if row['GLOT'] not in [None, ''] else None
        ))

    conn.commit()
    conn.close()
    print(f"Météo insérée/MAJ pour station {id_station}, année {annee}, direction {direction}")
    return df


# ------------------------------------------------
# 4) Mettre à jour en batch pour les 4 directions
# ------------------------------------------------
def maj_meteo_data_ville(db_path, annee, cle_api):
    """
    Met à jour la table METEO_DETAIL pour 4 directions
    (NE, SE, SO, NO) et leurs stations correspondantes,
    pour l'année spécifiée.
    """
    # Dictionnaire de stations par direction
    directions = {
        'NE': ['67124001', '69029001'],
        'SE': ['06088001', '13054001'],
        'SO': ['31069001', '33281001'],
        'NO': ['75114001', '35281001']
    }

    for direction in directions:
        for station_id in directions[direction]:
            insert_meteo_data_ville(db_path, station_id, annee, direction, cle_api)
            # Petite pause pour éviter de trop charger l'API
            time.sleep(5)

    print("Fin du traitement global des 4 directions.")


# ------------------------------------------------
# Test de la fonction
# ------------------------------------------------
if __name__ == "__main__":
    # Chemin vers la base de données
    DB_PATH = "/app/data/IAFetL.db"
    #https://portail-api.meteofrance.fr/web/fr/api/test/a5935def-80ae-4e7e-83bc-3ef622f0438d/a3f36565-73cf-4dc5-ba41-3d38f9eaac4b
    API_KEY = "eyJ4NXQiOiJZV0kxTTJZNE1qWTNOemsyTkRZeU5XTTRPV014TXpjek1UVmhNbU14T1RSa09ETXlOVEE0Tnc9PSIsImtpZCI6ImdhdGV3YXlfY2VydGlmaWNhdGVfYWxpYXMiLCJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.eyJzdWIiOiJMR0FMTE9ORUBjYXJib24uc3VwZXIiLCJhcHBsaWNhdGlvbiI6eyJvd25lciI6IkxHQUxMT05FIiwidGllclF1b3RhVHlwZSI6bnVsbCwidGllciI6IlVubGltaXRlZCIsIm5hbWUiOiJEZWZhdWx0QXBwbGljYXRpb24iLCJpZCI6MTk3OTgsInV1aWQiOiJhM2YzNjU2NS03M2NmLTRkYzUtYmE0MS0zZDM4ZjllYWFjNGIifSwiaXNzIjoiaHR0cHM6XC9cL3BvcnRhaWwtYXBpLm1ldGVvZnJhbmNlLmZyOjQ0M1wvb2F1dGgyXC90b2tlbiIsInRpZXJJbmZvIjp7IjUwUGVyTWluIjp7InRpZXJRdW90YVR5cGUiOiJyZXF1ZXN0Q291bnQiLCJncmFwaFFMTWF4Q29tcGxleGl0eSI6MCwiZ3JhcGhRTE1heERlcHRoIjowLCJzdG9wT25RdW90YVJlYWNoIjp0cnVlLCJzcGlrZUFycmVzdExpbWl0IjowLCJzcGlrZUFycmVzdFVuaXQiOiJzZWMifX0sImtleXR5cGUiOiJQUk9EVUNUSU9OIiwic3Vic2NyaWJlZEFQSXMiOlt7InN1YnNjcmliZXJUZW5hbnREb21haW4iOiJjYXJib24uc3VwZXIiLCJuYW1lIjoiQVJPTUUiLCJjb250ZXh0IjoiXC9wdWJsaWNcL2Fyb21lXC8xLjAiLCJwdWJsaXNoZXIiOiJhZG1pbl9tZiIsInZlcnNpb24iOiIxLjAiLCJzdWJzY3JpcHRpb25UaWVyIjoiNTBQZXJNaW4ifSx7InN1YnNjcmliZXJUZW5hbnREb21haW4iOiJjYXJib24uc3VwZXIiLCJuYW1lIjoiQVJQRUdFIiwiY29udGV4dCI6IlwvcHVibGljXC9hcnBlZ2VcLzEuMCIsInB1Ymxpc2hlciI6ImFkbWluX21mIiwidmVyc2lvbiI6IjEuMCIsInN1YnNjcmlwdGlvblRpZXIiOiI1MFBlck1pbiJ9LHsic3Vic2NyaWJlclRlbmFudERvbWFpbiI6ImNhcmJvbi5zdXBlciIsIm5hbWUiOiJEb25uZWVzUHVibGlxdWVzQ2xpbWF0b2xvZ2llIiwiY29udGV4dCI6IlwvcHVibGljXC9EUENsaW1cL3YxIiwicHVibGlzaGVyIjoiYWRtaW5fbWYiLCJ2ZXJzaW9uIjoidjEiLCJzdWJzY3JpcHRpb25UaWVyIjoiNTBQZXJNaW4ifSx7InN1YnNjcmliZXJUZW5hbnREb21haW4iOiJjYXJib24uc3VwZXIiLCJuYW1lIjoiRG9ubmVlc1B1YmxpcXVlc1BhcXVldE9ic2VydmF0aW9uIiwiY29udGV4dCI6IlwvcHVibGljXC9EUFBhcXVldE9ic1wvdjEiLCJwdWJsaXNoZXIiOiJiYXN0aWVuZyIsInZlcnNpb24iOiJ2MSIsInN1YnNjcmlwdGlvblRpZXIiOiI1MFBlck1pbiJ9LHsic3Vic2NyaWJlclRlbmFudERvbWFpbiI6ImNhcmJvbi5zdXBlciIsIm5hbWUiOiJQYXF1ZXRBUk9NRSIsImNvbnRleHQiOiJcL3ByZXZpbnVtXC9EUFBhcXVldEFST01FXC92MSIsInB1Ymxpc2hlciI6ImZyaXNib3VyZyIsInZlcnNpb24iOiJ2MSIsInN1YnNjcmlwdGlvblRpZXIiOiI1MFBlck1pbiJ9LHsic3Vic2NyaWJlclRlbmFudERvbWFpbiI6ImNhcmJvbi5zdXBlciIsIm5hbWUiOiJEb25uZWVzUHVibGlxdWVzT2JzZXJ2YXRpb24iLCJjb250ZXh0IjoiXC9wdWJsaWNcL0RQT2JzXC92MSIsInB1Ymxpc2hlciI6ImJhc3RpZW5nIiwidmVyc2lvbiI6InYxIiwic3Vic2NyaXB0aW9uVGllciI6IjUwUGVyTWluIn1dLCJleHAiOjE3NTIzMDk4NTYsInRva2VuX3R5cGUiOiJhcGlLZXkiLCJpYXQiOjE3MzY3NTc4NTYsImp0aSI6IjQzYjRmNDA1LTA0MzUtNDM1Ny05OGE3LTEyMjIwNzczMDgzMSJ9.HOUsAR5neX0SHyul5OTgxbnor9BAfVrSLHTRMEvJLFhHdhjGcGJ4cN65_Qvpsrr3_ePzT7E9qcTTt9vLV1bMjSJ7UNNEgRTj3JlWShSHG7YrmvlHhqeLRWbOH8_99V_6nuwGQtMcTkZ9kyBuiKPrIR-JcjHlFFNkB6mYCx2Eq8W3evSTCTuckR76wRMaK4UsiWKpZts37nApwzoS7Lnm-ALezKcGO2C7yQqKBmySwP0hI0X9gyFIPEIDlWJdb8xzCBWJRUmlIY7Asq8_2Effq3GWbm2ptTae5nSbTqKInWGqPCzBR5AFjdoe8r6VI7MSYoGPsj8kal2pOQWpbakMkA=="

    # Exemple : mettre à jour toutes les données de l'année 2024 pour NE/SE/SO/NO
    maj_meteo_data_ville(DB_PATH, 2025, API_KEY)
