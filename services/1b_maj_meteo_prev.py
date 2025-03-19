import os
import json
import requests
from dotenv import load_dotenv

def get_meteo_prev_data(lat, lon, cle_api):
    """
    Récupère les prévisions météo sur 4 jours pour les coordonnées indiquées.
    On demande les variables journalières suivantes :
      - precipitation_sum      (précipitations cumulées, équivalent de 'rr')
      - temperature_2m_max     (température maximale, équivalent de 'tx')
      - temperature_2m_min     (température minimale, équivalent de 'tn')
      - shortwave_radiation_sum (irradiation, équivalent de 'glot')
      - maximum_wind_speed_10m (vitesse maximale du vent à 10m, équivalent de 'ffm')
    On pourra calculer 'tm' comme la moyenne de tn et tx ultérieurement si besoin.
    
    Paramètres :
      - lat : latitude (float)
      - lon : longitude (float)
      - cle_api : ta clé API (str)
      
    Retourne le JSON de réponse en cas de succès, ou None en cas d'erreur.
    """
    headers = {
        'accept': '*/*',
        'apikey': cle_api
    }
    params = {
        "latitude": lat,
        "longitude": lon,
        "forecast_days": 4,  # On souhaite 4 jours de prévisions
        "daily": "precipitation_sum,temperature_2m_max,temperature_2m_min,shortwave_radiation_sum,maximum_wind_speed_10m",
        "timezone": "Europe/Paris"
    }
    url = "https://public-api.meteofrance.fr/public/DPPrevi/v1/meteofrance"
    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        print("Erreur lors de la récupération des prévisions :", response.status_code, response.text)
        return None

if __name__ == '__main__':
    load_dotenv()
    METEO_API_KEY = os.getenv("METEO_API_KEY")
    if not METEO_API_KEY:
        print("La variable d'environnement METEO_API_KEY n'est pas définie.")
        exit(1)
        
    # Coordonnées de Paris (par exemple)
    lat = 48.8566
    lon = 2.3522

    data = get_meteo_prev_data(lat, lon, METEO_API_KEY)
    if data:
        print("Données de prévisions récupérées :")
        print(json.dumps(data, indent=4))
    else:
        print("Erreur dans la récupération des données de prévision.")
