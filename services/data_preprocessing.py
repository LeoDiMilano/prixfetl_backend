import psycopg2
import pandas as pd
import warnings
import os
import sys

from dotenv import load_dotenv

load_dotenv()  # charge .env
DATA_RAW_DIR = os.getenv("DATA_RAW_DIR")
DATA_OUTPUT_DIR = os.getenv("DATA_OUTPUT_DIR")

# Désactiver les warnings spécifiques à Pandas
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)


class ApplePriceDataLoader:

    def __init__(self, db_config):
        """
        db_config est un dict contenant les infos de connexion, 
        par ex. :
        {
            "host": "prixfetl_postgres",
            "port": 5432,
            "database": "IAFetL",
            "user": "prixfetl",
            "password": "Leumces123"
        }
        """
        self.db_config = db_config

    def get_postgres_connection(self):
        """
        Retourne une connexion PostgreSQL en utilisant psycopg2 et la config stockée.
        """
        try:
            conn = psycopg2.connect(**self.db_config)
            return conn
        except psycopg2.Error as e:
            print(f"Erreur de connexion à la base PostgreSQL : {e}")
            raise

    def load_prices_dataframe(self):
        """
        Va exécuter la requête SQL et charger les résultats dans un DataFrame pandas.
        (on implémentera le corps de cette méthode plus tard)
        """
        query = """
		WITH COTATIONS AS
		(
			SELECT
				prix_jour,
				libelle_produit,
				stade,
				marche,
				DATE_INTERROGATION,
	            EXTRACT(YEAR FROM DATE_INTERROGATION) AS "ANNEE",
				SEMAINE ,
	            SAISON ,
	            CASE 
	                WHEN TO_CHAR((DATE_INTERROGATION::DATE + INTERVAL '1 day'), 'MM')::INTEGER < 8 
	                THEN (TO_CHAR((DATE_INTERROGATION::DATE + INTERVAL '1 day'), 'MM')::INTEGER + 12 - 7) 
	                ELSE (TO_CHAR((DATE_INTERROGATION::DATE + INTERVAL '1 day'), 'MM')::INTEGER - 7) 
	            END AS "MOIS_SAISON",
	            SEMAINE_SAISON
			FROM cotations_rnm_journalieres c
			UNION ALL
			SELECT 
				prix_jour,
				libelle_produit,
				stade,
				marche,
				DATE_INTERROGATION ,
	            EXTRACT(YEAR FROM DATE_INTERROGATION) AS "ANNEE",
				SEMAINE ,
	            SAISON ,
	            CASE 
	                WHEN TO_CHAR((DATE_INTERROGATION::DATE + INTERVAL '1 day'), 'MM')::INTEGER < 8 
	                THEN (TO_CHAR((DATE_INTERROGATION::DATE + INTERVAL '1 day'), 'MM')::INTEGER + 12 - 7) 
	                ELSE (TO_CHAR((DATE_INTERROGATION::DATE + INTERVAL '1 day'), 'MM')::INTEGER - 7) 
	            END AS "MOIS_SAISON",
	            SEMAINE_SAISON AS "SEMAINE_SAISON"
			FROM cotations_wiescirolnicze w
			WHERE UNITE = '1 kg'
			
		)


		SELECT 
            c.DATE_INTERROGATION AS "DATE_INTERROGATION",
            EXTRACT(YEAR FROM c.DATE_INTERROGATION) AS "ANNEE",
			c.SEMAINE AS "SEMAINE",
            c.SAISON AS "SAISON",
            CASE 
                WHEN TO_CHAR((c.DATE_INTERROGATION::DATE + INTERVAL '1 day'), 'MM')::INTEGER < 8 
                THEN (TO_CHAR((c.DATE_INTERROGATION::DATE + INTERVAL '1 day'), 'MM')::INTEGER + 12 - 7) 
                ELSE (TO_CHAR((c.DATE_INTERROGATION::DATE + INTERVAL '1 day'), 'MM')::INTEGER - 7) 
            END AS "MOIS_SAISON",
            c.SEMAINE_SAISON AS "SEMAINE_SAISON",
            CASE WHEN (b.INDICATEUR_VACANCES IS NULL) THEN 0 ELSE b.INDICATEUR_VACANCES END AS "VACANCES_INDICATEUR_S",
			
			AVG(CASE WHEN p.produit_groupe = 'DET BANANE AFRIQUE-AMERIQUE' THEN c.prix_jour END) AS "PRIX DET BANANE AFRIQUE-AMERIQUE",
			AVG(CASE WHEN p.produit_groupe = 'DET BANANE FRANCE' THEN c.prix_jour END) AS "PRIX DET BANANE FRANCE",
			AVG(CASE WHEN p.produit_groupe = 'DET KIWI VERT  FRANCE' THEN c.prix_jour END) AS "PRIX DET KIWI VERT  FRANCE",
			AVG(CASE WHEN p.produit_groupe = 'DET KIWI VERT  HORS FR' THEN c.prix_jour END) AS "PRIX DET KIWI VERT  HORS FR",
			AVG(CASE WHEN p.produit_groupe = 'DET ORANGE HORS' THEN c.prix_jour END) AS "PRIX DET ORANGE HORS",
			AVG(CASE WHEN p.produit_groupe = 'DET POMME GALA FRANCE 136/180G  VRAC' THEN c.prix_jour END) AS "PRIX DET POMME GALA FRANCE 136/180G  VRAC",
			AVG(CASE WHEN p.produit_groupe = 'DET POMME GALA FRANCE +170G  VRAC' THEN c.prix_jour END) AS "PRIX DET POMME GALA FRANCE +170G  VRAC",
			AVG(CASE WHEN p.produit_groupe = 'DET POMME GOLDEN FRANCE 136/180G  VRAC' THEN c.prix_jour END) AS "PRIX DET POMME GOLDEN FRANCE 136/180G  VRAC",
			AVG(CASE WHEN p.produit_groupe = 'DET POMME GOLDEN FRANCE +170G  VRAC' THEN c.prix_jour END) AS "PRIX DET POMME GOLDEN FRANCE +170G  VRAC",
			AVG(CASE WHEN p.produit_groupe = 'DET POMME GRANNY FRANCE 136/180G  VRAC' THEN c.prix_jour END) AS "PRIX DET POMME GRANNY FRANCE 136/180G  VRAC",
			AVG(CASE WHEN p.produit_groupe = 'DET POMME GRANNY FRANCE +170G  VRAC' THEN c.prix_jour END) AS "PRIX DET POMME GRANNY FRANCE +170G  VRAC",
			AVG(CASE WHEN p.produit_groupe = 'EXP CLÉMENTINE CORSE CAT.I' THEN c.prix_jour END) AS "PRIX EXP CLÉMENTINE CORSE CAT.I",
			AVG(CASE WHEN p.produit_groupe = 'EXP KIWI HAYWARD 105-115G' THEN c.prix_jour END) AS "PRIX EXP KIWI HAYWARD 105-115G",
			AVG(CASE WHEN p.produit_groupe = 'EXP KIWI HAYWARD 115-125G' THEN c.prix_jour END) AS "PRIX EXP KIWI HAYWARD 115-125G",
			AVG(CASE WHEN p.produit_groupe = 'EXP KIWI HAYWARD 125-135G' THEN c.prix_jour END) AS "PRIX EXP KIWI HAYWARD 125-135G",
			AVG(CASE WHEN p.produit_groupe = 'EXP NECTARINE BLANCHE A PLATEAU 1RG' THEN c.prix_jour END) AS "PRIX EXP NECTARINE BLANCHE A PLATEAU 1RG",
			AVG(CASE WHEN p.produit_groupe = 'EXP NECTARINE BLANCHE B PLATEAU 1RG' THEN c.prix_jour END) AS "PRIX EXP NECTARINE BLANCHE B PLATEAU 1RG",
			AVG(CASE WHEN p.produit_groupe = 'EXP NECTARINE JAUNE A PLATEAU 1RG' THEN c.prix_jour END) AS "PRIX EXP NECTARINE JAUNE A PLATEAU 1RG",
			AVG(CASE WHEN p.produit_groupe = 'EXP NECTARINE JAUNE B PLATEAU 1RG' THEN c.prix_jour END) AS "PRIX EXP NECTARINE JAUNE B PLATEAU 1RG",
			AVG(CASE WHEN p.produit_groupe = 'EXP PÊCHE BLANCHE A PLATEAU 1RG' THEN c.prix_jour END) AS "PRIX EXP PÊCHE BLANCHE A PLATEAU 1RG",
			AVG(CASE WHEN p.produit_groupe = 'EXP PÊCHE BLANCHE B PLATEAU 1RG' THEN c.prix_jour END) AS "PRIX EXP PÊCHE BLANCHE B PLATEAU 1RG",
			AVG(CASE WHEN p.produit_groupe = 'EXP PÊCHE JAUNE A PLATEAU 1RG' THEN c.prix_jour END) AS "PRIX EXP PÊCHE JAUNE A PLATEAU 1RG",
			AVG(CASE WHEN p.produit_groupe = 'EXP PÊCHE JAUNE B PLATEAU 1RG' THEN c.prix_jour END) AS "PRIX EXP PÊCHE JAUNE B PLATEAU 1RG",
			AVG(CASE WHEN p.produit_groupe = 'EXP POMME BRAEBURN FRANCE 170/220G CAT.I PLATEAU 1RG' THEN c.prix_jour END) AS "PRIX EXP POMME BRAEBURN FRANCE 170/220G CAT.I PLATEAU 1RG",
			AVG(CASE WHEN p.produit_groupe = 'EXP POMME ELSTAR FRANCE 170/220G CAT.I PLATEAU 1RG' THEN c.prix_jour END) AS "PRIX EXP POMME ELSTAR FRANCE 170/220G CAT.I PLATEAU 1RG",
			AVG(CASE WHEN p.produit_groupe = 'EXP POMME FUJI FRANCE 170/220G CAT.I PLATEAU 1RG' THEN c.prix_jour END) AS "PRIX EXP POMME FUJI FRANCE 170/220G CAT.I PLATEAU 1RG",
			AVG(CASE WHEN p.produit_groupe = 'EXP POMME FUJI FRANCE 201/270G CAT.I PLATEAU 1RG' THEN c.prix_jour END) AS "PRIX EXP POMME FUJI FRANCE 201/270G CAT.I PLATEAU 1RG",
			AVG(CASE WHEN p.produit_groupe = 'EXP POMME GALA FRANCE 100-113 CAT.I BUSHEL' THEN c.prix_jour END) AS "PRIX EXP POMME GALA FRANCE 100-113 CAT.I BUSHEL",
			AVG(CASE WHEN p.produit_groupe = 'EXP POMME GALA FRANCE 115/150G CAT.I SACHET' THEN c.prix_jour END) AS "PRIX EXP POMME GALA FRANCE 115/150G CAT.I SACHET",
			AVG(CASE WHEN p.produit_groupe = 'EXP POMME GALA FRANCE 125-138 CAT.I BUSHEL' THEN c.prix_jour END) AS "PRIX EXP POMME GALA FRANCE 125-138 CAT.I BUSHEL",
			AVG(CASE WHEN p.produit_groupe = 'EXP POMME GALA FRANCE 136/200G CAT.I CAISSE' THEN c.prix_jour END) AS "PRIX EXP POMME GALA FRANCE 136/200G CAT.I CAISSE",
			AVG(CASE WHEN p.produit_groupe = 'EXP POMME GALA FRANCE 138-150 CAT.I BUSHEL' THEN c.prix_jour END) AS "PRIX EXP POMME GALA FRANCE 138-150 CAT.I BUSHEL",
			AVG(CASE WHEN p.produit_groupe = 'EXP POMME GALA FRANCE 150/180G CAT.I PLATEAU 2RG' THEN c.prix_jour END) AS "PRIX EXP POMME GALA FRANCE 150/180G CAT.I PLATEAU 2RG",
			AVG(CASE WHEN p.produit_groupe = 'EXP POMME GALA FRANCE 170/220G CAT.I PLATEAU 1RG' THEN c.prix_jour END) AS "PRIX EXP POMME GALA FRANCE 170/220G CAT.I PLATEAU 1RG",
			AVG(CASE WHEN p.produit_groupe = 'EXP POMME GALA FRANCE 201/270G CAT.I PLATEAU 1RG' THEN c.prix_jour END) AS "PRIX EXP POMME GALA FRANCE 201/270G CAT.I PLATEAU 1RG",
			AVG(CASE WHEN p.produit_groupe = 'EXP POMME GOLDEN FRANCE 115/150G CAT.I SACHET' THEN c.prix_jour END) AS "PRIX EXP POMME GOLDEN FRANCE 115/150G CAT.I SACHET",
			AVG(CASE WHEN p.produit_groupe = 'EXP POMME GOLDEN FRANCE 136/200G CAT.I CAISSE' THEN c.prix_jour END) AS "PRIX EXP POMME GOLDEN FRANCE 136/200G CAT.I CAISSE",
			AVG(CASE WHEN p.produit_groupe = 'EXP POMME GOLDEN FRANCE 150/180G CAT.I PLATEAU 2RG' THEN c.prix_jour END) AS "PRIX EXP POMME GOLDEN FRANCE 150/180G CAT.I PLATEAU 2RG",
			AVG(CASE WHEN p.produit_groupe = 'EXP POMME GOLDEN FRANCE 170/220G CAT.I PLATEAU 1RG' THEN c.prix_jour END) AS "PRIX EXP POMME GOLDEN FRANCE 170/220G CAT.I PLATEAU 1RG",
			AVG(CASE WHEN p.produit_groupe = 'EXP POMME GOLDEN FRANCE 201/270G CAT.I PLATEAU 1RG' THEN c.prix_jour END) AS "PRIX EXP POMME GOLDEN FRANCE 201/270G CAT.I PLATEAU 1RG",
			AVG(CASE WHEN p.produit_groupe = 'EXP POMME GRANNY FRANCE 100-113 CAT.I BUSHEL' THEN c.prix_jour END) AS "PRIX EXP POMME GRANNY FRANCE 100-113 CAT.I BUSHEL",
			AVG(CASE WHEN p.produit_groupe = 'EXP POMME GRANNY FRANCE 125-138 CAT.I BUSHEL' THEN c.prix_jour END) AS "PRIX EXP POMME GRANNY FRANCE 125-138 CAT.I BUSHEL",
			AVG(CASE WHEN p.produit_groupe = 'EXP POMME GRANNY FRANCE 138-150 CAT.I BUSHEL' THEN c.prix_jour END) AS "PRIX EXP POMME GRANNY FRANCE 138-150 CAT.I BUSHEL",
			AVG(CASE WHEN p.produit_groupe = 'EXP POMME GRANNY FRANCE 150/180G CAT.I PLATEAU 2RG' THEN c.prix_jour END) AS "PRIX EXP POMME GRANNY FRANCE 150/180G CAT.I PLATEAU 2RG",
			AVG(CASE WHEN p.produit_groupe = 'EXP POMME GRANNY FRANCE 170/220G CAT.I PLATEAU 1RG' THEN c.prix_jour END) AS "PRIX EXP POMME GRANNY FRANCE 170/220G CAT.I PLATEAU 1RG",
			AVG(CASE WHEN p.produit_groupe = 'EXP POMME GRANNY FRANCE 201/270G CAT.I PLATEAU 1RG' THEN c.prix_jour END) AS "PRIX EXP POMME GRANNY FRANCE 201/270G CAT.I PLATEAU 1RG",
			AVG(CASE WHEN p.produit_groupe = 'GRO BANANE AFRIQUE-AMERIQUE RUNGIS' THEN c.prix_jour END) AS "PRIX GRO BANANE AFRIQUE-AMERIQUE RUNGIS",
			AVG(CASE WHEN p.produit_groupe = 'GRO BANANE DOM RUNGIS' THEN c.prix_jour END) AS "PRIX GRO BANANE DOM RUNGIS",
			AVG(CASE WHEN p.produit_groupe = 'GRO KIWI GOLD  COLIS IT MARCHE DE' THEN c.prix_jour END) AS "PRIX GRO KIWI GOLD  COLIS IT MARCHE DE",
			AVG(CASE WHEN p.produit_groupe = 'GRO KIWI GOLD  COLIS NZ MARCHE DE' THEN c.prix_jour END) AS "PRIX GRO KIWI GOLD  COLIS NZ MARCHE DE",
			AVG(CASE WHEN p.produit_groupe = 'GRO KIWI HAYWARD  COLIS IT MARCHE DE' THEN c.prix_jour END) AS "PRIX GRO KIWI HAYWARD  COLIS IT MARCHE DE",
			AVG(CASE WHEN p.produit_groupe = 'GRO NECTARINE BLANCHE A ESPAGNE' THEN c.prix_jour END) AS "PRIX GRO NECTARINE BLANCHE A ESPAGNE",
			AVG(CASE WHEN p.produit_groupe = 'GRO NECTARINE JAUNE A ESPAGNE' THEN c.prix_jour END) AS "PRIX GRO NECTARINE JAUNE A ESPAGNE",
			AVG(CASE WHEN p.produit_groupe = 'GRO ORANGE U.E. BIO' THEN c.prix_jour END) AS "PRIX GRO ORANGE U.E. BIO",
			AVG(CASE WHEN p.produit_groupe = 'GRO PÊCHE BLANCHE A ESPAGNE' THEN c.prix_jour END) AS "PRIX GRO PÊCHE BLANCHE A ESPAGNE",
			AVG(CASE WHEN p.produit_groupe = 'GRO PÊCHE JAUNE A ESPAGNE' THEN c.prix_jour END) AS "PRIX GRO PÊCHE JAUNE A ESPAGNE",
			AVG(CASE WHEN p.produit_groupe = 'GRO POMME BOSKOOP ALLEMAGNE 70-80MM CAT.I CAISSE' THEN c.prix_jour END) AS "PRIX GRO POMME BOSKOOP ALLEMAGNE 70-80MM CAT.I CAISSE",
			AVG(CASE WHEN p.produit_groupe = 'GRO POMME BOSKOOP FRANCE 170/220G CAT.I PLATEAU 1RG' THEN c.prix_jour END) AS "PRIX GRO POMME BOSKOOP FRANCE 170/220G CAT.I PLATEAU 1RG",
			AVG(CASE WHEN p.produit_groupe = 'GRO POMME BRAEBURN ALLEMAGNE 70-80MM CAT.I CAISSE' THEN c.prix_jour END) AS "PRIX GRO POMME BRAEBURN ALLEMAGNE 70-80MM CAT.I CAISSE",
			AVG(CASE WHEN p.produit_groupe = 'GRO POMME ELSTAR ALLEMAGNE 65-70MM CAT.I CAISSE' THEN c.prix_jour END) AS "PRIX GRO POMME ELSTAR ALLEMAGNE 65-70MM CAT.I CAISSE",
			AVG(CASE WHEN p.produit_groupe = 'GRO POMME ELSTAR ALLEMAGNE 70-80MM CAT.I CAISSE' THEN c.prix_jour END) AS "PRIX GRO POMME ELSTAR ALLEMAGNE 70-80MM CAT.I CAISSE",
			AVG(CASE WHEN p.produit_groupe = 'GRO POMME ELSTAR FRANCE 170/220G CAT.I PLATEAU 1RG' THEN c.prix_jour END) AS "PRIX GRO POMME ELSTAR FRANCE 170/220G CAT.I PLATEAU 1RG",
			AVG(CASE WHEN p.produit_groupe = 'GRO POMME FUJI FRANCE 201/270G CAT.I PLATEAU 1RG' THEN c.prix_jour END) AS "PRIX GRO POMME FUJI FRANCE 201/270G CAT.I PLATEAU 1RG",
			AVG(CASE WHEN p.produit_groupe = 'GRO POMME FUJI FRANCE 201/270G CAT.I PLATEAU' THEN c.prix_jour END) AS "PRIX GRO POMME FUJI FRANCE 201/270G CAT.I PLATEAU",
			AVG(CASE WHEN p.produit_groupe = 'GRO POMME GALA ALLEMAGNE 70-80MM CAT.I CAISSE' THEN c.prix_jour END) AS "PRIX GRO POMME GALA ALLEMAGNE 70-80MM CAT.I CAISSE",
			AVG(CASE WHEN p.produit_groupe = 'GRO POMME GALA FRANCE 115/150G CAT.I CAISSE' THEN c.prix_jour END) AS "PRIX GRO POMME GALA FRANCE 115/150G CAT.I CAISSE",
			AVG(CASE WHEN p.produit_groupe = 'GRO POMME GALA FRANCE 136/200G CAT.I CAISSE' THEN c.prix_jour END) AS "PRIX GRO POMME GALA FRANCE 136/200G CAT.I CAISSE",
			AVG(CASE WHEN p.produit_groupe = 'GRO POMME GALA FRANCE 150/180G CAT.I PLATEAU 2RG' THEN c.prix_jour END) AS "PRIX GRO POMME GALA FRANCE 150/180G CAT.I PLATEAU 2RG",
			AVG(CASE WHEN p.produit_groupe = 'GRO POMME GALA FRANCE 170/220G CAT.I PLATEAU 1RG' THEN c.prix_jour END) AS "PRIX GRO POMME GALA FRANCE 170/220G CAT.I PLATEAU 1RG",
			AVG(CASE WHEN p.produit_groupe = 'GRO POMME GALA FRANCE 170/220G CAT.I PLATEAU 2RG' THEN c.prix_jour END) AS "PRIX GRO POMME GALA FRANCE 170/220G CAT.I PLATEAU 2RG",
			AVG(CASE WHEN p.produit_groupe = 'GRO POMME GALA FRANCE 201/270G CAT.I PLATEAU 1RG' THEN c.prix_jour END) AS "PRIX GRO POMME GALA FRANCE 201/270G CAT.I PLATEAU 1RG",
			AVG(CASE WHEN p.produit_groupe = 'GRO POMME GALA FRANCE 201/270G CAT.I PLATEAU' THEN c.prix_jour END) AS "PRIX GRO POMME GALA FRANCE 201/270G CAT.I PLATEAU",
			AVG(CASE WHEN p.produit_groupe = 'GRO POMME GALA ITALIE 70-80MM CAT.I CAISSE' THEN c.prix_jour END) AS "PRIX GRO POMME GALA ITALIE 70-80MM CAT.I CAISSE",
			AVG(CASE WHEN p.produit_groupe = 'GRO POMME GOLDEN  170/220G CAT.I PLATEAU 2RG' THEN c.prix_jour END) AS "PRIX GRO POMME GOLDEN  170/220G CAT.I PLATEAU 2RG",
			AVG(CASE WHEN p.produit_groupe = 'GRO POMME GOLDEN FRANCE 115/150G CAT.I CAISSE' THEN c.prix_jour END) AS "PRIX GRO POMME GOLDEN FRANCE 115/150G CAT.I CAISSE",
			AVG(CASE WHEN p.produit_groupe = 'GRO POMME GOLDEN FRANCE 136/200G CAT.I CAISSE' THEN c.prix_jour END) AS "PRIX GRO POMME GOLDEN FRANCE 136/200G CAT.I CAISSE",
			AVG(CASE WHEN p.produit_groupe = 'GRO POMME GOLDEN FRANCE 150/180G CAT.I PLATEAU 2RG' THEN c.prix_jour END) AS "PRIX GRO POMME GOLDEN FRANCE 150/180G CAT.I PLATEAU 2RG",
			AVG(CASE WHEN p.produit_groupe = 'GRO POMME GOLDEN FRANCE 170/220G CAT.I PLATEAU 1RG' THEN c.prix_jour END) AS "PRIX GRO POMME GOLDEN FRANCE 170/220G CAT.I PLATEAU 1RG",
			AVG(CASE WHEN p.produit_groupe = 'GRO POMME GOLDEN FRANCE 170/220G CAT.I PLATEAU 2RG' THEN c.prix_jour END) AS "PRIX GRO POMME GOLDEN FRANCE 170/220G CAT.I PLATEAU 2RG",
			AVG(CASE WHEN p.produit_groupe = 'GRO POMME GOLDEN FRANCE 201/270G CAT.I PLATEAU 1RG' THEN c.prix_jour END) AS "PRIX GRO POMME GOLDEN FRANCE 201/270G CAT.I PLATEAU 1RG",
			AVG(CASE WHEN p.produit_groupe = 'GRO POMME GOLDEN FRANCE 201/270G CAT.I PLATEAU' THEN c.prix_jour END) AS "PRIX GRO POMME GOLDEN FRANCE 201/270G CAT.I PLATEAU",
			AVG(CASE WHEN p.produit_groupe = 'GRO POMME GOLDEN FRANCE COLO.1-2 CAT.I PLATEAU 1RG' THEN c.prix_jour END) AS "PRIX GRO POMME GOLDEN FRANCE COLO.1-2 CAT.I PLATEAU 1RG",
			AVG(CASE WHEN p.produit_groupe = 'GRO POMME GOLDEN ITALIE 70-75MM CAT.I PLATEAU' THEN c.prix_jour END) AS "PRIX GRO POMME GOLDEN ITALIE 70-75MM CAT.I PLATEAU",
			AVG(CASE WHEN p.produit_groupe = 'GRO POMME GRANNY FRANCE 150/180G CAT.I PLATEAU 2RG' THEN c.prix_jour END) AS "PRIX GRO POMME GRANNY FRANCE 150/180G CAT.I PLATEAU 2RG",
			AVG(CASE WHEN p.produit_groupe = 'GRO POMME GRANNY FRANCE 170/220G CAT.I PLATEAU 1RG' THEN c.prix_jour END) AS "PRIX GRO POMME GRANNY FRANCE 170/220G CAT.I PLATEAU 1RG",
			AVG(CASE WHEN p.produit_groupe = 'GRO POMME GRANNY FRANCE 170/220G CAT.I PLATEAU 2RG' THEN c.prix_jour END) AS "PRIX GRO POMME GRANNY FRANCE 170/220G CAT.I PLATEAU 2RG",
			AVG(CASE WHEN p.produit_groupe = 'GRO POMME GRANNY FRANCE 201/270G CAT.I PLATEAU 1RG' THEN c.prix_jour END) AS "PRIX GRO POMME GRANNY FRANCE 201/270G CAT.I PLATEAU 1RG",
			AVG(CASE WHEN p.produit_groupe = 'GRO POMME GRANNY ITALIE 70-75MM CAT.I PLATEAU' THEN c.prix_jour END) AS "PRIX GRO POMME GRANNY ITALIE 70-75MM CAT.I PLATEAU",
			AVG(CASE WHEN p.produit_groupe = 'GRO POMME JONAGOLD ALLEMAGNE 70-80MM CAT.I CAISSE' THEN c.prix_jour END) AS "PRIX GRO POMME JONAGOLD ALLEMAGNE 70-80MM CAT.I CAISSE",
			AVG(CASE WHEN p.produit_groupe = 'GRO POMME PINK-LADY  201/270G CAT.I PLATEAU 1RG' THEN c.prix_jour END) AS "PRIX GRO POMME PINK-LADY  201/270G CAT.I PLATEAU 1RG",
			AVG(CASE WHEN p.produit_groupe = 'GRO POMME PINK-LADY  80-85MM CAT.I PLATEAU 1RG' THEN c.prix_jour END) AS "PRIX GRO POMME PINK-LADY  80-85MM CAT.I PLATEAU 1RG",
			AVG(CASE WHEN p.produit_groupe = 'GRO POMME PINK-LADY FRANCE 201/270G CAT.I PLATEAU 1RG' THEN c.prix_jour END) AS "PRIX GRO POMME PINK-LADY FRANCE 201/270G CAT.I PLATEAU 1RG",
			AVG(CASE WHEN p.produit_groupe = 'GRO POMME PINK-LADY FRANCE 70-75MM CAT.I PLATEAU' THEN c.prix_jour END) AS "PRIX GRO POMME PINK-LADY FRANCE 70-75MM CAT.I PLATEAU",
			AVG(CASE WHEN p.produit_groupe = 'GRO POMME PINK-LADY ITALIE 70-75MM CAT.I PLATEAU' THEN c.prix_jour END) AS "PRIX GRO POMME PINK-LADY ITALIE 70-75MM CAT.I PLATEAU",
			AVG(CASE WHEN p.produit_groupe = 'GRO POMME ROUGE FRANCE 201/270G CAT.I PLATEAU 1RG' THEN c.prix_jour END) AS "PRIX GRO POMME ROUGE FRANCE 201/270G CAT.I PLATEAU 1RG",
			AVG(CASE WHEN p.produit_groupe = 'GRO POMME ROUGE ITALIE 70-80MM CAT.I CAISSE' THEN c.prix_jour END) AS "PRIX GRO POMME ROUGE ITALIE 70-80MM CAT.I CAISSE",
			AVG(CASE WHEN p.produit_groupe = 'IMP BANANE GUADELOUPE' THEN c.prix_jour END) AS "PRIX IMP BANANE GUADELOUPE",
			AVG(CASE WHEN p.produit_groupe = 'IMP CLÉMENTINE ESPAGNE-MAROC CAT.I 1(63-74MM)' THEN c.prix_jour END) AS "PRIX IMP CLÉMENTINE ESPAGNE-MAROC CAT.I 1(63-74MM)",
			AVG(CASE WHEN p.produit_groupe = 'IMP CLÉMENTINE ESPAGNE-MAROC CAT.I 2(58-69MM)' THEN c.prix_jour END) AS "PRIX IMP CLÉMENTINE ESPAGNE-MAROC CAT.I 2(58-69MM)",
			AVG(CASE WHEN p.produit_groupe = 'IMP CLÉMENTINE HAUT CAT.I 1(63-74MM)' THEN c.prix_jour END) AS "PRIX IMP CLÉMENTINE HAUT CAT.I 1(63-74MM)",
			AVG(CASE WHEN p.produit_groupe = 'IMP CLÉMENTINE HAUT CAT.I 2(58-69MM)' THEN c.prix_jour END) AS "PRIX IMP CLÉMENTINE HAUT CAT.I 2(58-69MM)",
			AVG(CASE WHEN p.produit_groupe = 'IMP ORANGE ESPAGNE CAT.I 3(81-92MM)' THEN c.prix_jour END) AS "PRIX IMP ORANGE ESPAGNE CAT.I 3(81-92MM)",
			AVG(CASE WHEN p.produit_groupe = 'IMP ORANGE ESPAGNE CAT.I 4(77-88MM)' THEN c.prix_jour END) AS "PRIX IMP ORANGE ESPAGNE CAT.I 4(77-88MM)",
			AVG(CASE WHEN p.produit_groupe = 'PRO KIWI ROUSSILLON 105-115G' THEN c.prix_jour END) AS "PRIX PRO KIWI ROUSSILLON 105-115G",
			AVG(CASE WHEN p.produit_groupe = 'PRO KIWI ROUSSILLON 115-125G' THEN c.prix_jour END) AS "PRIX PRO KIWI ROUSSILLON 115-125G",
			AVG(CASE WHEN p.produit_groupe = 'PRO KIWI ROUSSILLON 95-105G' THEN c.prix_jour END) AS "PRIX PRO KIWI ROUSSILLON 95-105G",
			AVG(CASE WHEN p.produit_groupe = 'PRO POMME GALA FRANCE 136/200G CAT.I CAISSE' THEN c.prix_jour END) AS "PRIX PRO POMME GALA FRANCE 136/200G CAT.I CAISSE",
			AVG(CASE WHEN p.produit_groupe = 'PRO POMME GALA FRANCE 170/220G CAT.I PLATEAU 2RG' THEN c.prix_jour END) AS "PRIX PRO POMME GALA FRANCE 170/220G CAT.I PLATEAU 2RG",
			AVG(CASE WHEN p.produit_groupe = 'PRO POMME GALA FRANCE 201/270G CAT.I PLATEAU 1RG' THEN c.prix_jour END) AS "PRIX PRO POMME GALA FRANCE 201/270G CAT.I PLATEAU 1RG",
			AVG(CASE WHEN p.produit_groupe = 'PRO POMME GOLDEN FRANCE 136/200G CAT.I CAISSE' THEN c.prix_jour END) AS "PRIX PRO POMME GOLDEN FRANCE 136/200G CAT.I CAISSE",
			AVG(CASE WHEN p.produit_groupe = 'PRO POMME GOLDEN FRANCE 170/220G CAT.I PLATEAU 2RG' THEN c.prix_jour END) AS "PRIX PRO POMME GOLDEN FRANCE 170/220G CAT.I PLATEAU 2RG",
			AVG(CASE WHEN p.produit_groupe = 'PRO POMME GOLDEN FRANCE 201/270G CAT.I PLATEAU 1RG' THEN c.prix_jour END) AS "PRIX PRO POMME GOLDEN FRANCE 201/270G CAT.I PLATEAU 1RG",

			AVG(CASE WHEN p.produit_groupe = 'MARCHE_PL POMME GALA PL' THEN c.prix_jour END) AS "PRIX MARCHE_PL POMME GALA PL",
			AVG(CASE WHEN p.produit_groupe = 'MARCHE_PL POMME GOLDEN-DELICIUS PL' THEN c.prix_jour END) AS "PRIX MARCHE_PL POMME GOLDEN-DELICIUS PL",
			AVG(CASE WHEN p.produit_groupe = 'MARCHE_PL POMME JABLKA PL' THEN c.prix_jour END) AS "PRIX MARCHE_PL POMME JABLKA PL",
			AVG(CASE WHEN p.produit_groupe = 'MARCHE_PL POMME JONAGOLD PL' THEN c.prix_jour END) AS "PRIX MARCHE_PL POMME JONAGOLD PL",
			AVG(CASE WHEN p.produit_groupe = 'MARCHE_PL POMME JONAGORED PL' THEN c.prix_jour END) AS "PRIX MARCHE_PL POMME JONAGORED PL",
			AVG(CASE WHEN p.produit_groupe = 'MARCHE_PL POMME LIGOL PL' THEN c.prix_jour END) AS "PRIX MARCHE_PL POMME LIGOL PL",
			AVG(CASE WHEN p.produit_groupe = 'MARCHE_PL POMME SZAMPION PL' THEN c.prix_jour END) AS "PRIX MARCHE_PL POMME SZAMPION PL"

			FROM COTATIONS c
            LEFT OUTER JOIN produit_marche_stade p on p.libelle_produit = c.libelle_produit and p.stade = c.stade and p.marche = c.marche
            LEFT OUTER JOIN VACANCES_JOURS_FERIES b ON b.SAISON = c.SAISON AND b.SEMAINE_SAISON = c.SEMAINE_SAISON
        WHERE c.SEMAINE_SAISON <=43
        GROUP BY
            c.DATE_INTERROGATION,
            c.SAISON,
            CASE 
                WHEN TO_CHAR((c.DATE_INTERROGATION::DATE + INTERVAL '1 day'), 'MM')::INTEGER < 8 
                THEN (TO_CHAR((c.DATE_INTERROGATION::DATE + INTERVAL '1 day'), 'MM')::INTEGER + 12 - 7) 
                ELSE (TO_CHAR((c.DATE_INTERROGATION::DATE + INTERVAL '1 day'), 'MM')::INTEGER - 7) 
            END,
            c.SEMAINE_SAISON, c.SEMAINE,
            CASE WHEN (b.INDICATEUR_VACANCES IS NULL) THEN 0 ELSE b.INDICATEUR_VACANCES END
		ORDER BY 1 """ 
       
        conn = self.get_postgres_connection()
        try:
            # Exécute la requête et charge le résultat dans un DataFrame
            df = pd.read_sql(query, conn, index_col=None)  # Assure que 'DATE_INTERROGATION' n'est pas l'index
            df['DATE_INTERROGATION'] = pd.to_datetime(df['DATE_INTERROGATION'], errors='coerce')
            # Optionnel : Réinitialiser l'index au cas où (sûreté)
            df.reset_index(drop=True, inplace=True)
        finally:
            conn.close()
        
        return df

    def complete_with_trade_data(self, df):
        """
        Complète le DataFrame 'df' (contenant déjà les données chargées depuis la base)
        avec les informations issues du fichier EU_APPLES_trade_data_en.csv
        (export / import / prix moyen ...), décalées de 3 mois pour aligner les périodes.
        """
        # 1) Lecture du fichier CSV
        df_trade = pd.read_csv(os.path.join(DATA_RAW_DIR,"EU_APPLES_trade_data_en.csv"))

        # 2) Préparation des données
        df_trade['Marketing Year'] = df_trade['Marketing Year'].apply(lambda x: x.split('/')[0])
        df_trade = df_trade.sort_values(by=['Marketing Year', 'Month Order in MY'])
        df_trade['Marketing Year'] = df_trade['Marketing Year'].astype(int)
        df_trade['Month Order in MY'] = df_trade['Month Order in MY'].astype(int)
        
        # 3) Filtrage : on ne garde que la France, on regroupe par année/mois et Flow
        df_trade = df_trade[df_trade['Member State'] == 'France']
        df_trade = df_trade[['Marketing Year', 'Month Order in MY', 'Flow', 
                            'Value in thousand euro', 'Quantity in tonnes']]
        df_trade = df_trade.groupby(['Marketing Year', 'Month Order in MY', 'Flow']).sum().reset_index()

        # 4) Calcul du prix moyen
        df_trade['Price'] = df_trade['Value in thousand euro'] / df_trade['Quantity in tonnes']

        # 5) Pivot table (pour séparer Export vs. Import en colonnes)
        df_trade = df_trade.pivot_table(
            columns='Flow',
            index=['Marketing Year', 'Month Order in MY'],
            values=['Price', 'Quantity in tonnes']
        ).reset_index()

        # 6) Renommer les colonnes
        df_trade.columns = [
            'Marketing Year', 
            'Month Order in MY', 
            'PRIX_MOYEN_EXPORT_M-3', 
            'PRIX_MOYEN_IMPORT_M-3', 
            'TOTAL_EXPORT_M-3', 
            'TOTAL_IMPORT_M-3'
        ]
        df_trade = df_trade.fillna(0)

        # 7) Création des colonnes SAISON / MOIS_SAISON "décalées de 3 mois"
        #    => On simule la dispo 3 mois plus tard.
        df_trade['SAISON'] = df_trade['Marketing Year'].shift(-3)
        df_trade['MOIS_SAISON'] = df_trade['Month Order in MY'].shift(-3)

        # 8) Jointure
        df = pd.merge(
            df, 
            df_trade,
            left_on=['SAISON', 'MOIS_SAISON'],
            right_on=['SAISON', 'MOIS_SAISON'],
            how='left'
        )

        # 9) Nettoyage des colonnes inutiles
        #pd.set_option('display.max_columns', None)
        #print(df.columns)
        df.drop(columns=['Marketing Year', 'Month Order in MY'], inplace=True)

        return df

    def complete_with_forecasts(self, df):
        """
        Complète le DataFrame 'df' avec les prévisions de récolte
        issues du fichier APPLE_SHORT_TERM_OUTLOOK.xlsx.
        """
        # Charger les données depuis le fichier Excel
        file_path = os.path.join(DATA_RAW_DIR, "APPLE_SHORT_TERM_OUTLOOK.xlsx")
        df_prev = pd.read_excel(file_path)

        # Transposer les données
        df_prev = df_prev.T

        # Supprimer la seconde ligne
        df_prev = df_prev.drop(df_prev.index[1])

        # Utiliser la première ligne comme en-tête pour les noms de colonnes
        df_prev.columns = df_prev.iloc[0]
        df_prev = df_prev.drop(df_prev.index[0])

        # Réinitialiser les index
        df_prev = df_prev.reset_index()

        # Renommer la première colonne en 'SAISON'
        df_prev = df_prev.rename(columns={'index': 'SAISON'})

        # Dans la colonne SAISON, ne garder que le chiffre avant '/' et convertir en entier
        df_prev['SAISON'] = df_prev['SAISON'].apply(lambda x: x.split('/')[0])
        df_prev['SAISON'] = df_prev['SAISON'].astype(int)

        # Renommer les colonnes pour plus de clarté
        df_prev = df_prev.rename(columns={
            'Total production': 'PREV_PRODUCTION_SAISON',
            'Ending stocks (fresh)': 'PREV_STOCKS_FIN_SAISON',
            'Self-sufficiency rate (fresh) %': 'PREV_TAUX_AUTONOMIE_FRAIS',
            'Self-sufficiency rate (processed) %': 'PREV_TAUX_AUTONOMIE_TRANSFORME'
        })

        # Supprimer les colonnes inutiles
        columns_to_drop = [
            'Area (1000 ha)', 'Yield (t/ha)', 'Losses and feed use', 'Usable production',
            'Production (fresh)', 'Exports (fresh)', 'Imports (fresh)', 'Consumption (fresh)',
            'per capita consumption (kg) - fresh', 'Change in stocks (fresh)',
            'Production (processed)', 'Exports (processed)', 'Imports (processed)',
            'Consumption (processed)', 'per capita consumption (kg) - processed'
        ]
        df_prev = df_prev.drop(columns=columns_to_drop)
        # Convertir les colonnes prévisionnelles en numérique
        cols_to_convert = [
            'PREV_PRODUCTION_SAISON',
            'PREV_STOCKS_FIN_SAISON',
            'PREV_TAUX_AUTONOMIE_FRAIS',
            'PREV_TAUX_AUTONOMIE_TRANSFORME'
        ]

        for col in cols_to_convert:
            if col in df_prev.columns:
                df_prev[col] = pd.to_numeric(df_prev[col], errors='coerce')
                #print(f"Colonne {col} convertie en numérique.")

        # Jointure des deux DataFrames sur la colonne 'SAISON'
        df = pd.merge(df, df_prev, left_on='SAISON', right_on='SAISON', how='left')
        #print("Types des colonnes après import Excel :")
        #print(df_prev.dtypes)

        return df

    def complete_with_meteo(self, df):
        """
        Complète le DataFrame 'df' avec les données météo
        des 2 semaines précédentes (S-1 et S-2) en utilisant ANNEE et SEMAINE pour la jointure.
        """
        # 1) Charger les données météo depuis PostgreSQL
        conn = self.get_postgres_connection()
        query = '''
        SELECT 
            MIN(DATE_METEO) AS "DATE_LUNDI",
            annee AS "ANNEE",
            semaine AS "SEMAINE",
            SUM(CASE WHEN DIRECTION = 'NE' THEN RR ELSE 0 END) AS "NE_SOMME_RR",
            AVG(CASE WHEN DIRECTION = 'NE' THEN TM ELSE NULL END) AS "NE_MOYENNE_TM",
            MIN(CASE WHEN DIRECTION = 'NE' THEN TN ELSE NULL END) AS "NE_MIN_TN",
            MAX(CASE WHEN DIRECTION = 'NE' THEN TX ELSE NULL END) AS "NE_MAX_TX",
            AVG(CASE WHEN DIRECTION = 'NE' THEN UN ELSE NULL END) AS "NE_MOYENNE_UN",
            MAX(CASE WHEN DIRECTION = 'NE' THEN UN ELSE NULL END) AS "NE_MAX_UN",
            AVG(CASE WHEN DIRECTION = 'NE' THEN FFM ELSE NULL END) AS "NE_MOYENNE_FFM",
            SUM(CASE WHEN DIRECTION = 'NE' THEN GLOT ELSE 0 END) AS "NE_SOMME_GLOT"
        FROM METEO_DETAIL
        GROUP BY annee, semaine
        ORDER BY annee, semaine;
        '''

        try:
            df_meteo_global = pd.read_sql_query(query, conn)
            conn.close()
        except Exception as e:
            print("Erreur lors de l'exécution de la requête SQL :", e)
            return df  # Renvoie le DataFrame d'origine si la requête échoue

        # Vérifie si df_meteo_global contient des données
        if df_meteo_global.empty:
            print("Aucune donnée météo récupérée depuis la base de données.")
            return df

        df['ANNEE'] = df['ANNEE'].astype(int)
        df['SEMAINE'] = df['SEMAINE'].astype(int)
        df_meteo_global['ANNEE'] = df_meteo_global['ANNEE'].astype(int)
        df_meteo_global['SEMAINE'] = df_meteo_global['SEMAINE'].astype(int)


        # 2) Ajouter les colonnes SEMAINE_S-1 et SEMAINE_S-2 dans df
        df['SEMAINE_S-1'] = df['SEMAINE'] - 1
        df['SEMAINE_S-2'] = df['SEMAINE'] - 2
        df['ANNEE_S-1'] = df['ANNEE']
        df['ANNEE_S-2'] = df['ANNEE']

        # Gérer le changement d'année pour S-1 et S-2
        df.loc[df['SEMAINE_S-1'] == 0, 'ANNEE_S-1'] -= 1
        df.loc[df['SEMAINE_S-1'] == 0, 'SEMAINE_S-1'] = 52
        df.loc[df['SEMAINE_S-2'] <= 0, 'ANNEE_S-2'] -= 1
        df.loc[df['SEMAINE_S-2'] <= 0, 'SEMAINE_S-2'] = 52 + df['SEMAINE_S-2']



        # 3) Jointure pour la semaine S-1
        df = pd.merge(
            df,
            df_meteo_global,
            left_on=['ANNEE_S-1', 'SEMAINE_S-1'],
            right_on=['ANNEE', 'SEMAINE'],
            how='left',
            suffixes=('', '_S-1')
        )

        # 4) Jointure pour la semaine S-2
        df = pd.merge(
            df,
            df_meteo_global,
            left_on=['ANNEE_S-2', 'SEMAINE_S-2'],
            right_on=['ANNEE', 'SEMAINE'],
            how='left',
            suffixes=('', '_S-2')
        )

        # 5) Nettoyage des colonnes inutiles
        cols_to_drop = [
            'DATE_LUNDI', 'DATE_LUNDI_S-1', 'DATE_LUNDI_S-2', 
            'ANNEE_S-1', 'SEMAINE_S-1', 'ANNEE_S-2', 'SEMAINE_S-2'
        ]
        df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)
        #print('Colonnes nettoyées :', cols_to_drop)
        #print(df.columns) 

        return df

    def add_vacations(self, df):
        """
        Ajoute des indicateurs de vacances décalés S-1 et S+1.

        Args:
            df (pd.DataFrame): Le DataFrame à traiter.

        Returns:
            pd.DataFrame: Le DataFrame avec les indicateurs de vacances ajoutés.
        """
        # Vérifie que la colonne 'VACANCES_INDICATEUR_S' existe
        if 'VACANCES_INDICATEUR_S' not in df.columns:
            raise ValueError("La colonne 'VACANCES_INDICATEUR_S' est manquante dans le DataFrame.")

        # Créer les indicateurs décalés
        df['INDICATEUR_VACANCES_S-1'] = df['VACANCES_INDICATEUR_S'].shift(1).fillna(0)
        df['INDICATEUR_VACANCES_S+1'] = df['VACANCES_INDICATEUR_S'].shift(-1).fillna(0)

        print("Les indicateurs de vacances décalés S-1 et S+1 ont été ajoutés.")
        return df

    def add_variations(self, df):
        """
        Ajoute des colonnes de variation hebdomadaire et mensuelle pour les colonnes de prix,
        les importations/exportations, en une seule opération pour éviter la fragmentation du DataFrame.
        Remplace les valeurs nulles dans les colonnes de variations par 0.
        """

        # 1) Repérer toutes les colonnes de prix
        prix_jour_cols = [col for col in df.columns if col.startswith('PRIX ')]

        # 2) Convertir les colonnes de prix en type numérique (remplace les erreurs par NaN)
        df[prix_jour_cols] = df[prix_jour_cols].apply(pd.to_numeric, errors='coerce')

        # 3) Construire un DataFrame temporaire pour les nouvelles colonnes
        new_variations = pd.DataFrame(index=df.index)

        # 3a) Variations hebdomadaires et mensuelles pour chaque colonne de prix
        for col in prix_jour_cols:
            # Vérifie que la colonne est bien numérique avant calcul
            if df[col].dtype.kind in 'fc':  # 'f' pour float, 'c' pour complex (généralement float)
                new_variations[f'VAR_HEBDO_{col}'] = df[col].diff()
                new_variations[f'VAR_MENSUEL_{col}'] = df[col].diff(periods=4)

        # 3b) Variations pour tes autres colonnes spécifiques (export/import)
        export_import_cols = ['PRIX_MOYEN_EXPORT_M-3', 'PRIX_MOYEN_IMPORT_M-3', 'TOTAL_EXPORT_M-3', 'TOTAL_IMPORT_M-3']
        for col in export_import_cols:
            if col in df.columns and df[col].dtype.kind in 'fc':
                new_variations[f'VAR_MENSUEL_{col}'] = df[col].diff(periods=4)

        # 4) Concaténer les nouvelles colonnes au DataFrame principal
        df = pd.concat([df, new_variations], axis=1)

        # 5) Remplacer les valeurs nulles dans les colonnes de variations par 0
        var_columns = [col for col in new_variations.columns if col.startswith('VAR_')]
        df[var_columns] = df[var_columns].fillna(0)

        # 6) Forcer la copie pour défragmenter
        df = df.copy()

        print("Les colonnes de variations ont été ajoutées et les valeurs nulles remplacées par 0, évitant la fragmentation.")
        return df

    def handle_missing_values(self, df):
        """
        Gère les valeurs manquantes en ajoutant des indicateurs de disponibilité
        pour chaque colonne de prix.
        Interpole les NaN uniquement si la semaine précédente et suivante ne sont pas nulles.
        Sinon, laisse les NaN inchangés.
        
        Args:
            df (pd.DataFrame): Le DataFrame à traiter.
        
        Returns:
            pd.DataFrame: Le DataFrame avec les indicateurs de disponibilité ajoutés
                        et les valeurs manquantes traitées.
        """
        # Identifier toutes les colonnes de prix
        price_columns = [col for col in df.columns if col.startswith('PRIX ')]

        # Interpoler les NaN isolés (où la semaine précédente et suivante ne sont pas NaN)
        for col in price_columns:
            #print(f"Traitement de la colonne : {col}")
            
            # Vérifie si la colonne contient des NaN
            if df[col].isna().sum() > 0:
                # Créer des masques pour les valeurs précédentes et suivantes non NaN
                prev_not_nan = df[col].shift(1).notna()
                next_not_nan = df[col].shift(-1).notna()
                
                # Masque pour les NaN isolés
                single_nan_mask = df[col].isna() & prev_not_nan & next_not_nan
                
                # Afficher les indices des NaN identifiés pour vérification
                #print(f"Indices des NaN isolés détectés : {df.loc[single_nan_mask].index.tolist()}")
                
                # Interpoler uniquement les NaN isolés
                if single_nan_mask.any():
                    df.loc[single_nan_mask, col] = (
                        df[col].shift(1)[single_nan_mask] + df[col].shift(-1)[single_nan_mask]
                    ) / 2
                 #   print(f"Interpolé les valeurs manquantes isolées dans la colonne '{col}'.")
                #else:
                #    print(f"Colonne '{col}' : aucune valeur manquante isolée à interpoler.")
            #else:
                #print(f"Colonne '{col}' : aucune valeur manquante.")

        # Créer un DataFrame temporaire pour les indicateurs de disponibilité
        availability_indicators = pd.DataFrame(index=df.index)
        for col in price_columns:
            # Créer une colonne binaire indiquant la disponibilité
            indicator_col = f'{col}_DISPONIBLE'
            availability_indicators[indicator_col] = df[col].notna().astype(int)

        # Concaténer les indicateurs de disponibilité au DataFrame principal
        df = pd.concat([df, availability_indicators], axis=1)

        # Forcer la copie pour défragmenter le DataFrame
        df = df.copy()

        print("Les indicateurs de disponibilité ont été ajoutés.")
        return df

if __name__ == '__main__':
    # Exemple de config de connexion (adapte au besoin)
    db_config = {
        "host": "prixfetl_postgres",
        "port": 5432,
        "database": "IAFetL",
        "user": "prixfetl",
        "password": "Leumces123"
    }
    # Desactivation des warnings de perf qui ont tendance a polluer les logs
    warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

    # Instanciation
    loader = ApplePriceDataLoader(db_config)

    # 1) Chargement des données depuis PostgreSQL
    df_sql = loader.load_prices_dataframe()

    # 2) Complément avec le fichier CSV
    df_complet = loader.complete_with_trade_data(df_sql)

    # Complément avec les prévisions
    df_complet = loader.complete_with_forecasts(df_complet)

    # 5) Compléter avec meteo (S-1, S-2)
    df_complet = loader.complete_with_meteo(df_complet)
    
     # 5) Gérer les valeurs manquantes en ajoutant des indicateurs de disponibilité
    df_complet = loader.handle_missing_values(df_complet)

    # 6) Ajouter les variations et remplacer les NaN par 0 dans les colonnes VAR_
    df_complet = loader.add_variations(df_complet)

    # 7) Ajouter les indicateurs de vacances décalés
    df_complet = loader.add_vacations(df_complet)

    # 3) Conversion explicite de DATE_INTERROGATION en texte formaté
    #df_complet['DATE_INTERROGATION'] = pd.to_datetime(df_complet['DATE_INTERROGATION'], errors='coerce')
    #df_complet['DATE_INTERROGATION'] = df_complet['DATE_INTERROGATION'].dt.strftime('%Y-%m-%d')

    # 4) Réinitialisation de l'index pour éviter des erreurs d'export
    #df_complet.reset_index(inplace=True)

    # 5) Export des 100 premières lignes vers Excel
    output_path = os.path.join(DATA_OUTPUT_DIR,"df_complet_sample.xlsx")
    df_complet.head(100).to_excel(output_path, index=False)
    print(f"Les 100 premières lignes ont été exportées vers {output_path}")

