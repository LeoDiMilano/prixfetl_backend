import os
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

# Charger les variables d'environnement depuis un fichier .env s'il existe
load_dotenv()

POSTGRES_HOST = os.environ.get('POSTGRES_HOST', 'localhost')
POSTGRES_PORT = os.environ.get('POSTGRES_PORT', '5432')
POSTGRES_DB   = os.environ.get('POSTGRES_DB', 'IAFetL')
POSTGRES_USER = os.environ.get('POSTGRES_USER', 'postgres')
POSTGRES_PASSWORD = os.environ.get('POSTGRES_PASSWORD', '')

def get_connection():
    """
    Retourne une connexion psycopg2. 
    Attention: il faut penser à fermer la connexion après utilisation !
    """
    conn = psycopg2.connect(
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        dbname=POSTGRES_DB,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD
    )
    return conn

def fetch_all(query, params=None):
    """
    Exécute un SELECT et renvoie toutes les lignes sous forme d'une liste de dict.
    """
    if params is None:
        params = {}
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as curs:
            curs.execute(query, params)
            rows = curs.fetchall()
            return [dict(row) for row in rows]
    finally:
        conn.close()

def fetch_one(query, params=None):
    """
    Exécute un SELECT et renvoie une seule ligne sous forme de dict.
    """
    if params is None:
        params = {}
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as curs:
            curs.execute(query, params)
            row = curs.fetchone()
            return dict(row) if row else None
    finally:
        conn.close()

def execute(query, params=None):
    """
    Exécute une requête (INSERT, UPDATE, DELETE, CREATE, etc.) et renvoie le nombre de lignes affectées.
    """
    if params is None:
        params = {}
    conn = get_connection()
    try:
        with conn.cursor() as curs:
            curs.execute(query, params)
            affected = curs.rowcount
        conn.commit()
        return affected
    finally:
        conn.close()
