import os
import psycopg2
from psycopg2.extras import RealDictCursor
from fastapi import FastAPI, Form, HTTPException, Security, Depends
from fastapi.security.api_key import APIKeyHeader
from fastapi.responses import JSONResponse
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware
import re
import unicodedata
from images import router as images_router
from auth import check_api_key
from fastapi import FastAPI, Request
from fastapi import Request

app = FastAPI()
app.include_router(images_router)  # Pas de prefix !

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ✅ Autorise toutes les origines (à restreindre en prod)
    allow_credentials=True,
    allow_methods=["*"],  # ✅ Autorise toutes les méthodes (GET, POST, etc.)
    allow_headers=["*"],  # ✅ Autorise tous les headers (y compris Authorization)
)


# Fonction pour normaliser un titre en slug (é → e, ç → c, etc.)
def generate_slug(titre):
    titre = unicodedata.normalize('NFKD', titre).encode('ascii', 'ignore').decode('utf-8')
    titre = re.sub(r'\s+', '-', titre.lower())  # Remplace espaces par "-"
    titre = re.sub(r'[^a-z0-9\-]', '', titre)  # Supprime caractères spéciaux restants
    return titre

# Connexion à PostgreSQL
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://articles_user:Leumces123@prixfetl_postgres:5432/articles")

def get_db_connection():
    try:
        conn = psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)
        return conn
    except Exception as e:
        print(f"Erreur de connexion à la base de données : {e}")
        return None


# Vérifier la connexion à PostgreSQL
@app.get("articles/test-db", dependencies=[Depends(check_api_key)])
def test_db():
    conn = get_db_connection()
    if conn is None:
        return {"error": "Impossible de se connecter à la base de données"}
    try:
        cur = conn.cursor()
        cur.execute("SELECT 'Connexion réussie' AS message;")
        result = cur.fetchone()
        cur.close()
        conn.close()
        return {"message": result["message"]}
    except Exception as e:
        return {"error": str(e)}

# Ajouter un article (nécessite une clé API)

@app.post("/articles/add", dependencies=[Depends(check_api_key)])
async def create_article(
    titre: str = Form(...),
    langue: str = Form(...),
    article: str = Form(...),
    date_publication: str = Form(...),
    image1: Optional[str] = Form(None),
    image1_title: Optional[str] = Form(None),
    image1_link: Optional[str] = Form(None),
    image2: Optional[str] = Form(None),
    image2_title: Optional[str] = Form(None),
    image2_link: Optional[str] = Form(None),
    article_resume: Optional[str] = Form(None),
    post_linkedin: Optional[str] = Form(None),
    post_linkedin_2: Optional[str] = Form(None),
    url: Optional[str] = Form(None)  # URL peut être fourni, sinon on le génère
):
    conn = get_db_connection()
    if conn is None:
        raise HTTPException(status_code=500, detail="Impossible de se connecter à la base de données")

    try:
        cur = conn.cursor()

        # Insertion initiale pour récupérer l'ID
        cur.execute("""
            INSERT INTO articles (
                titre, langue, article, date_publication,
                image1, image1_title, image1_link,
                image2, image2_title, image2_link,
                article_resume, post_linkedin, post_linkedin_2
            ) 
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id;
        """, (
            titre, langue, article, date_publication,
            image1, image1_title, image1_link,
            image2, image2_title, image2_link,
            article_resume, post_linkedin, post_linkedin_2
        ))

        article_id = cur.fetchone()["id"]

        # ✅ Génération automatique du slug et de l'URL
        slug = generate_slug(titre)
        generated_url = f"https://www.intellibooster.com/news/{article_id}/{slug}"

        # ✅ Mise à jour de l'URL dans la base
        cur.execute("""
            UPDATE articles SET url = %s WHERE id = %s;
        """, (generated_url, article_id))

        conn.commit()
        cur.close()
        conn.close()

        return {
            "id": article_id,
            "url": generated_url,
            "message": "Article ajouté avec succès"
        }

    except Exception as e:
        return {"error": str(e)}

# Gestion des requêtes OPTIONS pour CORS
@app.options("/articles/all")
async def preflight():
    response = JSONResponse({"message": "Preflight response"})
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "X-API-KEY, Content-Type"
    response.headers["Access-Control-Allow-Credentials"] = "true"
    return response

# Récupérer tous les articles (nécessite une clé API)
@app.get("/articles/all", dependencies=[Depends(check_api_key)])
@app.get("/articles/all", dependencies=[Depends(check_api_key)])
def get_articles():
    conn = get_db_connection()
    if conn is None:
        raise HTTPException(status_code=500, detail="Impossible de se connecter à la base de données")

    try:
        cur = conn.cursor()
        cur.execute("SELECT * FROM articles ORDER BY date_publication DESC;")
        articles = cur.fetchall()
        cur.close()
        conn.close()
        return {"articles": articles}
    except Exception as e:
        return {"error": str(e)}
