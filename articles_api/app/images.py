import os
import uuid
import aioftp
import asyncpg
from fastapi import APIRouter, File, UploadFile, Form, HTTPException, Depends
from fastapi.responses import JSONResponse
from typing import Optional
import httpx
from auth import check_api_key

# ---------- AJOUT PILLOW ----------
from io import BytesIO
from PIL import Image

# Config FTP et DB
FTP_URL = os.getenv("FTP_URL")  # ex: "ftp.cluster027.hosting.ovh.net"
FTP_USER = os.getenv("FTP_USER")
FTP_PASSWORD = os.getenv("FTP_PASSWORD")
FTP_UPLOAD_PATH = "/www/images/"
BASE_URL = "https://www.intellibooster.com/images/"
DATABASE_URL = os.getenv("DATABASE_URL")

router = APIRouter()

async def get_db():
    return await asyncpg.connect(DATABASE_URL)

# ---------------------------------------------------------------------
# Fonction utilitaire : compresse en JPEG avec un max de 300 Ko
# ---------------------------------------------------------------------
def compress_to_jpeg(
    content: bytes,
    max_size_kb: int = 300,
    initial_quality: int = 85,
    min_quality: int = 30
) -> bytes:
    """
    Convertit l'image en JPEG et compresse par palier
    jusqu'à ce qu'elle soit sous la taille max_size_kb (en Ko),
    ou jusqu'à atteindre min_quality.
    Retourne les bytes de l'image compressée.
    """
    img = Image.open(BytesIO(content))

    # Convertir en mode "RGB" pour éviter les problèmes (ex: PNG à 4 canaux)
    if img.mode != "RGB":
        img = img.convert("RGB")

    # On va itérer en diminuant la qualité
    quality = initial_quality
    output = BytesIO()

    while True:
        output.seek(0)
        output.truncate(0)

        # Sauvegarde en JPEG dans un buffer
        img.save(output, format="JPEG", quality=quality)

        size_kb = output.tell() / 1024
        if size_kb <= max_size_kb or quality <= min_quality:
            break
        # Baisser la qualité par palier
        quality -= 5

    return output.getvalue()


def generate_unique_name_from_title(titre: Optional[str], extension: str) -> str:
    """
    Génère un nom de fichier avec l'extension d'origine si `keep_format == 1`,
    sinon `.jpg` par défaut.
    """
    short_part = (titre[:10] if titre else "image")  # si titre est None
    # Supprimer espaces/char spéciaux potentiellement
    short_part = short_part.replace(" ", "_").lower()

    # Suffixe unique sur 5 ou 6 caractères, par ex:
    unique_suffix = uuid.uuid4().hex[:5]

    return f"{short_part}-{unique_suffix}.{extension}"


@router.post("/images/upload-image")
async def upload_image(
    file: UploadFile = File(...),
    nom: Optional[str] = Form(None),
    titre: Optional[str] = Form(None),
    keep_format: Optional[int] = Form(0),  # 0 = compression (par défaut), 1 = garde le format original
    db=Depends(get_db),
    api_key: None = Depends(check_api_key)
):
    try:
        # Lecture brute du fichier
        content = await file.read()

        # Déterminer l'extension d'origine
        extension = file.filename.split(".")[-1].lower()

        # Si keep_format est 1, ne pas compresser ni convertir
        if keep_format == 1:
            processed_content = content
        else:
            processed_content = compress_to_jpeg(content)
            extension = "jpg"  # Converti en JPEG

        # Générer un nom de fichier si non fourni
        if not nom:
            nom = generate_unique_name_from_title(titre, extension)

        # Nettoyer le nom
        nom = nom.replace(" ", "_").lower()

        # Construire l'URL de l'image
        url = f"{BASE_URL}{nom}"

        # Upload sur le serveur FTP
        async with aioftp.Client.context(FTP_URL, user=FTP_USER, password=FTP_PASSWORD) as client:
            async with client.upload_stream(FTP_UPLOAD_PATH + nom) as stream:
                await stream.write(processed_content)

        # Insérer dans la base de données
        query = """
        INSERT INTO images (nom, titre, url)
        VALUES ($1, $2, $3)
        RETURNING id;
        """
        image_id = await db.fetchval(query, nom, titre, url)

        return JSONResponse({
            "id": image_id,
            "nom": nom,
            "titre": titre,
            "url": url,
            "message": "Image uploadée avec succès"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/images/add-image-url")
async def add_image_url(
    url_origine: str = Form(...),
    nom: Optional[str] = Form(None),
    titre: Optional[str] = Form(None),
    keep_format: Optional[int] = Form(0),  # 0 = compression (par défaut), 1 = garde le format original
    db=Depends(get_db),
    api_key: None = Depends(check_api_key)
):
    try:
        # Télécharger l'image depuis l'URL d'origine
        async with httpx.AsyncClient() as client:
            response = await client.get(url_origine)
            if response.status_code != 200:
                raise HTTPException(status_code=400, detail="Impossible de télécharger l'image")
            content = response.content  # Contenu brut de l’image

        # Déterminer l'extension d'origine
        extension = url_origine.split(".")[-1].split("?")[0].lower()

        # Si keep_format est 1, ne pas compresser ni convertir
        if keep_format == 1:
            processed_content = content
        else:
            processed_content = compress_to_jpeg(content)
            extension = "jpg"  # Converti en JPEG

        # Générer un nom de fichier si non fourni
        if not nom:
            nom = generate_unique_name_from_title(titre, extension)

        # Nettoyer le nom
        nom = nom.replace(" ", "_").lower()

        # Construire l'URL finale
        url = f"{BASE_URL}{nom}"

        # Upload sur le serveur FTP
        async with aioftp.Client.context(FTP_URL, user=FTP_USER, password=FTP_PASSWORD) as client:
            async with client.upload_stream(FTP_UPLOAD_PATH + nom) as stream:
                await stream.write(processed_content)

        # Insérer dans la base de données
        query = """
        INSERT INTO images (nom, titre, url, url_origine)
        VALUES ($1, $2, $3, $4)
        RETURNING id;
        """
        image_id = await db.fetchval(query, nom, titre, url, url_origine)

        return JSONResponse({
            "id": image_id,
            "nom": nom,
            "titre": titre,
            "url": url,
            "url_origine": url_origine,
            "message": "Image téléchargée et ajoutée avec succès"
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/images/update-image")
async def update_image(
    file: Optional[UploadFile] = File(None),
    url_origine: Optional[str] = Form(None),
    nom: str = Form(...),
    titre: Optional[str] = Form(None),
    update: bool = Form(False),
    db=Depends(get_db),
    api_key: None = Depends(check_api_key)
):
    try:
        # Vérifier si l'image existe déjà
        existing_image = await db.fetchrow("SELECT * FROM images WHERE nom = $1", nom)
        if not existing_image:
            raise HTTPException(status_code=404, detail="L'image n'existe pas")

        # Si update=False, empêcher l'écrasement
        if not update:
            raise HTTPException(status_code=400, detail="L'image existe déjà, ajoutez `update=True` pour la modifier")

        # Mise à jour des valeurs
        new_titre = titre if titre else existing_image["titre"]
        new_url_origine = url_origine if url_origine else existing_image["url_origine"]

        # Si un nouveau fichier est fourni, on le compresse puis on remplace l'ancien
        if file:
            content = await file.read()
            compressed_content = compress_to_jpeg(content)

            # Upload sur FTP
            async with aioftp.Client.context(FTP_URL, user=FTP_USER, password=FTP_PASSWORD) as client:
                async with client.upload_stream(FTP_UPLOAD_PATH + nom) as stream:
                    await stream.write(compressed_content)

        # Mise à jour en base de données
        query = """
        UPDATE images
        SET titre = $1, url_origine = $2
        WHERE nom = $3
        RETURNING id, nom, titre, url, url_origine;
        """
        updated_image = await db.fetchrow(query, new_titre, new_url_origine, nom)

        return JSONResponse({
            "id": updated_image["id"],
            "nom": updated_image["nom"],
            "titre": updated_image["titre"],
            "url": updated_image["url"],
            "url_origine": updated_image["url_origine"],
            "message": "Image mise à jour avec succès (compressée si nouveau fichier)."
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/images/list-images")
async def list_images(db=Depends(get_db), api_key: None = Depends(check_api_key)):
    try:
        # Récupérer toutes les images
        query = "SELECT * FROM images ORDER BY created_at DESC"
        records = await db.fetch(query)

        # Convertir les records en dictionnaires et convertir datetime en string
        images = [
            {**dict(record), "created_at": record["created_at"].isoformat() if record["created_at"] else None}
            for record in records
        ]

        return JSONResponse({
            "images": images,
            "message": "Liste des images récupérée avec succès"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
