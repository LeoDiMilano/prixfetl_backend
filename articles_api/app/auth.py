# /home/ubuntu/PrixFetL/articles_api/app/auth.py

import os
from fastapi import Header, HTTPException, Depends
from fastapi.security.api_key import APIKeyHeader

API_KEY = os.getenv("API_KEY", "fallback_key")  # Remplace avec ta clé réelle
api_key_header = APIKeyHeader(name="X-API-KEY")

# Vérification de l'API key
def check_api_key(api_key: str = Depends(api_key_header)) -> None:
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")
