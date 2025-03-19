# Utilisation d'une image Python légère et sécurisée
FROM python:3.11-slim

# Définir un dossier de travail
WORKDIR /app

# Installer les dépendances système nécessaires
RUN apt-get update && apt-get install -y \
    libpq-dev && \
    rm -rf /var/lib/apt/lists/*

# Installer les dépendances Python
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Définir la variable d'environnement pour le stockage des images
ENV UPLOAD_FOLDER="/app/images"

# Créer le dossier des images
RUN mkdir -p ${UPLOAD_FOLDER}

# Exposer le port du service
EXPOSE 3001

# Lancer Gunicorn avec Uvicorn Worker
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "-w", "2", "-b", "0.0.0.0:3001", "main:app"]
