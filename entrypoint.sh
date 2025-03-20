#!/bin/bash

# Script d'entrée pour le conteneur Docker
# Ce script configure l'environnement et lance les services nécessaires

set -e

# Vérifier si le répertoire virtuel existe, sinon le créer
if [ ! -d "/app/venv" ]; then
    echo "Création de l'environnement virtuel Python..."
    python -m venv /app/venv
fi

# Activer l'environnement virtuel
. /app/venv/bin/activate


# Installer les dépendances si requirements.txt existe
if [ -f "/app/requirements.txt" ]; then
    echo "Installation des dépendances Python..."
    pip install -r /app/requirements.txt
fi

# Installer Chrome et ChromeDriver pour Selenium
if ! command -v google-chrome &> /dev/null; then
    echo "Installation de Google Chrome..."
    apt-get update
    apt-get install -y wget gnupg
    wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | apt-key add -
    echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" > /etc/apt/sources.list.d/google-chrome.list
    apt-get update
    apt-get install -y google-chrome-stable
fi

# Configurer cron
echo "Configuration de cron..."
crontab /app/crontab
service cron start

# Démarrer SSH
service ssh start# Démarrer SSH
service ssh start

# Lancer l'application Flask en arrière-plan si demandé
#if [ "$1" = "api" ]; then
#    echo "Démarrage de l'API Flask..."
#    gunicorn --bind 0.0.0.0:5000 "main:create_app()" --daemon
#fi

# Garder le conteneur en vie
echo "Conteneur prêt et en attente..."
gunicorn --bind 0.0.0.0:5000 "main:create_app()"

tail -f /dev/null
