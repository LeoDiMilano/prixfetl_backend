#!/bin/bash
export PATH=$PATH:/usr/sbin

# Créer le fichier de log et définir les permissions (exécuté en tant que root)
touch /var/log/cron.log.txt
chmod 0644 /var/log/cron.log.txt

# Démarrer le service cron
service cron start

# Démarrer SSH
service ssh start

export PYTHONPATH="/app:/app/routers:/app/services"
# Lancer Flask en arrière-plan
flask run --host=0.0.0.0 --port=5000 &

# Maintenir le conteneur actif
tail -f /dev/null