import os
from flask import Flask, request, jsonify
from .routers.history import history_bp
from .routers.config import config_bp
from .routers.predict import predict_bp
from flask_cors import CORS
from functools import wraps
# from .utils import require_token  <--  On ne l'importe plus ici
# from .decorators import require_token
from .routers.decorators import require_token # <-- Import correct

# 1) On utilise une variable d'env "API_SECRET_TOKEN" si elle existe,
#    sinon on prend "DEV_TOKEN_1234" en valeur par défaut.
API_TOKEN_SECRET = os.environ.get("API_SECRET_TOKEN", "e67d89305f2763bd4920bb0deda2e33467a9154522a99f5a6268e1a222ab9e75")

# Nouvelle fonction pour appliquer le décorateur aux vues.
def apply_token_auth():
    def before_request():
        # Récupérer la vue
        view = request.endpoint
        if view not in history_bp.view_functions and view not in predict_bp.view_functions:
           return
        token = request.headers.get('Authorization')
        if not token or token != f'Bearer {API_TOKEN_SECRET}':
            return jsonify({'error': 'Unauthorized'}), 401
            
    return before_request


def create_app():
    app = Flask(__name__)

    # 2) Configurer CORS pour autoriser ton/tes domaines
    #    Tu peux bien sûr en rajouter ou mettre "*" si tu préfères
    CORS(app, resources={
        r"/api/*": {
            "origins": [
                "https://intellibooster.com",
                "https://www.intellibooster.com",
                "https://www.intelliboost.fr",
                "https://api.intellibooster.com"
            ]
        }
    })

    # 3) Appliquer le décorateur "require_token" avant chaque requête
    #    sur certains blueprints (ex: history, predict).
    #    Si tu veux protéger config_bp aussi, ajoute-le. Sinon, laisse-le libre.
    app.before_request(apply_token_auth())

    # 4) Enregistrer les blueprints
    app.register_blueprint(history_bp, url_prefix='/api')
    app.register_blueprint(config_bp, url_prefix='/api')
    app.register_blueprint(predict_bp, url_prefix='/api')

    return app

if __name__ == '__main__':
    my_app = create_app()
    # Pour un environnement de dev, debug=True
    my_app.run(host='0.0.0.0', port=5000, debug=True)