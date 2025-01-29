from flask import Flask
from routers.history import history_bp
from routers.config import config_bp
from routers.predict import predict_bp

def create_app():
    app = Flask(__name__)

    # Enregistrement des blueprints sous /api
    app.register_blueprint(history_bp, url_prefix='/api')
    app.register_blueprint(config_bp, url_prefix='/api')
    app.register_blueprint(predict_bp, url_prefix='/api')

    return app

if __name__ == '__main__':
    my_app = create_app()
    # Pour un environnement de dev, debug=True 
    my_app.run(host='0.0.0.0', port=5000, debug=True)
