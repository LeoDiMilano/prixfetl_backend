from flask import Flask
from routers.predict import predict_bp
from routers.history import history_bp
# etc.

app = Flask(__name__)
app.register_blueprint(predict_bp, url_prefix='/api')
app.register_blueprint(history_bp, url_prefix='/api')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)


#Ton site “front” (celui que tu as montré en maquette) pourra appeler ce webservice :

#GET /api/history?produit=Gala-Cal-65-70&nb_semaines=10 → renvoie l’historique de variations
#GET /api/config → renvoie la config pour l’affichage (produits disponibles, etc.)
#POST /api/predict → renvoie la prévision pour la/les semaines à venir
#Ainsi, la sélection de l’utilisateur (cases à cocher “Golden delicious - Cal. 70-80” ou “Variation prix S+1, etc.”) sera transmise à l’API.