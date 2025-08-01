from flask import Flask
from app.routes import routes

def create_app():
    app = Flask(__name__)

    # Enregistrement des routes
    app.register_blueprint(routes)

    return app
