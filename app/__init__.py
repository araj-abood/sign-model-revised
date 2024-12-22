from flask_cors import CORS
from flask import Flask

def create_app():
    app = Flask(__name__)
    CORS(app)  # Enable CORS for frontend compatibility
    from .routes import main as main_blueprint
    app.register_blueprint(main_blueprint)
    return app
