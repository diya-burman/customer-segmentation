# app/_init_.py

from flask import Flask, render_template

def create_app():
    app = Flask(__name__)
    
    from .api import api_bp
    app.register_blueprint(api_bp, url_prefix='/api')
    
    @app.route('/')
    def index():
        return render_template('index.html')
    
    return app