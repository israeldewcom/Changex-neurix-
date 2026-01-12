#!/usr/bin/env python3
"""
ChangeX Neurix - Production WSGI with graceful degradation
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, jsonify

# Create minimal app first
app = Flask(__name__)

# Basic config
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-key-change-in-prod')
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///app.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Try to load full app, fall back to basic
try:
    from app import create_app
    from config import ProductionConfig
    
    # Try to create full app
    app = create_app(ProductionConfig)
    print("✅ Full application loaded successfully")
    
except ImportError as e:
    print(f"⚠️ Missing dependencies: {e}")
    print("Running in minimal mode...")
    
    # Setup minimal routes
    @app.route('/')
    def index():
        return jsonify({
            'status': 'minimal',
            'message': 'App is running in minimal mode. Install full requirements.',
            'endpoints': ['/health', '/install']
        })
    
    @app.route('/health')
    def health():
        return jsonify({'status': 'healthy', 'mode': 'minimal'})
    
    @app.route('/install')
    def install():
        return jsonify({
            'instructions': 'Run: pip install -r requirements.txt',
            'note': 'Some packages may fail. Try requirements-minimal.txt first'
        })

# Export for Gunicorn
application = app

if __name__ == '__main__':
    app.run()
