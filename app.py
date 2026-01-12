from flask import Flask, jsonify, render_template
import os

app = Flask(__name__)

# Basic config for Render
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key')
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///app.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Basic routes
@app.route('/')
def index():
    return jsonify({
        'status': 'running',
        'message': 'ChangeX Neurix API',
        'endpoints': ['/health', '/admin', '/api/v2'],
        'note': 'AI features will be available in premium deployment'
    })

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'database': 'connected' if os.environ.get('DATABASE_URL') else 'local',
        'environment': os.environ.get('FLASK_ENV', 'development')
    })

@app.route('/admin')
def admin():
    return jsonify({
        'message': 'Admin panel available at /admin after setup',
        'setup': 'Run database migrations first'
    })

# Try to load full app, but don't fail if dependencies missing
try:
    # Check for database
    from flask_sqlalchemy import SQLAlchemy
    db = SQLAlchemy(app)
    
    # Simple User model for testing
    class User(db.Model):
        id = db.Column(db.Integer, primary_key=True)
        username = db.Column(db.String(80), unique=True)
        email = db.Column(db.String(120), unique=True)
    
    # Create tables (for testing)
    with app.app_context():
        db.create_all()
    
    print("✅ Database setup successful")
    
except ImportError as e:
    print(f"⚠️ Missing dependency: {e}")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
