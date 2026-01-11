# run.py - DEVELOPMENT SERVER
#!/usr/bin/env python3
"""
ChangeX Neurix - Development Server
"""

import os
from app import create_app, socketio
from config import DevelopmentConfig

app = create_app(DevelopmentConfig)

if __name__ == '__main__':
    print("🚀 ChangeX Neurix Development Server Starting...")
    print(f"Environment: {app.config['ENV']}")
    print(f"Debug: {app.config['DEBUG']}")
    print(f"Database: {app.config['SQLALCHEMY_DATABASE_URI']}")
    print(f"Redis: {app.config['REDIS_URL']}")
    print(f"API: http://localhost:5000")
    print(f"Admin: http://localhost:5000/admin")
    print("Press Ctrl+C to stop")
    
    socketio.run(
        app,
        host='0.0.0.0',
        port=5000,
        debug=True,
        use_reloader=True,
        log_output=True
    )
