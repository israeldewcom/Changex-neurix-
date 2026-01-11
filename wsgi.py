# wsgi.py - PRODUCTION WSGI ENTRY POINT
#!/usr/bin/env python3
"""
ChangeX Neurix - Production WSGI Entry Point
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import create_app
from config import ProductionConfig

# Create application instance
application = create_app(ProductionConfig)

if __name__ == '__main__':
    # Run with gunicorn in production
    application.run()
