#!/bin/bash
set -e

echo "Starting ChangeX Neurix..."
echo "PORT: $PORT"
echo "FLASK_ENV: $FLASK_ENV"

# Wait a moment for database if needed
sleep 2

# Start Gunicorn
exec gunicorn wsgi:application \
  --bind 0.0.0.0:$PORT \
  --workers=2 \
  --threads=4 \
  --worker-class=gthread \
  --access-logfile - \
  --error-logfile - \
  --timeout 120
