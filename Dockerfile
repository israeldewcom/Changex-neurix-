FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy app
COPY . .

# Download spaCy model
RUN python -m spacy download en_core_web_lg

CMD ["gunicorn", "wsgi:application", "--bind", "0.0.0.0:$PORT", "--workers", "4", "--worker-class", "gevent"]
