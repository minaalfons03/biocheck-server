FROM python:3.10-slim

# Install system dependencies needed by dlib and face_recognition
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel
RUN pip install dlib==19.24.2 --verbose
RUN pip install -r requirements.txt

# Copy app
COPY app.py .

EXPOSE 8080

CMD gunicorn app:app --bind 0.0.0.0:$PORT --workers 2 --timeout 120
