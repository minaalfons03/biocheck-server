FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# FIX 3: Pre-download VGG-Face model at build time so first /recognise call doesn't time out
RUN python -c "from deepface import DeepFace; DeepFace.build_model('VGG-Face')"

COPY app.py .

CMD gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --timeout 120
