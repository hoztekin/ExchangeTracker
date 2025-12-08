FROM python:3.9-slim

WORKDIR /app

# Sistem bağımlılıkları - cache optimize
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Gerekli klasörleri önceden oluştur
RUN mkdir -p /app/logs /app/data/raw /app/data/technical /app/models /app/outputs

# Python bağımlılıkları (cache ler)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Uygulama dosyalarını kopyala
COPY . .

# Port
EXPOSE 8501

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Streamlit konfigürasyonu
ENV STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true

# Streamlit'i başlat
CMD ["streamlit", "run", "app.py"]