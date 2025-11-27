FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y gcc g++ && rm -rf /var/lib/apt/lists/*

COPY requirements-pipeline.txt .
RUN pip install --no-cache-dir -r requirements-pipeline.txt

COPY pipeline/ ./pipeline/
COPY run_pipeline.py .

RUN mkdir -p /app/logs /app/data/technical /app/models

CMD ["python", "run_pipeline.py"]