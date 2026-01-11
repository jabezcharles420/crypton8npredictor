FROM python:3.10-slim

WORKDIR /app

# Install system deps (needed for numpy / tensorflow sometimes)
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Railway uses PORT env var (usually 8080)
EXPOSE 8080

CMD ["sh", "-c", "uvicorn ml_service:app --host 0.0.0.0 --port ${PORT}"]
