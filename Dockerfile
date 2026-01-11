FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy files
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app/

# Expose port
EXPOSE 8080

# Run the server
CMD ["sh", "-c", "uvicorn ml_service:app --host 0.0.0.0 --port ${PORT:-8080}"]

