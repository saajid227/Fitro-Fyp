FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

EXPOSE 8000

# Render (and other PaaS) provide the port via $PORT.
CMD ["sh", "-c", "uvicorn Frontend.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
