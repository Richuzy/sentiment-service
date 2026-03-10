FROM python:3.11-slim

WORKDIR /app

# Install system deps (often needed by python wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency file first (Docker cache optimization)
COPY api/requirements.txt /app/api/requirements.txt

RUN pip install --no-cache-dir -r /app/api/requirements.txt

# Copy code + model
COPY api/ /app/api/
COPY ml/ /app/ml/
COPY model/ /app/model/

# Make imports work
ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]