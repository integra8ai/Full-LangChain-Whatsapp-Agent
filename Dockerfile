FROM python:3.11-slim

WORKDIR /app

# system deps for unstructured / pdf parsing if needed (optional)
RUN apt-get update && apt-get install -y build-essential libpoppler-cpp-dev pkg-config poppler-utils curl git && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=5000

CMD ["gunicorn", "-w", "2", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:5000", "app:app"]
