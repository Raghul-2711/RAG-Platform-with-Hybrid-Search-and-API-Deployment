FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Build the indexes at image-build time so the container starts ready to serve.
# Remove this line if you'd rather call POST /ingest after the container is up.
RUN python ingest.py

EXPOSE 5000
ENV PORT=5000

CMD ["python", "app.py"]
