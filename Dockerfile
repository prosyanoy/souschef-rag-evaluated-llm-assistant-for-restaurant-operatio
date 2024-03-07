FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY souschef souschef
COPY app app
COPY data data
COPY eval eval
ENV SOUSCHEF_INDEX_DIR=/app/.cache/index SOUSCHEF_DB_PATH=/app/.cache/souschef.db
RUN python -m souschef.cli ingest
EXPOSE 8000 8501
CMD ["uvicorn", "souschef.api:app", "--host", "0.0.0.0", "--port", "8000"]
