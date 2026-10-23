FROM python:3.12-slim
WORKDIR /srv
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV DB_PATH=/data/index.db UPLOAD_DIR=/data/uploads
VOLUME /data
EXPOSE 8000
CMD ["uvicorn", "rag.app:app", "--host", "0.0.0.0", "--port", "8000"]
