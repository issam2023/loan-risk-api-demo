FROM python:3.10-slim

WORKDIR /service

COPY requirements-runtime.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ app/
COPY models/ models/

ENV MODEL_PATH=models/loan_default_model_v1.joblib
ENV METADATA_PATH=models/loan_default_model_v1.json
ENV MAX_BATCH=500

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
