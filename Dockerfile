FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

COPY similarity_engine.py rag_engine.py api.py ./
COPY mqtt_bridge.py mqtt_simulator.py streamlit_app.py ./
COPY tests ./tests

RUN mkdir -p /app/data /app/artifacts

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
