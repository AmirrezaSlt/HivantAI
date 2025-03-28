FROM python:3.13.1-slim
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
WORKDIR /llm
COPY agent ./agent
COPY providers ./providers