FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=200 -r requirements.txt
RUN pip install --no-cache-dir --default-timeout=200 torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY app ./app

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]