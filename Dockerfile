FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip

# 🔥 PRIMA installiamo torch CPU
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

# 🔥 POI installiamo il resto
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "esercitazione:app", "--host", "0.0.0.0", "--port", "8000"]