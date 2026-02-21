from datasets import load_dataset
from transformers import pipeline
import pandas as pd
import pickle
from fastapi import  FastAPI
from pydantic import BaseModel
import pickle
from prometheus_client import Counter, Histogram, generate_latest
from fastapi.responses import Response
import time
from fastapi.responses import HTMLResponse 
app = FastAPI()

# importo delle statistiche di prometheus
TOTALE_RICHIESTE = Counter(
    "totale_richieste_api",
    "Numero totale di richieste ricevute dall'API"
)

TOTALE_PREDIZIONI = Counter(
    "totale_predizioni_per_classe",
    "Numero di predizioni suddivise per classe di sentiment",
    ["classe"]
)

TEMPO_RISPOSTA = Histogram(
    "tempo_risposta_secondi",
    "Tempo di risposta dell'API in secondi"
)

SOMMA_CONFIDENCE = Counter(
    "somma_confidence_per_classe",
    "Somma dei punteggi di confidence per classe",
    ["classe"]
)

#Uilizzo un modello pre impostato
model_path = "cardiffnlp/twitter-roberta-base-sentiment-latest"
sentiment_task = pipeline("sentiment-analysis", model=model_path, tokenizer=model_path)

# Creo una classe che verrà usata per chiamare il controller
class Message(BaseModel):
    msg: str

# La chiamata post oltre che utilizzare il modello verificare il sentimento aggiorna anche le statistiche
@app.post("/predict")
async def predict(messaggio: Message):
  
    start_time = time.time()
    TOTALE_RICHIESTE.inc()

    prediction = sentiment_task(messaggio.msg)

    label = prediction[0]["label"].lower()
    score = prediction[0]["score"]

    TOTALE_PREDIZIONI.labels(classe=label).inc()
    SOMMA_CONFIDENCE.labels(classe=label).inc(score)

    TEMPO_RISPOSTA.observe(time.time() - start_time)
    return prediction


#Questa serve per le metriche, prende i COunter e li espone in maniera che Prometheus li possa leggere
@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type="text/plain")

#Ho creato questa per poter provare su hf il tutto
@app.get("/", response_class=HTMLResponse)
async def home():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)