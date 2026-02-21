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

model_path = "cardiffnlp/twitter-roberta-base-sentiment-latest"
sentiment_task = pipeline("sentiment-analysis", model=model_path, tokenizer=model_path)
# mi carico il dataset
##
##df = pd.read_csv("Twitter_Data.csv")
##df = df.dropna()
# prendo solo i primi 100 valori e stampo qualche esempio ma solo perchè interessa a me non a scopo progettuale
##df_small = df[:100]


# ciclo e vedo il risultato
##for index, row in df_small.iterrows():
##    testo = row["clean_text"]
##    valore_reale = row["category"]

##    risultato = sentiment_task(testo)[0]["label"]

##    print(f"Frase: {testo}")

##    print(f"Valore reale: {valore_reale}")
##    print(f"risultato:{risultato}")

class Message(BaseModel):
    msg: str


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

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type="text/plain")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)