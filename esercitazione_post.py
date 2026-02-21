import pandas as pd
import requests
import time
import random
from transformers import pipeline
from datasets import load_dataset

URL = "http://api:8000/predict"
model_path = "cardiffnlp/twitter-roberta-base-sentiment-latest"
dataset = load_dataset("tweet_eval", "sentiment")

sentiment_task = pipeline("sentiment-analysis", model=model_path, tokenizer=model_path)
# 1. Caricamento e pulizia rapida

df = dataset["test"].to_pandas()[:10]
valori_corretti = []
valori_predetti = []
ACCURACY_THRESHOLD = 0.60   # soglia minima per considerare il modello accettabile

label_map = {
    0: "negative",
    1: "neutral",
    2: "positive"
}
# 2. Ciclo di invio
for index, row in df.iterrows():
    time.sleep(random.uniform(0.1, 0.5))
    testo = str(row["clean_text"])
    valore_reale = row["category"]

    # Prepariamo il payload per l'API
    payload = {"msg": testo}


    response = requests.post(URL, json=payload)

    if response.status_code == 200:
        previsione = response.json()[0]
        print(previsione)
        print(f"[{index}] Testo: {testo[:50]}...")
        print(f"    - Reale: {valore_reale} | Predetto: {previsione['label']} (Conf: {previsione['score']:.2f})")
    else:
        print(f"[{index}] Errore API: {response.status_code}")
        time.sleep(1)


# 2. Ciclo di invio
for index, row in df.iterrows():
    time.sleep(random.uniform(0.1, 0.5))
    testo = str(row["clean_text"])
    valore_reale = row["category"]

    # Prepariamo il payload per l'API
    payload = {"msg": testo}


    response = requests.post(URL, json=payload)

    if response.status_code == 200:
        previsione = response.json()[0]
        print(previsione)
        print(f"[{index}] Testo: {testo[:50]}...")
        print(f"    - Reale: {valore_reale} | Predetto: {previsione['label']} (Conf: {previsione['score']:.2f})")
    else:
        print(f"[{index}] Errore API: {response.status_code}")
        time.sleep(1)
