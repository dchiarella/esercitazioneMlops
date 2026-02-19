import pandas as pd
import requests
import time
import random

# Configurazione
URL = "http://127.0.0.1:8000/interference"  # Assicurati che l'endpoint sia lo stesso di app.py
CSV_FILE = "Twitter_Data.csv"

# 1. Caricamento e pulizia rapida

df = pd.read_csv(CSV_FILE)



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
