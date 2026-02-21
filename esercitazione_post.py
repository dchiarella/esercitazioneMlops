import pandas as pd
import requests
import time
import random
# URL che chiamiamo espsota tramite container
URL = "http://api:8000/predict"

# Usiamo TweetEval solo come dataset locale semplice
from datasets import load_dataset
dataset = load_dataset("tweet_eval", "sentiment")
df = dataset["test"].to_pandas().sample(frac=1,  random_state=42)

print("Avvio simulazione traffico...")
print("Attendo che l'API sia pronta...")

# Mi sono aiutato con Gemini, per utilizzare questo trucco per aspettare che il container che espone il servizio sia avviato prima di andare avanti, ci vorrebbe un timeout che lo blocca dopo un certo tempo
while True:
    try:
        r = requests.get("http://api:8000/metrics")
        if r.status_code == 200:
            print("API pronta ")
            break
    except:
        pass

    time.sleep(2)
# ciclo sul daset e richiamo il controller 
for index, row in df.iterrows():
    time.sleep(random.uniform(0.1, 0.3))

    testo = str(row["text"])
    payload = {"msg": testo}

    try:
        response = requests.post(URL, json=payload)

        if response.status_code == 200:
            previsione = response.json()[0]
            print(f"[{index}] {previsione['label']} ({previsione['score']:.2f})")
        else:
            print(f"[{index}] Errore API: {response.status_code}")

    except Exception as e:
        print(f"[{index}] Errore connessione: {e}")

print("Simulazione completata")