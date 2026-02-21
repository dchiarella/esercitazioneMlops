import pandas as pd
import sys

from datasets import load_dataset
from transformers import pipeline
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

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
for index, row in df.iterrows():
    testo = str(row["text"])
    valore_reale = row["label"]

    # Saltiamo se label non valida
    if valore_reale not in label_map:
        continue

    prediction = sentiment_task(testo)[0]["label"]

    valori_corretti.append(label_map[valore_reale])
    valori_predetti.append(prediction)


accuracy = accuracy_score(valori_corretti, valori_predetti)

print(" RISULTATI VALIDAZIONE")
print("----------------------------")
print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:")
print(classification_report(valori_corretti, valori_predetti))


if accuracy < ACCURACY_THRESHOLD:
    print("Accuracy sotto soglia! Blocco pipeline.")
    sys.exit(1)
else:
    print("Accuracy sopra soglia. Validazione superata.")
    sys.exit(0)