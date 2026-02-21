---
title: Sentiment Analysis MLOps
colorFrom: blue
colorTo: indigo
sdk: docker
app_file: app.py
pinned: false
app_port: 8000
---

Progetto MLOps completo per l’analisi del sentiment basato su un modello pre-addestrato HuggingFace, con pipeline CI/CD, monitoraggio runtime e validazione automatica.

---

Obiettivo del progetto

Implementare un sistema MLOps che:

- Utilizza un modello pre-addestrato di HuggingFace
- Espone il modello tramite API FastAPI
- Containerizza l'applicazione con Docker
- Implementa metriche con Prometheus
- Visualizza metriche con Grafana
- Valida automaticamente il modello in CI
- Effettua deploy automatico su HuggingFace Spaces

---


Modello pre-addestrato: cardiffnlp/twitter-roberta-base-sentiment-latest


Classi previste:
- negative
- neutral
- positive


---

Architettura

Abbiamo un Docker che esponde il modello su uno specifico URL tramite fastAPI, tale docker contiene al suo interno delle metriche specifiche 
  
    "Numero totale di richieste ricevute dall'API"
    "Numero di predizioni suddivise per classe di sentiment"
    "Tempo di risposta dell'API in secondi"
    "Somma dei punteggi di confidence per classe"
    Esposte tramite Prometheus e Grafana, in particolare abbiamo chiamato il container dove gira il modello api che oltre il modello stesso esporra le metriche. Prometheus tramite il suo yaml va a verificare queste API.
    Grafana deve essere configurato, andando ad aggiungere come datasource prometheus all'url esposto, nel nostro caso prometheus:9000 ( il container è chiamato cosi).

Per testare il tutto è stato creato un client esercitazione_post che utilizza questo dataset :
  from datasets import load_dataset
  dataset = load_dataset("tweet_eval", "sentiment")
e fa una chiamata al controller esposto (/predict) ogni due secondi. Questo ci permette di verificare su grafana le metriche

E' stato anche fatta una classe di validazione validate.py che prende lo stesso dataset precedente, prende 10 record random e verifica, utilizzando lo stesso modello che l'accuracy sia sopra la soglia dello 0.60 ( so benissimo che 10 record sono pochi, è stato fatto a fine di esperienza).

E' stato creato in file di test ( test_esercitazione.py) che chiama il modello e verifica che la risposta sia fra le classi previste.
Infine tramite secret è con l'aiuto di Gemini è stato fatto un deploy su HuggingFace.

Per avviare il progetto posizionarsi nella cartella principale e avviare il tutto con docker-compose up --build.

Esiste una pipeline che 
"Installazione dipendenze"
"Esecuzione unit test"
"Validazione accuracy modello"
"Build Docker"
"Deploy automatico su HuggingFace Space"




