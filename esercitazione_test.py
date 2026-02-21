from fastapi.testclient import TestClient
from esercitazione import app

client = TestClient(app)


def test_predict_endpoint():
    payload = {"msg": "I love my  life!"}
    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    data = response.json()

    # Controlliamo struttura risposta
    assert isinstance(data, list)
    assert "label" in data[0]
    assert "score" in data[0]

    # Controlliamo che la label sia valida
    assert data[0]["label"] in ["positive", "neutral", "negative"]
