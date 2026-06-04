from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_detect_english():
    response = client.post(
        "/detect",
        json={"text": "The quick brown fox jumps over the lazy dog. This is a simple English sentence."},
    )
    assert response.status_code in [200, 503]
    if response.status_code == 200:
        data = response.json()
        assert "detected_language" in data
        assert "top_languages" in data
        assert "explanation" in data
        assert len(data["top_languages"]) == 3
        assert data["detected_language"] == "English"


def test_detect_spanish():
    response = client.post(
        "/detect",
        json={"text": "El rápido zorro marrón salta sobre el perro perezoso. Esta es una oración en español."},
    )
    assert response.status_code in [200, 503]
    if response.status_code == 200:
        assert response.json()["detected_language"] == "Spanish"


def test_detect_too_short():
    response = client.post("/detect", json={"text": "hi"})
    assert response.status_code == 422


def test_languages_endpoint():
    response = client.get("/api/languages")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_model_info():
    response = client.get("/api/model-info")
    assert response.status_code == 200
