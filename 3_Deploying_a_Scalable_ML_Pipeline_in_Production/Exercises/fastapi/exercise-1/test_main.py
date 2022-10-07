from fastapi.testclient import TestClient
import json

from main import app

client = TestClient(app)

def test_get():
    response = client.get("/items")
    assert response.status_code == 200
    #assert response.json() == [{"item": "Foo"}, {"item": "Bar"}]

def test_post():
    data = json.dumps({"item": "foo"})
    response = client.post("/1?q=5", data=data)
    assert response.status_code == 200
    assert response.json()["item_id"] == 1
    assert response.json()["q"] == "5"
    assert response.json()["item"]["item"] == "foo"