# tests/test_app.py
import os
import json
import pytest
from unittest.mock import patch
from app import app, order_lookup_tool

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_health(client):
    rv = client.get("/health")
    assert rv.status_code == 200
    data = json.loads(rv.data)
    assert data["status"] == "ok"

def test_order_lookup_mock():
    # if ORDER_API_URL not set, fallback mock should return "mock"
    os.environ.pop("ORDER_API_URL", None)
    res = order_lookup_tool("1234")
    assert "mock" in res.lower()

@patch("app.safe_get")
def test_order_lookup_real(mock_safe_get):
    mock_safe_get.return_value = {"status": "shipped", "eta": "2025-12-05"}
    os.environ["ORDER_API_URL"] = "https://api.example.com"
    # call the tool with the patched safe_get
    res = order_lookup_tool("1234")
    assert "shipped" in res.lower()
