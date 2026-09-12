import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    response = client.get('/api/health')
    assert response.status_code == 200
    assert response.json()['status'] == 'ok'


def test_register_and_create_interview():
    response = client.post('/api/auth/register', json={
        'email': 'test-suite@example.com',
        'password': 'secret123',
        'name': 'Test Suite',
    })
    assert response.status_code in (200, 409)
    if response.status_code == 409:
        response = client.post('/api/auth/login', json={'email': 'test-suite@example.com', 'password': 'secret123'})
    token = response.json()['token']
    response = client.post('/api/interviews', headers={'Authorization': f'Bearer {token}'}, json={
        'role': 'Software Engineer',
        'experience': '1–2 Years',
        'interview_type': 'Technical',
        'difficulty': 'Medium',
        'total_questions': 5,
    })
    assert response.status_code == 200
    assert response.json()['next_question']
