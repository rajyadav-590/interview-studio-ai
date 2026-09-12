"""Train/evaluate the Interview Studio readiness model.

Usage: python ml/train_readiness_model.py
The application also trains lazily from data/readiness_dataset.csv.
Replace the bootstrap dataset with ethically collected/public interview data
before reporting final academic metrics.
"""
from backend.app.ml.analytics import get_readiness_model
print(get_readiness_model()[1])
