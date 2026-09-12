from __future__ import annotations

import csv
import math
import re
from pathlib import Path
from typing import Any

from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[3]
DATASET = ROOT / "data" / "readiness_dataset.csv"

_MODEL = None
_METRICS = None

STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "to", "of", "in", "on", "for",
    "and", "or", "with", "how", "what", "why", "when", "where", "can", "could", "would",
    "you", "your", "i", "it", "this", "that", "be", "do", "does", "from", "as", "by",
}
FILLERS = {"um", "uh", "like", "basically", "actually", "literally", "you know", "sort of", "kind of"}


def _load_dataset():
    rows = []
    with DATASET.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append({k: float(v) for k, v in row.items()})
    features = [
        "technical_score", "correctness", "relevance", "clarity", "communication",
        "semantic_similarity", "concept_coverage", "answer_length", "filler_ratio",
    ]
    X = [[r[k] for k in features] for r in rows]
    y = [r["readiness_score"] for r in rows]
    return X, y, features


def get_readiness_model():
    global _MODEL, _METRICS
    if _MODEL is not None:
        return _MODEL, _METRICS
    X, y, features = _load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=220, random_state=42, max_depth=10, min_samples_leaf=3)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    _MODEL = model
    _METRICS = {
        "model": "Random Forest Regressor",
        "features": features,
        "test_size": len(X_test),
        "mae": round(float(mean_absolute_error(y_test, pred)), 2),
        "r2": round(float(r2_score(y_test, pred)), 3),
    }
    return _MODEL, _METRICS


def text_features(question: str, answer: str) -> dict[str, Any]:
    q = (question or "").strip()
    a = (answer or "").strip()
    words = re.findall(r"[A-Za-z0-9+#.]+", a.lower())
    q_words = [w for w in re.findall(r"[A-Za-z0-9+#.]+", q.lower()) if w not in STOPWORDS and len(w) > 2]
    answer_set = set(words)
    keyword_hits = sorted(set(q_words) & answer_set)
    coverage = round((len(keyword_hits) / max(1, len(set(q_words)))) * 100, 1)
    try:
        tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        matrix = tfidf.fit_transform([q, a])
        similarity = float((matrix[0] @ matrix[1].T).toarray()[0][0])
    except ValueError:
        similarity = 0.0
    filler_count = 0
    lower = a.lower()
    for filler in FILLERS:
        filler_count += len(re.findall(r"\b" + re.escape(filler) + r"\b", lower))
    word_count = len(words)
    return {
        "word_count": word_count,
        "unique_words": len(set(words)),
        "keyword_hits": keyword_hits[:12],
        "concept_coverage": coverage,
        "semantic_similarity": round(similarity, 3),
        "filler_count": filler_count,
        "filler_ratio": round(filler_count / max(1, word_count), 3),
        "estimated_answer_seconds": round(max(8, word_count / 2.15), 1),
        "estimated_wpm": 129,
    }


def analyze_answer(question: str, answer: str, evaluation: dict[str, Any]) -> dict[str, Any]:
    features = text_features(question, answer)
    technical = float(evaluation.get("technical_knowledge", 0) or 0) * 10
    correctness = float(evaluation.get("correctness", 0) or 0) * 10
    relevance = float(evaluation.get("relevance", 0) or 0) * 10
    clarity = float(evaluation.get("clarity", 0) or 0) * 10
    communication = float(evaluation.get("communication", 0) or 0) * 10
    model, _ = get_readiness_model()
    vector = [[
        technical, correctness, relevance, clarity, communication,
        features["semantic_similarity"] * 100,
        features["concept_coverage"],
        min(250, features["word_count"]),
        features["filler_ratio"] * 100,
    ]]
    prediction = float(model.predict(vector)[0])
    prediction = max(0, min(100, prediction))
    return {
        **features,
        "technical_score": round(technical, 1),
        "readiness_prediction": round(prediction, 1),
        "communication_signal": "Strong" if communication >= 75 else "Developing" if communication >= 50 else "Needs practice",
    }


def aggregate_readiness(analyses: list[dict[str, Any]], evaluations: list[dict[str, Any]]) -> dict[str, Any]:
    if not analyses:
        return {"score": 0, "label": "Not enough data", "trend": 0, "metrics": get_readiness_model()[1]}
    score = sum(x["readiness_prediction"] for x in analyses) / len(analyses)
    if len(analyses) > 1:
        trend = analyses[-1]["readiness_prediction"] - analyses[0]["readiness_prediction"]
    else:
        trend = 0
    label = "Strong" if score >= 80 else "Good" if score >= 65 else "Developing" if score >= 50 else "Needs Improvement"
    return {
        "score": round(score, 1),
        "label": label,
        "trend": round(trend, 1),
        "average_semantic_similarity": round(sum(x["semantic_similarity"] for x in analyses) / len(analyses), 3),
        "average_concept_coverage": round(sum(x["concept_coverage"] for x in analyses) / len(analyses), 1),
        "average_filler_ratio": round(sum(x["filler_ratio"] for x in analyses) / len(analyses), 3),
        "metrics": get_readiness_model()[1],
    }
