"""Curated offline interview question bank.

The bank is deliberately metadata-rich so it can act as a reliable fallback when
an LLM is unavailable. 30 questions are available per supported domain from ten
core topics (three different question styles per topic), with difficulty metadata
and expected concepts for deterministic evaluation.
"""
from __future__ import annotations

from typing import Any
from pathlib import Path
from difflib import SequenceMatcher


def _q(topic: str, concepts: list[str], difficulty: str, kind: str, question: str) -> dict[str, Any]:
    return {
        "topic": topic,
        "difficulty": difficulty,
        "type": kind,
        "question": question,
        "expected_concepts": concepts,
    }


DOMAIN_TOPICS: dict[str, list[tuple[str, list[str]]]] = {
    "Software Development": [
        ("Version control", ["git", "commit", "branch", "merge", "history"]),
        ("Object-oriented programming", ["class", "object", "encapsulation", "inheritance", "polymorphism"]),
        ("SOLID principles", ["single responsibility", "open closed", "liskov", "interface segregation", "dependency inversion"]),
        ("Software testing", ["unit test", "integration test", "test case", "mock", "regression"]),
        ("Debugging", ["reproduce", "logs", "breakpoint", "isolate", "root cause"]),
        ("REST APIs", ["http", "get", "post", "status code", "json"]),
        ("Error handling", ["exception", "validation", "logging", "error response", "recovery"]),
        ("Database design", ["normalization", "primary key", "foreign key", "index", "transaction"]),
        ("Code review", ["review", "readability", "bug", "security", "maintainability"]),
        ("Software architecture", ["separation of concerns", "modularity", "scalability", "layer", "dependency"]),
    ],
    "Web Development": [
        ("HTTP and web fundamentals", ["http", "request", "response", "status code", "client"]),
        ("HTML semantics", ["semantic", "html", "accessibility", "form", "structure"]),
        ("CSS responsive design", ["css", "flexbox", "grid", "media query", "responsive"]),
        ("JavaScript fundamentals", ["javascript", "scope", "closure", "promise", "event"]),
        ("DOM and browser events", ["dom", "event listener", "element", "event propagation"]),
        ("REST API integration", ["api", "fetch", "json", "authentication", "error handling"]),
        ("Frontend performance", ["lazy loading", "bundle", "caching", "rendering", "performance"]),
        ("Web security", ["xss", "csrf", "authentication", "authorization", "input validation"]),
        ("Accessibility", ["wcag", "keyboard", "aria", "screen reader", "contrast"]),
        ("Deployment", ["build", "environment variable", "hosting", "https", "deployment"]),
    ],
    "Python Development": [
        ("Python data structures", ["list", "tuple", "set", "dictionary", "complexity"]),
        ("Functions and scope", ["function", "parameter", "scope", "return", "lambda"]),
        ("Object-oriented Python", ["class", "object", "inheritance", "method", "encapsulation"]),
        ("Exceptions", ["try", "except", "finally", "exception", "custom"]),
        ("Iterators and generators", ["iterator", "generator", "yield", "lazy", "iteration"]),
        ("Decorators", ["decorator", "wrapper", "function", "closure", "@"]),
        ("Virtual environments", ["venv", "pip", "dependency", "requirements", "isolation"]),
        ("Testing Python", ["pytest", "unit test", "mock", "assert", "fixture"]),
        ("APIs with Python", ["fastapi", "flask", "endpoint", "request", "response"]),
        ("Python performance", ["profiling", "complexity", "memory", "caching", "optimization"]),
    ],
    "Data Structures & Algorithms": [
        ("Arrays and strings", ["array", "string", "index", "iteration", "complexity"]),
        ("Hash tables", ["hash", "key", "value", "collision", "lookup"]),
        ("Stacks and queues", ["stack", "queue", "lifo", "fifo", "operation"]),
        ("Linked lists", ["node", "pointer", "head", "traversal", "insertion"]),
        ("Trees", ["tree", "node", "root", "traversal", "binary search tree"]),
        ("Graphs", ["graph", "vertex", "edge", "bfs", "dfs"]),
        ("Sorting", ["sorting", "merge sort", "quick sort", "comparison", "complexity"]),
        ("Searching", ["binary search", "linear search", "sorted", "logarithmic", "complexity"]),
        ("Dynamic programming", ["dynamic programming", "subproblem", "memoization", "tabulation", "optimal"]),
        ("Big-O complexity", ["time complexity", "space complexity", "big o", "input size", "growth"]),
    ],
    "Data Science": [
        ("Data preprocessing", ["missing values", "encoding", "scaling", "outliers", "cleaning"]),
        ("Exploratory data analysis", ["eda", "distribution", "correlation", "visualization", "outlier"]),
        ("Statistics", ["mean", "median", "variance", "standard deviation", "distribution"]),
        ("Probability", ["probability", "conditional", "independence", "bayes", "event"]),
        ("Feature engineering", ["feature", "transformation", "encoding", "selection", "domain"]),
        ("Model evaluation", ["train", "test", "cross validation", "metric", "generalization"]),
        ("Data leakage", ["leakage", "training", "test", "future information", "validation"]),
        ("Pandas", ["dataframe", "series", "groupby", "merge", "filter"]),
        ("NumPy", ["array", "vectorization", "broadcasting", "shape", "numpy"]),
        ("Data visualization", ["chart", "plot", "axis", "distribution", "insight"]),
    ],
    "Machine Learning": [
        ("Supervised learning", ["labeled data", "features", "target", "training", "prediction"]),
        ("Unsupervised learning", ["unlabeled", "clustering", "patterns", "groups", "dimensionality"]),
        ("Linear regression", ["regression", "continuous", "features", "coefficient", "loss"]),
        ("Logistic regression", ["classification", "probability", "sigmoid", "binary", "threshold"]),
        ("Decision trees", ["tree", "split", "feature", "impurity", "leaf"]),
        ("Random forests", ["ensemble", "trees", "bootstrap", "random features", "voting"]),
        ("Overfitting", ["overfitting", "generalization", "validation", "regularization", "complexity"]),
        ("Feature scaling", ["scaling", "standardization", "normalization", "distance", "features"]),
        ("Cross-validation", ["cross validation", "fold", "training", "validation", "generalization"]),
        ("Classification metrics", ["precision", "recall", "f1", "confusion matrix", "accuracy"]),
    ],
    "Deep Learning": [
        ("Neural networks", ["neuron", "layer", "weights", "activation", "forward"]),
        ("Backpropagation", ["backpropagation", "gradient", "loss", "chain rule", "weights"]),
        ("Activation functions", ["relu", "sigmoid", "tanh", "activation", "nonlinearity"]),
        ("Loss functions", ["loss", "mse", "cross entropy", "optimization", "error"]),
        ("Optimizers", ["gradient descent", "adam", "learning rate", "optimizer", "update"]),
        ("CNNs", ["convolution", "filter", "pooling", "feature map", "image"]),
        ("RNNs", ["sequence", "hidden state", "recurrent", "time step", "sequence"]),
        ("Transformers", ["attention", "self attention", "token", "embedding", "transformer"]),
        ("Regularization", ["dropout", "regularization", "overfitting", "weight decay", "generalization"]),
        ("Training stability", ["batch", "learning rate", "normalization", "gradient", "convergence"]),
    ],
    "Data Analytics": [
        ("Data cleaning", ["missing values", "duplicates", "outliers", "validation", "cleaning"]),
        ("SQL analytics", ["select", "join", "group by", "having", "aggregate"]),
        ("KPI design", ["kpi", "metric", "target", "business objective", "measurement"]),
        ("Dashboard design", ["dashboard", "visualization", "filter", "kpi", "user"]),
        ("Excel analysis", ["pivot table", "formula", "lookup", "chart", "spreadsheet"]),
        ("Power BI", ["power bi", "model", "measure", "dax", "dashboard"]),
        ("Trend analysis", ["trend", "time series", "growth", "seasonality", "comparison"]),
        ("Customer analysis", ["customer", "segmentation", "behavior", "retention", "cohort"]),
        ("A/B testing", ["control", "treatment", "hypothesis", "statistical significance", "experiment"]),
        ("Data storytelling", ["insight", "context", "visual", "audience", "recommendation"]),
    ],
    "Database / SQL": [
        ("SQL joins", ["join", "inner join", "left join", "foreign key", "matching"]),
        ("Aggregation", ["group by", "having", "count", "sum", "aggregate"]),
        ("Indexes", ["index", "query", "lookup", "performance", "write overhead"]),
        ("Normalization", ["normalization", "redundancy", "functional dependency", "normal form", "table"]),
        ("Transactions", ["transaction", "acid", "commit", "rollback", "consistency"]),
        ("Isolation", ["isolation", "dirty read", "non repeatable", "phantom", "concurrency"]),
        ("Query optimization", ["explain", "index", "execution plan", "join", "performance"]),
        ("Constraints", ["primary key", "foreign key", "unique", "check", "not null"]),
        ("Views and procedures", ["view", "stored procedure", "query", "security", "reuse"]),
        ("Database security", ["authentication", "authorization", "sql injection", "least privilege", "parameterized"]),
    ],
    "AI / NLP": [
        ("Tokenization", ["token", "text", "word", "subword", "sequence"]),
        ("Embeddings", ["embedding", "vector", "semantic", "similarity", "representation"]),
        ("TF-IDF", ["tf idf", "term frequency", "inverse document frequency", "weight", "document"]),
        ("Text classification", ["classification", "label", "features", "training", "prediction"]),
        ("Named entity recognition", ["entity", "person", "organization", "location", "sequence"]),
        ("Sentiment analysis", ["sentiment", "positive", "negative", "text", "classification"]),
        ("Attention", ["attention", "query", "key", "value", "context"]),
        ("LLMs", ["language model", "token", "transformer", "context", "generation"]),
        ("Prompt engineering", ["prompt", "context", "instruction", "output", "constraint"]),
        ("NLP evaluation", ["precision", "recall", "f1", "ground truth", "evaluation"]),
    ],
    "Computer Vision": [
        ("Image preprocessing", ["resize", "normalize", "augmentation", "image", "pixels"]),
        ("Convolution", ["convolution", "kernel", "filter", "feature map", "stride"]),
        ("Image classification", ["class", "label", "cnn", "training", "prediction"]),
        ("Object detection", ["bounding box", "object", "confidence", "iou", "detection"]),
        ("Segmentation", ["segmentation", "pixel", "mask", "semantic", "instance"]),
        ("Data augmentation", ["augmentation", "flip", "rotate", "crop", "generalization"]),
        ("Transfer learning", ["pretrained", "fine tuning", "feature extractor", "weights", "dataset"]),
        ("IoU and detection metrics", ["iou", "precision", "recall", "threshold", "bounding box"]),
        ("OpenCV", ["opencv", "image", "video", "filter", "computer vision"]),
        ("Deployment", ["inference", "latency", "model", "optimization", "deployment"]),
    ],
    "Cybersecurity": [
        ("Authentication", ["authentication", "identity", "password", "mfa", "token"]),
        ("Authorization", ["authorization", "role", "permission", "access control", "least privilege"]),
        ("Web vulnerabilities", ["xss", "csrf", "sql injection", "input validation", "sanitization"]),
        ("Encryption", ["encryption", "symmetric", "asymmetric", "key", "cipher"]),
        ("Hashing", ["hash", "password", "salt", "one way", "integrity"]),
        ("Network security", ["firewall", "network", "port", "protocol", "segmentation"]),
        ("Secure coding", ["validation", "secrets", "dependency", "logging", "secure"]),
        ("Incident response", ["incident", "detect", "contain", "eradicate", "recover"]),
        ("Threat modeling", ["threat", "asset", "risk", "attack surface", "mitigation"]),
        ("Security monitoring", ["log", "siem", "alert", "anomaly", "monitoring"]),
    ],
    "DevOps / Cloud": [
        ("CI/CD", ["continuous integration", "continuous deployment", "pipeline", "build", "test"]),
        ("Docker", ["container", "image", "dockerfile", "registry", "isolation"]),
        ("Kubernetes", ["pod", "deployment", "service", "cluster", "container"]),
        ("Cloud fundamentals", ["cloud", "compute", "storage", "network", "scaling"]),
        ("AWS architecture", ["aws", "ec2", "s3", "iam", "vpc"]),
        ("Infrastructure as code", ["infrastructure as code", "terraform", "configuration", "version control", "automation"]),
        ("Monitoring", ["metrics", "logs", "alerts", "observability", "dashboard"]),
        ("Secrets management", ["secret", "environment variable", "vault", "key", "rotation"]),
        ("Scalability", ["load balancing", "horizontal scaling", "vertical scaling", "autoscaling", "capacity"]),
        ("Deployment strategies", ["blue green", "canary", "rollback", "release", "deployment"]),
    ],
    "General Software Engineering": [
        ("Programming fundamentals", ["variable", "function", "condition", "loop", "data structure"]),
        ("OOP", ["class", "object", "encapsulation", "inheritance", "polymorphism"]),
        ("Version control", ["git", "commit", "branch", "merge", "history"]),
        ("Testing", ["unit test", "integration test", "assertion", "regression", "coverage"]),
        ("Debugging", ["reproduce", "logs", "breakpoint", "root cause", "fix"]),
        ("APIs", ["api", "http", "endpoint", "request", "response"]),
        ("Databases", ["table", "primary key", "foreign key", "query", "index"]),
        ("Design principles", ["modularity", "separation of concerns", "coupling", "cohesion", "maintainability"]),
        ("Security basics", ["authentication", "authorization", "validation", "encryption", "least privilege"]),
        ("Problem solving", ["requirements", "break down", "tradeoff", "test", "validate"]),
    ],
}


def build_question_bank() -> dict[str, list[dict[str, Any]]]:
    """Build a 30-question-per-domain offline bank.

    Each topic has three distinct forms: concept, scenario, and implementation /
    trade-off. The bank is deterministic, metadata-rich, and large enough for
    the application's maximum 20-question interview while leaving headroom.
    """
    bank: dict[str, list[dict[str, Any]]] = {}
    for domain, topics in DOMAIN_TOPICS.items():
        questions: list[dict[str, Any]] = []
        for idx, (topic, concepts) in enumerate(topics):
            difficulty = "Easy" if idx < 3 else ("Medium" if idx < 7 else "Hard")
            questions.append(_q(
                topic, concepts, difficulty, "conceptual",
                f"What is {topic}, and why is it important in {domain.lower()}? Give a concise example.",
            ))
            questions.append(_q(
                topic, concepts, difficulty, "scenario",
                f"Imagine you are building a {domain.lower()} project and need to use {topic}. Describe how you would apply it and one problem you would watch for.",
            ))
            questions.append(_q(
                topic, concepts, difficulty, "implementation",
                f"For {topic} in {domain.lower()}, explain one practical implementation approach and one important trade-off or design consideration.",
            ))
        bank[domain] = questions
    return bank


def load_question_bank() -> dict[str, list[dict[str, Any]]]:
    """Load the reviewed bank from data/question_bank when packaged.

    The generated in-code bank remains a safe fallback for development, while
    JSON files make the academic dataset explicit, editable, and auditable.
    """
    import json
    root = Path(__file__).resolve().parents[3]
    directory = root / "data" / "question_bank"
    loaded: dict[str, list[dict[str, Any]]] = {}
    if directory.exists():
        for path in sorted(directory.glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                domain = payload.get("domain")
                questions = payload.get("questions")
                if isinstance(domain, str) and isinstance(questions, list) and questions:
                    loaded[domain] = questions
            except Exception:
                continue
    return loaded or build_question_bank()


QUESTION_BANK = load_question_bank()


def normalize_question(text: str) -> str:
    import re
    text = re.sub(r"[^a-z0-9\s]", " ", (text or "").lower())
    return re.sub(r"\s+", " ", text).strip()


def find_question(question: str) -> dict[str, Any] | None:
    target = normalize_question(question)
    for items in QUESTION_BANK.values():
        for item in items:
            if normalize_question(item["question"]) == target:
                return item
    return None


def find_similar_question(question: str, threshold: float = 0.28) -> dict[str, Any] | None:
    """Find the closest bank item for local evaluation metadata.

    Gemini questions are not necessarily byte-for-byte equal to bank questions.
    A lightweight lexical similarity bridge lets the local evaluator reuse the
    bank's expected concepts without claiming semantic equivalence.
    """
    target = normalize_question(question)
    if not target:
        return None
    target_tokens = set(target.split())
    best = None
    best_score = 0.0
    for items in QUESTION_BANK.values():
        for item in items:
            candidate = normalize_question(item["question"])
            if candidate == target:
                return item
            cand_tokens = set(candidate.split())
            jaccard = len(target_tokens & cand_tokens) / max(1, len(target_tokens | cand_tokens))
            sequence = SequenceMatcher(None, target, candidate).ratio()
            score = max(jaccard, sequence * 0.85)
            if score > best_score:
                best_score = score
                best = item
    return best if best_score >= threshold else None


def questions_for_domain(domain: str) -> list[dict[str, Any]]:
    if domain in QUESTION_BANK:
        return QUESTION_BANK[domain]
    return QUESTION_BANK["General Software Engineering"]
