# Interview Studio AI — Final Year Project Edition

An AI-powered, adaptive, multimodal interview assessment platform for resume-aware practice, job-description matching, interview evaluation, NLP analytics and machine-learning-based readiness prediction.

## What was upgraded

- Resume intelligence and JD matching remain powered by the existing LLM layer.
- Added deterministic NLP analytics: TF-IDF semantic similarity, question keyword/concept coverage and filler-word analysis.
- Added a Random Forest readiness prediction model with reproducible holdout metrics.
- Added an **AI Insights** dashboard for answer-level signals, readiness prediction, priority areas and model methodology.
- Added a clean ML research area under `ml/` and a bootstrap dataset under `data/`.
- Added an optional PyTorch CNN scaffold for future log-mel audio experiments. It is isolated so the main application does not require PyTorch.
- Removed duplicate project copies, generated environments, videos, databases and secrets from the submission package.

## Project structure

```text
Interview-Studio-AI/
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── llm/
│   │   └── ml/
│   ├── tests/
│   └── requirements.txt
├── frontend/
│   └── src/
├── data/
│   └── readiness_dataset.csv
├── ml/
│   ├── train_readiness_model.py
│   └── deep_learning_audio.py
├── requirements-ml.txt
├── .env.example
└── README.md
```

## Run locally

### Backend

```bash
cd backend
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
cd ..
uvicorn backend.app.main:app --reload
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

The frontend uses `VITE_API_URL` when set; otherwise it uses `/api`.

## Environment

Copy `.env.example` to `.env` in the project root. At minimum configure a strong `JWT_SECRET`. Configure the Gemini/LLM key required by the existing LLM service if you want live LLM generation and evaluation. The application has explicit fallback behavior when the LLM is unavailable.

**Never commit `.env` or real API keys.**

## ML component

The readiness model is a Random Forest regressor. It uses:

- technical score
- correctness
- relevance
- clarity
- communication
- semantic similarity
- concept coverage
- answer length
- filler-word ratio

Run:

```bash
python ml/train_readiness_model.py
```

The bundled CSV is a **bootstrap/demo dataset**, not a claim of real-world interview performance. For the final academic evaluation, replace it with a consented or public-domain dataset and document the source, preprocessing, train/test split and limitations.

## Optional deep-learning research

`ml/deep_learning_audio.py` contains a PyTorch CNN scaffold for log-mel spectrogram experiments. If used, install `requirements-ml.txt`, collect only consented audio, and evaluate communication-signal classes rather than personality, emotion, protected traits or mental state.

## Human intervention required during setup

1. Install Python and Node.js on the machine.
2. Create `.env` from `.env.example` and add the student's own API credentials if live LLM mode is required.
3. Replace the bootstrap ML dataset with a defensible public/consented dataset before claiming final research metrics.
4. Test microphone/camera permissions in the browser before a live interview.
5. For deployment, configure a production database, HTTPS and secure secrets.

The rest of the application code is included and can be executed after those environment-specific steps.
