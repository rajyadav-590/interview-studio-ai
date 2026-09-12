# Interview Studio — Execution Guide

## Backend

From the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r backend\requirements.txt
```

Create `.env` from `.env.example` and set `GEMINI_API_KEY` locally. Never commit the key.

Start the API:

```powershell
uvicorn backend.app.main:app --reload
```

Health check:

```powershell
curl http://127.0.0.1:8000/api/health
```

## ML readiness model

```powershell
python ml\train_readiness_model.py
```

## Frontend

```powershell
cd frontend
npm install
npm run dev
```

Open the Vite URL, normally `http://localhost:5173`.

## Interview modes

### New interview
Only the selected role, experience, interview type, difficulty, domain, and question count are authoritative. Saved resume/JD data is ignored.

### Candidate Profile / Job Specific
The selected interview parameters are combined with the latest analyzed resume/JD context.

## Fault tolerance

If Gemini question generation fails, the backend selects an unused question from `data/question_bank/`. If Gemini answer evaluation fails, the local NLP evaluator continues the session. If an unexpected generation exception occurs after an answer is saved, the API attempts an emergency unique question-bank recovery instead of returning an unhandled server error.

## Testing fallback safely

Do not intentionally exhaust an API key. For a controlled offline test, temporarily remove or blank `GEMINI_API_KEY` in the local `.env`, restart the backend, and start a new interview. Restore the key afterward. The health endpoint should report that the LLM is not configured, and questions/evaluations should continue through local fallback.
