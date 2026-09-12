# Interview Studio — Fault-Tolerant AI Interview Architecture

The interview engine is intentionally hybrid.

## Online path

Gemini generates questions and evaluates answers when available. The adaptive controller supplies role, experience, interview type, domain, current difficulty, interview source, previous questions, previous scores, and permitted resume/JD context.

## Offline path

When Gemini is unavailable, quota-limited, rate-limited, or returns invalid output:

1. The curated question bank supplies a unique question.
2. Questions are selected for the current domain and current difficulty.
3. Recent topics and exact/lexical duplicates are avoided.
4. The local evaluator calculates observable NLP signals.
5. The readiness Random Forest can continue using the normalized feature vector.
6. The final report remains available through the deterministic report fallback.

The UI/API reports `source=question-bank` for fallback questions and `source=local` for local evaluation so academic demonstrations do not falsely claim that Gemini evaluated an answer when it did not.

## Question bank

There are 30 curated questions per supported domain (three variants across ten core topics), with difficulty and expected-concept metadata. The bank is stored in `backend/app/interview/question_bank.py` and can be expanded without changing the interview API.

## Retry safety

A duplicate answer submission for the same question is treated idempotently. A failed Gemini request never reuses the immediately previous question.


## Persistent interview state

The active question is persisted in the `interviews` table. This prevents refreshes,
network retries, double-clicks, or an LLM failure from changing the question shown to
the candidate. Submitted answers also retain question-source and topic provenance.

## Recovery behavior

If Gemini cannot generate a question, the service retries with duplicate protection and
then selects an unused question from the selected domain. If an unexpected exception
occurs after an answer is saved, the API performs an emergency local recovery instead
of returning an unhandled 500 response. The frontend can also recover authoritative
interview state from the server after a transient submission failure.

## Local evaluation

When Gemini evaluation is unavailable, the local evaluator uses question-bank concepts
when available, lexical/TF-IDF similarity, concept coverage, technical evidence,
answer-length and communication signals. It is explicitly marked `source=local`; it
does not pretend to provide the same qualitative judgment as Gemini.
