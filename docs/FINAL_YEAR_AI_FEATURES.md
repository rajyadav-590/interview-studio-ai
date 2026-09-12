# Final-year AI feature map

## Core AI
1. LLM resume intelligence
2. LLM job-description intelligence
3. Resume/JD matching
4. Adaptive interview question generation
5. LLM answer evaluation
6. Final qualitative report

## Independent NLP layer
- TF-IDF semantic similarity between question and answer
- question concept/keyword coverage
- filler-word analysis
- answer length and vocabulary statistics

## Independent ML layer
- Random Forest readiness prediction
- reproducible 80/20 holdout evaluation
- MAE and R² exposed by `/api/ai/model-metrics`
- answer-level predictions exposed by `/api/interviews/{id}/ai-assessment`

## Optional deep learning research
- `ml/deep_learning_audio.py` provides a PyTorch CNN for log-mel audio experiments.
- This should only be trained/evaluated on consented/public data.
- It should classify communication signals defined by the research protocol, not personality, protected traits or mental state.

## Academic caution
The included readiness dataset is bootstrap/demo data generated for a runnable prototype. It must not be presented as a real human interview dataset. Before final submission, replace it with a documented public or consented dataset and report its provenance and limitations.


## Adaptive Interview Context Isolation (Upgrade)

The interview engine now distinguishes New Interview, Domain Practice, Candidate Profile, and Job-Specific sessions. New Interview sessions do not read saved resume/JD data. Candidate Profile and Job-Specific sessions explicitly load analyzed resume/JD context. A selected interview domain is passed to Gemini and enforced in the deterministic planner/prompt policy.

The session also maintains a current difficulty state. Strong answers (8/10+) increase difficulty, weak answers (<5/10) decrease it, and middle scores keep the current level. Previous questions and recent evaluations are provided to Gemini so question generation varies topic, format, depth, and difficulty.
