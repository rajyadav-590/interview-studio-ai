# Interview Studio workspace guidance

- Keep LLM provider keys backend-only; never prefix them with `VITE_`.
- Preserve the API contract documented in `README.md` when changing frontend or backend behavior.
- Use structured JSON validated before persisting evaluations and reports.
- Run `npm --prefix frontend run build` after frontend changes.
- Run `python -m py_compile backend/app/main.py` after backend changes.
- Keep user ownership checks on every interview and report endpoint.
