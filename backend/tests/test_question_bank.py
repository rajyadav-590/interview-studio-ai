from backend.app.interview.question_bank import QUESTION_BANK, find_question, find_similar_question


def test_question_bank_has_30_questions_per_domain():
    assert len(QUESTION_BANK) >= 14
    assert all(len(items) >= 30 for items in QUESTION_BANK.values())


def test_questions_are_unique_within_domain():
    for items in QUESTION_BANK.values():
        normalized = [item["question"].strip().lower() for item in items]
        assert len(normalized) == len(set(normalized))


def test_bank_lookup_and_similarity_lookup():
    item = QUESTION_BANK["Software Development"][0]
    assert find_question(item["question"]) is not None
    assert find_similar_question("Explain version control and why it matters in software development.") is not None
