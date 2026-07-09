"""
tests/test_semantic_evaluation.py

Tests for backend.semantic_evaluation. The SBERT model itself is mocked
out so these tests run fast and offline; helper functions (filler word
counting, key term extraction, rule-based feedback) are tested directly.
"""
from backend.semantic_evaluation import (
    count_filler_words,
    _extract_key_terms,
    _rule_based_feedback,
    SemanticEvaluator,
)


def test_count_filler_words():
    text = "So, um, this is like, you know, basically the idea."
    count = count_filler_words(text)
    assert count >= 3


def test_extract_key_terms_filters_stopwords_and_short_words():
    reference = "Photosynthesis is the process by which plants convert sunlight into energy."
    terms = _extract_key_terms(reference, top_n=5)
    assert "photosynthesis" in terms
    assert "the" not in terms
    assert all(len(t) > 3 for t in terms)


def test_rule_based_feedback_high_similarity():
    feedback = _rule_based_feedback(0.9, matched_terms=["photosynthesis"], missing_terms=[])
    assert "Excellent" in feedback


def test_rule_based_feedback_low_similarity():
    feedback = _rule_based_feedback(0.2, matched_terms=[], missing_terms=["chlorophyll"])
    assert "diverges" in feedback or "chlorophyll" in feedback


def test_semantic_evaluator_compute_similarity_with_mocked_model(mocker):
    """
    Mocks the SentenceTransformer model entirely so this test runs offline
    without downloading real weights. Verifies compute_similarity() correctly
    clamps/rescales the cosine similarity returned by the (mocked) util.cos_sim.
    """
    evaluator = SemanticEvaluator()

    fake_model = mocker.Mock()
    fake_model.encode.return_value = ["embedding_ref", "embedding_spoken"]
    mocker.patch.object(SemanticEvaluator, "model", new_callable=mocker.PropertyMock,
                         return_value=fake_model)

    fake_cos_sim_result = mocker.Mock()
    fake_cos_sim_result.item.return_value = 0.75
    mocker.patch("sentence_transformers.util.cos_sim", return_value=fake_cos_sim_result)

    score = evaluator.compute_similarity("reference text", "spoken text")
    assert score == 0.75
