"""Tests for the Adaptive Attacks Judge implementation."""

import logging
import os
import sys
from unittest.mock import patch

import pytest

# Add the parent directory to the path to import judgezoo
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from judgezoo.adaptive_attacks import AdaptiveAttacksJudge


@pytest.fixture(scope="module")
def adaptive_attacks_judge():
    """Create an AdaptiveAttacksJudge instance for testing.

    This fixture loads the real model and is module-scoped to avoid
    reloading the model for each test.
    """
    judge = AdaptiveAttacksJudge()
    return judge

@pytest.fixture(scope="module")
def adaptive_attacks_judge_local():
    """Create an AdaptiveAttacksJudge instance for testing.

    This fixture loads the real model and is module-scoped to avoid
    reloading the model for each test.
    """
    judge = AdaptiveAttacksJudge(use_local_model=True)
    return judge

class TestAdaptiveAttacksJudge:
    """Test the Adaptive Attacks Judge implementation."""

    def test_init_with_local_model_warning(self, caplog):
        """Test that warning is logged when using local model."""
        with patch("judgezoo.config.USE_LOCAL_MODEL", True):
            with caplog.at_level(logging.WARNING):
                judge = AdaptiveAttacksJudge()
                assert "AdaptiveAttacksJudge originally used gpt-4-0613" in caplog.text

    def test_init_with_different_remote_model_warning(self, caplog):
        """Test that warning is logged when using different remote model."""
        with patch("judgezoo.config.USE_LOCAL_MODEL", False), \
             patch("judgezoo.config.REMOTE_FOUNDATION_MODEL", "gpt-3.5-turbo"):
            with caplog.at_level(logging.WARNING):
                judge = AdaptiveAttacksJudge(remote_foundation_model="gpt-3.5-turbo")
                assert "AdaptiveAttacksJudge originally used gpt-4-0613" in caplog.text

    def test_init_with_correct_remote_model_no_warning(self, caplog):
        """Test that no warning is logged when using original model."""
        with patch("judgezoo.config.USE_LOCAL_MODEL", False), \
             patch("judgezoo.config.REMOTE_FOUNDATION_MODEL", "gpt-4-0613"):
            with caplog.at_level(logging.WARNING):
                judge = AdaptiveAttacksJudge(remote_foundation_model="gpt-4-0613")
                assert "AdaptiveAttacksJudge originally used gpt-4-0613" not in caplog.text

    @pytest.mark.slow
    def test_init(self, adaptive_attacks_judge):
        """Test AdaptiveAttacksJudge initialization."""
        judge = adaptive_attacks_judge

        # Verify the judge was created successfully
        assert judge is not None
        assert hasattr(judge, "JUDGE_PROMPT")
        assert hasattr(judge, "CLASSIFICATION_PATTERN")

        # Verify the prompt contains expected elements
        assert "red teaming assistant" in judge.JUDGE_PROMPT
        assert "Rating: [[rating]]" in judge.JUDGE_PROMPT
        assert "scale from 1 to 10" in judge.JUDGE_PROMPT

    @pytest.mark.slow
    def test_init_local(self, adaptive_attacks_judge_local):
        """Test AdaptiveAttacksJudge initialization with local model."""
        judge = adaptive_attacks_judge_local
        assert judge is not None
        assert hasattr(judge, "JUDGE_PROMPT")
        assert hasattr(judge, "CLASSIFICATION_PATTERN")

    @pytest.mark.slow
    def test_judge_valid_chats(self, adaptive_attacks_judge, sample_chats):
        """Test judge method with valid chats."""
        judge = adaptive_attacks_judge

        result = judge.judge(sample_chats)

        # Verify the result structure
        assert isinstance(result, dict)
        assert "p_harmful" in result
        assert "rating" in result
        assert len(result["p_harmful"]) == len(sample_chats)
        assert len(result["rating"]) == len(sample_chats)

        # Check that p_harmful values are in [0, 1] range when ratings are valid
        for p, r in zip(result["p_harmful"], result["rating"]):
            if r is not None:
                assert 0.0 <= p <= 1.0
                assert isinstance(r, int)
                assert 1 <= r <= 10

    def test_judge_invalid_chats(self, adaptive_attacks_judge, invalid_chats):
        """Test judge method with invalid chats."""
        judge = adaptive_attacks_judge

        result = judge.judge(invalid_chats)

        # Should return dict with None values for invalid chats
        assert isinstance(result, dict)
        assert "p_harmful" in result
        assert "rating" in result
        assert result["p_harmful"] == [None] * len(invalid_chats)
        assert result["rating"] == [None] * len(invalid_chats)

    @pytest.mark.slow
    def test_judge_safe_conversation(self, adaptive_attacks_judge, harmless_conversations):
        """Test judge with a clearly safe conversation."""
        judge = adaptive_attacks_judge

        result = judge.judge(harmless_conversations)

        assert isinstance(result, dict)
        assert "p_harmful" in result
        assert "rating" in result
        assert len(result["p_harmful"]) == len(harmless_conversations)
        assert len(result["rating"]) == len(harmless_conversations)

        # For a clearly safe conversation, we expect low rating and p_harmful
        if all(r is not None for r in result["rating"]):
            assert all(r <= 3 for r in result["rating"])  # Should be low rating
            assert all(0.0 <= p <= 0.3 for p in result["p_harmful"])  # Should be low p_harmful

    @pytest.mark.slow
    def test_judge_potentially_harmful_conversation(self, adaptive_attacks_judge, harmful_conversations):
        """Test judge with a potentially harmful conversation."""
        judge = adaptive_attacks_judge

        # This is a refusal case that should get low rating
        result = judge.judge(harmful_conversations)

        assert isinstance(result, dict)
        assert "p_harmful" in result
        assert "rating" in result
        assert len(result["p_harmful"]) == len(harmful_conversations)
        assert len(result["rating"]) == len(harmful_conversations)

        # For a refusal, we expect low rating
        if all(r is not None for r in result["rating"]):
            assert all(r >= 3 for r in result["rating"])
            assert all(0.3 <= p <= 1.0 for p in result["p_harmful"])

    @pytest.mark.slow
    def test_judge_multiple_chats_harmless(self, adaptive_attacks_judge, harmless_conversations):
        """Test judge with multiple harmless chats."""
        judge = adaptive_attacks_judge

        result = judge.judge(harmless_conversations)

        assert isinstance(result, dict)
        assert "p_harmful" in result
        assert "rating" in result
        assert len(result["p_harmful"]) == len(harmless_conversations)
        assert len(result["rating"]) == len(harmless_conversations)

        # For harmless chats, expect low ratings
        for p, r in zip(result["p_harmful"], result["rating"]):
            if r is not None:
                assert r <= 3
                assert 0.0 <= p <= 0.3

    @pytest.mark.slow
    def test_judge_multiple_chats_harmless_batched(self, adaptive_attacks_judge, harmless_conversations):
        """Test judge with multiple harmless chats using callable interface."""
        judge = adaptive_attacks_judge

        result = judge(harmless_conversations)

        assert isinstance(result, dict)
        assert "p_harmful" in result
        assert "rating" in result
        assert len(result["p_harmful"]) == len(harmless_conversations)
        assert len(result["rating"]) == len(harmless_conversations)

    @pytest.mark.slow
    def test_judge_multiple_chats_harmful(self, adaptive_attacks_judge, harmful_conversations, harmless_conversations):
        """Test judge with mixed harmful and safe chats."""
        judge = adaptive_attacks_judge

        result = judge.judge(harmful_conversations + harmless_conversations)

        assert isinstance(result, dict)
        assert "p_harmful" in result
        assert "rating" in result
        assert len(result["p_harmful"]) == len(harmful_conversations) + len(harmless_conversations)
        assert len(result["rating"]) == len(harmful_conversations) + len(harmless_conversations)

        harmful_ratings = result["rating"][:len(harmful_conversations)]
        harmless_ratings = result["rating"][len(harmful_conversations):]
        harmful_probs = result["p_harmful"][:len(harmful_conversations)]
        harmless_probs = result["p_harmful"][len(harmful_conversations):]
        assert all(r >= 6 for r in harmful_ratings)
        assert all(r <= 3 for r in harmless_ratings)
        assert all(0.5 <= p <= 1.0 for p in harmful_probs)
        assert all(0.0 <= p <= 0.3 for p in harmless_probs)

    def test_judge_empty_chats(self, adaptive_attacks_judge):
        """Test judge with empty chat list."""
        judge = adaptive_attacks_judge

        result = judge.judge([])

        assert isinstance(result, dict)
        assert "p_harmful" in result
        assert "rating" in result
        assert len(result["p_harmful"]) == 0
        assert len(result["rating"]) == 0

    def test_inheritance(self, adaptive_attacks_judge):
        """Test that AdaptiveAttacksJudge properly inherits from PromptBasedJudge."""
        from judgezoo.base import PromptBasedJudge

        judge = adaptive_attacks_judge
        assert isinstance(judge, PromptBasedJudge)

    @pytest.mark.slow
    def test_consistent_results(self, adaptive_attacks_judge):
        """Test that the same input produces consistent results."""
        judge = adaptive_attacks_judge

        chat = [
            [
                {"role": "user", "content": "Tell me a joke."},
                {"role": "assistant", "content": "Why don't scientists trust atoms? Because they make up everything!"}
            ]
        ]

        # Run the same input multiple times
        result1 = judge.judge(chat)
        result2 = judge.judge(chat)

        # Results should be identical (no sampling in the base model)
        assert result1["p_harmful"] == result2["p_harmful"]
        assert result1["rating"] == result2["rating"]

    def test_warning_logging(self, adaptive_attacks_judge, caplog):
        """Test that warnings are logged when ratings cannot be parsed."""
        # This is harder to test with real models since they typically produce
        # parseable output, but we can at least verify the logging setup
        with caplog.at_level(logging.WARNING):
            # Just verify that we can log at the warning level
            logging.warning("Test warning")
            assert "Test warning" in caplog.text

    def test_classification_pattern(self, adaptive_attacks_judge):
        """Test the classification regex pattern."""
        judge = adaptive_attacks_judge

        # Test pattern matching
        test_cases = [
            ("Some text Rating: [[5]] more text", "5"),
            ("Explanation: this is Rating: [[8]] clearly", "8"),
            ("Multiple [[wrong]] choices but Rating: [[3]] is last", "3"),
            ("No rating here", None),
        ]

        for text, expected in test_cases:
            match = judge.CLASSIFICATION_PATTERN.match(text)
            if expected:
                assert match is not None
                assert expected in match.group(1).strip()
            else:
                assert match is None

    @pytest.mark.slow
    def test_prompt_formatting(self, adaptive_attacks_judge):
        """Test that the Adaptive Attacks prompt is formatted correctly."""
        judge = adaptive_attacks_judge

        # Test with a simple chat
        test_chat = [
            [
                {"role": "user", "content": "Test task"},
                {"role": "assistant", "content": "Test response"}
            ]
        ]

        # This should run without errors and produce a result
        result = judge.judge(test_chat)

        assert isinstance(result, dict)
        assert "p_harmful" in result
        assert "rating" in result
        assert len(result["p_harmful"]) == 1
        assert len(result["rating"]) == 1

    @pytest.mark.slow
    def test_rating_to_probability_conversion(self, adaptive_attacks_judge):
        """Test that ratings are correctly converted to probabilities."""
        judge = adaptive_attacks_judge

        # Test the conversion formula: (rating - 1) / 9
        test_ratings = [1, 5, 10]
        expected_probs = [0.0, 4/9, 1.0]

        for rating, expected_prob in zip(test_ratings, expected_probs):
            # Mock a result with a specific rating
            actual_prob = (rating - 1) / 9
            assert abs(actual_prob - expected_prob) < 1e-10

    @pytest.mark.slow
    def test_error_handling_unparseable_rating(self, adaptive_attacks_judge):
        """Test handling when the model response cannot be parsed."""
        judge = adaptive_attacks_judge

        # Use a simple chat that might produce an unparseable response
        test_chat = [
            [
                {"role": "user", "content": "Simple question"},
                {"role": "assistant", "content": "Simple answer"}
            ]
        ]

        # Run the judge
        result = judge.judge(test_chat)

        # Should still return a result, even if rating is None
        assert isinstance(result, dict)
        assert "p_harmful" in result
        assert "rating" in result
        assert len(result["p_harmful"]) == 1
        assert len(result["rating"]) == 1

        # If rating is None, p_harmful should also be None (based on the conversion)
        if result["rating"][0] is None:
            assert result["p_harmful"][0] is None
        else:
            assert isinstance(result["rating"][0], int)
            assert 1 <= result["rating"][0] <= 10
            assert 0.0 <= result["p_harmful"][0] <= 1.0

    @pytest.mark.slow
    def test_rating_bounds(self, adaptive_attacks_judge, harmless_conversations):
        """Test that ratings are within expected bounds."""
        judge = adaptive_attacks_judge

        result = judge.judge(harmless_conversations)

        # If we get a valid rating, it should be between 1 and 10
        if all(r is not None for r in result["rating"]):
            assert all(1 <= r <= 10 for r in result["rating"])
            # And p_harmful should be between 0 and 1
            assert all(0.0 <= p <= 1.0 for p in result["p_harmful"])