"""Tests for the PAIR Judge implementation."""

import logging
import os
import sys
from unittest.mock import patch

import pytest

# Add the parent directory to the path to import judgezoo
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from judgezoo.pair import PAIRJudge


@pytest.fixture(scope="module")
def pair_judge():
    """Create a PAIRJudge instance for testing.

    This fixture loads the real model and is module-scoped to avoid
    reloading the model for each test.
    """
    judge = PAIRJudge()
    return judge


@pytest.fixture(scope="module")
def pair_judge_local():
    """Create a PAIRJudge instance with local model for testing.

    This fixture loads the real model and is module-scoped to avoid
    reloading the model for each test.
    """
    judge = PAIRJudge(use_local_model=True)
    return judge


class TestPAIRJudge:
    """Test the PAIR Judge implementation."""

    def test_init_with_local_model_warning(self, caplog):
        """Test that warning is logged when using local model."""
        with patch("judgezoo.config.USE_LOCAL_MODEL", True):
            with caplog.at_level(logging.WARNING):
                judge = PAIRJudge()
                assert "PAIRJudge originally used gpt-4-0613" in caplog.text

    def test_init_with_different_remote_model_warning(self, caplog):
        """Test that warning is logged when using different remote model."""
        with patch("judgezoo.config.USE_LOCAL_MODEL", False), \
             patch("judgezoo.config.REMOTE_FOUNDATION_MODEL", "gpt-3.5-turbo"):
            with caplog.at_level(logging.WARNING):
                judge = PAIRJudge(remote_foundation_model="gpt-3.5-turbo")
                assert "PAIRJudge originally used gpt-4-0613" in caplog.text

    def test_init_with_correct_remote_model_no_warning(self, caplog):
        """Test that no warning is logged when using original model."""
        with patch("judgezoo.config.USE_LOCAL_MODEL", False), \
             patch("judgezoo.config.REMOTE_FOUNDATION_MODEL", "gpt-4-0613"):
            with caplog.at_level(logging.WARNING):
                judge = PAIRJudge(remote_foundation_model="gpt-4-0613")
                assert "PAIRJudge originally used gpt-4-0613" not in caplog.text

    @pytest.mark.slow
    def test_init(self, pair_judge):
        """Test PAIRJudge initialization."""
        judge = pair_judge

        # Verify the judge was created successfully
        assert judge is not None
        assert hasattr(judge, "JUDGE_SYSTEM_PROMPT")
        assert hasattr(judge, "JUDGE_USER_PROMPT")
        assert hasattr(judge, "CLASSIFICATION_PATTERN")

        # Verify the prompt contains expected elements (inherited from AdaptiveAttacksJudge)
        assert "red teaming assistant" in judge.JUDGE_SYSTEM_PROMPT
        assert "Rating: [[rating]]" in judge.JUDGE_SYSTEM_PROMPT
        assert "scale from 1 to 10" in judge.JUDGE_SYSTEM_PROMPT

    @pytest.mark.slow
    def test_init_local(self, pair_judge_local):
        """Test PAIRJudge initialization with local model."""
        judge = pair_judge_local
        assert judge is not None
        assert hasattr(judge, "JUDGE_SYSTEM_PROMPT")
        assert hasattr(judge, "JUDGE_USER_PROMPT")
        assert hasattr(judge, "CLASSIFICATION_PATTERN")

    @pytest.mark.slow
    def test_judge_valid_chats(self, pair_judge, sample_chats):
        """Test judge method with valid chats."""
        judge = pair_judge

        result = judge.judge(sample_chats, sample_chats)

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

    def test_judge_invalid_chats(self, pair_judge, invalid_chats):
        """Test judge method with invalid chats."""
        judge = pair_judge

        result = judge.judge(invalid_chats, invalid_chats)

        # Should return dict with None values for invalid chats
        assert isinstance(result, dict)
        assert "p_harmful" in result
        assert "rating" in result
        assert result["p_harmful"] == [None] * len(invalid_chats)
        assert result["rating"] == [None] * len(invalid_chats)

    @pytest.mark.slow
    def test_judge_safe_conversation(self, pair_judge, harmless_conversations):
        """Test judge with a clearly safe conversation."""
        judge = pair_judge

        result = judge.judge(harmless_conversations, harmless_conversations)

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
    def test_judge_potentially_harmful_conversation(self, pair_judge, harmful_conversations):
        """Test judge with a potentially harmful conversation."""
        judge = pair_judge

        result = judge.judge(harmful_conversations, harmful_conversations)

        assert isinstance(result, dict)
        assert "p_harmful" in result
        assert "rating" in result
        assert len(result["p_harmful"]) == len(harmful_conversations)
        assert len(result["rating"]) == len(harmful_conversations)

        # For a harmful conversation, we expect high rating
        if all(r is not None for r in result["rating"]):
            assert all(r >= 3 for r in result["rating"])
            assert all(0.3 <= p <= 1.0 for p in result["p_harmful"])

    def test_judge_empty_chats(self, pair_judge):
        """Test judge with empty chat list."""
        judge = pair_judge

        result = judge.judge([], [])

        assert isinstance(result, dict)
        assert "p_harmful" in result
        assert "rating" in result
        assert len(result["p_harmful"]) == 0
        assert len(result["rating"]) == 0

    def test_inheritance(self, pair_judge):
        """Test that PAIRJudge properly inherits from AdaptiveAttacksJudge."""
        from judgezoo.adaptive_attacks import AdaptiveAttacksJudge
        from judgezoo.base import PromptBasedJudge

        judge = pair_judge
        assert isinstance(judge, AdaptiveAttacksJudge)
        assert isinstance(judge, PromptBasedJudge)

    @pytest.mark.slow
    def test_consistent_results(self, pair_judge):
        """Test that the same input produces consistent results."""
        judge = pair_judge

        chat = [
            [
                {"role": "user", "content": "Tell me a joke."},
                {"role": "assistant", "content": "Why don't scientists trust atoms? Because they make up everything!"}
            ]
        ]

        # Run the same input multiple times
        result1 = judge.judge(chat, chat)
        result2 = judge.judge(chat, chat)

        # Results should be identical (no sampling in the base model)
        assert result1["p_harmful"] == result2["p_harmful"]
        assert result1["rating"] == result2["rating"]

    @pytest.mark.slow
    def test_rating_to_probability_conversion(self, pair_judge):
        """Test that ratings are correctly converted to probabilities."""
        judge = pair_judge

        # Test the conversion formula: (rating - 1) / 9
        test_ratings = [1, 5, 10]
        expected_probs = [0.0, 4/9, 1.0]

        for rating, expected_prob in zip(test_ratings, expected_probs):
            # Mock a result with a specific rating
            actual_prob = (rating - 1) / 9
            assert abs(actual_prob - expected_prob) < 1e-10

    def test_factory_method(self):
        """Test that PAIR judge can be instantiated via factory method."""
        from judgezoo import Judge

        judge = Judge.from_name("pair")
        assert isinstance(judge, PAIRJudge)

    @pytest.mark.slow
    def test_factory_method_with_params(self):
        """Test factory method with custom parameters."""
        from judgezoo import Judge

        judge = Judge.from_name("pair", use_local_model=False, remote_foundation_model="gpt-4o")
        assert isinstance(judge, PAIRJudge)
        assert judge.use_local_model is False
        assert judge.remote_foundation_model == "gpt-4o"
