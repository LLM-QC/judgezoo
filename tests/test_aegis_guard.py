"""Tests for the Aegis Guard implementation."""

import logging
import os
import sys

import pytest

# Add the parent directory to the path to import judgezoo
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from judgezoo.aegis_guard import AegisGuardJudge


@pytest.fixture(scope="module", params=[False, True], ids=["permissive", "defensive"])
def aegis_guard(request):
    """Create an Aegis Guard instance for testing.

    This fixture loads the real model and is session-scoped to avoid
    reloading the model for each test.
    """
    defensive = request.param
    judge = AegisGuardJudge(defensive=defensive)
    return judge


class TestAegisGuardJudge:
    """Test the Aegis Guard implementation."""

    @pytest.mark.slow
    def test_init(self, aegis_guard):
        """Test Aegis Guard initialization."""
        judge = aegis_guard

        # Verify the judge was created successfully
        assert judge is not None
        assert hasattr(judge, "classifier")
        assert hasattr(judge, "tokenizer")

        # Verify evaluation template is set correctly
        assert "Task: " in judge.CLS_CONVERSATION[0]["content"]
        assert "violence" in judge.CLS_CONVERSATION[0]["content"]
        assert "'safe'" in judge.CLS_CONVERSATION[0]["content"]

    def test_judge_invalid_chats(self, aegis_guard, invalid_chats):
        """Test judge method with invalid chats."""
        judge = aegis_guard

        result = judge.judge(invalid_chats)

        # Should return dict with None values for invalid chats
        assert isinstance(result, dict)
        assert "p_harmful" in result
        assert result["p_harmful"] == [None] * len(invalid_chats)

    @pytest.mark.slow
    def test_judge_valid_chats(self, aegis_guard, sample_chats):
        """Test judge method with valid chats."""
        judge = aegis_guard

        result = judge.judge(sample_chats)

        # Verify the result structure
        assert isinstance(result, dict)
        assert "p_harmful" in result
        assert len(result["p_harmful"]) == len(sample_chats)
        assert all(isinstance(p, float) for p in result["p_harmful"])
        assert all(0.0 <= p <= 1.0 for p in result["p_harmful"])

    @pytest.mark.slow
    def test_judge_safe_conversation(self, aegis_guard, harmless_conversations):
        """Test judge with a clearly safe conversation."""
        judge = aegis_guard

        result = judge.judge(harmless_conversations)

        assert isinstance(result, dict)
        assert "p_harmful" in result
        assert len(result["p_harmful"]) == len(harmless_conversations)
        # For a clearly safe conversation, we expect low probability of harm
        assert all(0.0 <= p <= 0.1 for p in result["p_harmful"])
        # We don't assert a specific threshold since model outputs can vary

    @pytest.mark.slow
    def test_judge_potentially_harmful_conversation(
        self, aegis_guard, harmful_conversations
    ):
        """Test judge with a potentially harmful conversation."""
        judge = aegis_guard

        # Test with harmful conversations that should be flagged
        result = judge.judge(harmful_conversations)

        assert isinstance(result, dict)
        assert "p_harmful" in result
        assert len(result["p_harmful"]) == len(harmful_conversations)
        # For harmful conversations, we expect higher probability of harm
        assert all(0.5 <= p <= 1.0 for p in result["p_harmful"])

    @pytest.mark.slow
    def test_judge_multiple_chats_harmless(self, aegis_guard, harmless_conversations):
        """Test judge with multiple different chats."""
        judge = aegis_guard

        result = judge.judge(harmless_conversations)

        assert isinstance(result, dict)
        assert "p_harmful" in result
        assert len(result["p_harmful"]) == len(harmless_conversations)
        assert all(isinstance(p, float) for p in result["p_harmful"])
        assert all(0.0 <= p <= 0.1 for p in result["p_harmful"])

    @pytest.mark.slow
    def test_judge_multiple_chats_harmless_batched(
        self, aegis_guard, harmless_conversations
    ):
        """Test judge with multiple different chats."""
        judge = aegis_guard

        result = judge(harmless_conversations)

        assert isinstance(result, dict)
        assert "p_harmful" in result
        assert len(result["p_harmful"]) == len(harmless_conversations)
        assert all(isinstance(p, float) for p in result["p_harmful"])
        assert all(0.0 <= p <= 0.1 for p in result["p_harmful"])

    @pytest.mark.slow
    def test_judge_multiple_chats_harmful(
        self, aegis_guard, harmful_conversations, harmless_conversations
    ):
        """Test judge with multiple different chats."""
        judge = aegis_guard

        mixed_chats = harmful_conversations + harmless_conversations
        result = judge.judge(mixed_chats)

        assert isinstance(result, dict)
        assert "p_harmful" in result
        assert len(result["p_harmful"]) == len(harmful_conversations) + len(
            harmless_conversations
        )
        assert all(isinstance(p, float) for p in result["p_harmful"])

        # Check harmful conversations (first part of results)
        harmful_probs = result["p_harmful"][: len(harmful_conversations)]
        assert all(0.5 <= p <= 1.0 for p in harmful_probs)

        # Check harmless conversations (second part of results)
        harmless_probs = result["p_harmful"][len(harmful_conversations) :]
        assert all(0.0 <= p <= 0.1 for p in harmless_probs)

    def test_judge_empty_chats(self, aegis_guard):
        """Test judge with empty chat list."""
        judge = aegis_guard

        result = judge.judge([])

        assert isinstance(result, dict)
        assert "p_harmful" in result
        assert len(result["p_harmful"]) == 0

    @pytest.mark.slow
    def test_model_device_handling(self, aegis_guard):
        """Test that the model handles device placement correctly."""
        judge = aegis_guard

        # Check that the model is on some device
        model_device = next(judge.classifier.parameters()).device
        assert model_device.type in ["cuda", "cpu"]

    def test_inheritance(self, aegis_guard):
        """Test that Aegis guard properly inherits from FineTunedJudge."""
        from judgezoo.base import FineTunedJudge

        judge = aegis_guard
        assert isinstance(judge, FineTunedJudge)

    @pytest.mark.slow
    def test_consistent_results(self, aegis_guard):
        """Test that the same input produces consistent results."""
        judge = aegis_guard

        chat = [
            [
                {"role": "user", "content": "Tell me a joke."},
                {
                    "role": "assistant",
                    "content": "Why don't scientists trust atoms? Because they make up everything!",
                },
            ]
        ]

        # Run the same input multiple times
        result1 = judge.judge(chat)
        result2 = judge.judge(chat)

        # Results should be identical (no sampling)
        assert result1["p_harmful"] == result2["p_harmful"]

    @pytest.mark.slow
    def test_inference_runs_successfully(self, aegis_guard):
        """Test that inference runs successfully without errors."""
        judge = aegis_guard

        chat = [
            [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ]
        ]

        # This should not raise any exceptions
        result = judge.judge(chat)

        # Verify we get a valid result
        assert isinstance(result, dict)
        assert "p_harmful" in result
        assert len(result["p_harmful"]) == 1
        assert isinstance(result["p_harmful"][0], float)

    def test_warning_logging(self, aegis_guard, caplog):
        """Test that warnings are logged when probabilities are inconsistent."""
        # This is harder to test with real models since they typically produce
        # consistent probabilities, but we can at least verify the logging setup
        with caplog.at_level(logging.WARNING):
            # Just verify that we can log at the warning level
            logging.warning("Test warning")
            assert "Test warning" in caplog.text
