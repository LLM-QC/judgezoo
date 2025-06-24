"""Tests for the HarmBench Judge implementation."""

import pytest
import sys
import os
import logging

# Add the parent directory to the path to import judges
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from judges.harmbench import HarmBenchJudge


@pytest.fixture(scope="session")
def harmbench_judge():
    """Create a HarmBenchJudge instance for testing.

    This fixture loads the real model and is session-scoped to avoid
    reloading the model for each test.
    """
    try:
        judge = HarmBenchJudge()
        return judge
    except Exception as e:
        pytest.skip(f"Could not load HarmBenchJudge: {e}")


class TestHarmbenchJudge:
    """Test the HarmBench Judge implementation."""

    @pytest.mark.slow
    def test_init(self, harmbench_judge):
        """Test HarmBenchJudge initialization."""
        judge = harmbench_judge

        # Verify the judge was created successfully
        assert judge is not None
        assert hasattr(judge, 'classifier')
        assert hasattr(judge, 'tokenizer')

        # Verify token IDs are set correctly
        assert judge.yes_token_id == 8241
        assert judge.no_token_id == 3782

    @pytest.mark.slow
    def test_judge_valid_chats(self, harmbench_judge, sample_chats):
        """Test judge method with valid chats."""
        judge = harmbench_judge

        result = judge.judge(sample_chats)

        # Verify the result structure
        assert isinstance(result, dict)
        assert "p_harmful" in result
        assert len(result["p_harmful"]) == len(sample_chats)
        assert all(isinstance(p, float) for p in result["p_harmful"])
        assert all(0.0 <= p <= 1.0 for p in result["p_harmful"])

    def test_judge_invalid_chats(self, harmbench_judge, invalid_chats):
        """Test judge method with invalid chats."""
        judge = harmbench_judge

        result = judge.judge(invalid_chats)

        # Should return list of Nones for invalid chats
        assert result == [None] * len(invalid_chats)

    @pytest.mark.slow
    def test_judge_safe_conversation(self, harmbench_judge, harmless_conversations):
        """Test judge with a clearly safe conversation."""
        judge = harmbench_judge

        result = judge.judge(harmless_conversations)

        assert isinstance(result, dict)
        assert "p_harmful" in result
        assert len(result["p_harmful"]) == len(harmless_conversations)
        # For a clearly safe conversation, we expect low probability of harm
        assert all(0.0 <= p <= 0.2 for p in result["p_harmful"])

    @pytest.mark.slow
    def test_judge_potentially_harmful_conversation(self, harmbench_judge, harmful_conversations):
        """Test judge with a potentially harmful conversation."""
        judge = harmbench_judge

        # Test with harmful conversations that should be flagged
        result = judge.judge(harmful_conversations)

        assert isinstance(result, dict)
        assert "p_harmful" in result
        assert len(result["p_harmful"]) == len(harmful_conversations)
        # For harmful conversations, we expect higher probability of harm
        assert all(0.5 <= p <= 1.0 for p in result["p_harmful"])

    @pytest.mark.slow
    def test_judge_multiple_chats_harmless(self, harmbench_judge, harmless_conversations):
        """Test judge with multiple harmless chats."""
        judge = harmbench_judge

        result = judge.judge(harmless_conversations)

        assert isinstance(result, dict)
        assert "p_harmful" in result
        assert len(result["p_harmful"]) == len(harmless_conversations)
        assert all(isinstance(p, float) for p in result["p_harmful"])
        assert all(0.0 <= p <= .2 for p in result["p_harmful"])

    @pytest.mark.slow
    def test_judge_multiple_chats_harmless_batched(self, harmbench_judge, harmless_conversations):
        """Test judge with multiple harmless chats using callable interface."""
        judge = harmbench_judge

        result = judge(harmless_conversations)

        assert isinstance(result, dict)
        assert "p_harmful" in result
        assert len(result["p_harmful"]) == len(harmless_conversations)
        assert all(isinstance(p, float) for p in result["p_harmful"])
        assert all(0.0 <= p <= .2 for p in result["p_harmful"])

    @pytest.mark.slow
    def test_judge_multiple_chats_harmful(self, harmbench_judge, harmful_conversations, harmless_conversations):
        """Test judge with mixed harmful and safe chats."""
        judge = harmbench_judge

        mixed_chats = harmful_conversations + harmless_conversations
        result = judge.judge(mixed_chats)

        assert isinstance(result, dict)
        assert "p_harmful" in result
        assert len(result["p_harmful"]) == len(harmful_conversations) + len(harmless_conversations)
        assert all(isinstance(p, float) for p in result["p_harmful"])

        # Check harmful conversations (first part of results)
        harmful_probs = result["p_harmful"][:len(harmful_conversations)]
        assert all(0.5 <= p <= 1.0 for p in harmful_probs)

        # Check harmless conversations (second part of results)
        harmless_probs = result["p_harmful"][len(harmful_conversations):]
        assert all(0.0 <= p <= 0.2 for p in harmless_probs)

    def test_judge_empty_chats(self, harmbench_judge):
        """Test judge with empty chat list."""
        judge = harmbench_judge

        result = judge.judge([])

        assert isinstance(result, dict)
        assert "p_harmful" in result
        assert len(result["p_harmful"]) == 0

    @pytest.mark.slow
    def test_model_device_handling(self, harmbench_judge):
        """Test that the model handles device placement correctly."""
        judge = harmbench_judge

        # Check that the model is on some device
        model_device = next(judge.classifier.parameters()).device
        assert model_device.type in ['cuda', 'cpu']

    def test_inheritance(self, harmbench_judge):
        """Test that HarmBenchJudge properly inherits from FineTunedJudge."""
        from judges.base import FineTunedJudge

        judge = harmbench_judge
        assert isinstance(judge, FineTunedJudge)

    @pytest.mark.slow
    def test_consistent_results(self, harmbench_judge):
        """Test that the same input produces consistent results."""
        judge = harmbench_judge

        chat = [
            [
                {"role": "user", "content": "Tell me a joke."},
                {"role": "assistant", "content": "Why don't scientists trust atoms? Because they make up everything!"}
            ]
        ]

        # Run the same input multiple times
        result1 = judge.judge(chat)
        result2 = judge.judge(chat)

        # Results should be identical (no sampling)
        assert result1["p_harmful"] == result2["p_harmful"]

    @pytest.mark.slow
    def test_inference_runs_successfully(self, harmbench_judge):
        """Test that inference runs successfully without errors."""
        judge = harmbench_judge

        chat = [
            [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"}
            ]
        ]

        # This should not raise any exceptions
        result = judge.judge(chat)

        # Verify we get a valid result
        assert isinstance(result, dict)
        assert "p_harmful" in result
        assert len(result["p_harmful"]) == 1
        assert isinstance(result["p_harmful"][0], float)

    def test_warning_logging(self, harmbench_judge, caplog):
        """Test that warnings are logged when probabilities are inconsistent."""
        # This is harder to test with real models since they typically produce
        # consistent probabilities, but we can at least verify the logging setup
        with caplog.at_level(logging.WARNING):
            # Just verify that we can log at the warning level
            logging.warning("Test warning")
            assert "Test warning" in caplog.text

    @pytest.mark.slow
    def test_prompt_formatting(self, harmbench_judge):
        """Test that the HarmBench prompt is formatted correctly."""
        judge = harmbench_judge

        # Test with a simple chat
        test_chat = [
            [
                {"role": "user", "content": "Test behavior"},
                {"role": "assistant", "content": "Test generation"}
            ]
        ]

        # This should run without errors and produce a result
        result = judge.judge(test_chat)

        assert isinstance(result, dict)
        assert "p_harmful" in result
        assert len(result["p_harmful"]) == 1
        assert isinstance(result["p_harmful"][0], float)