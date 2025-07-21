"""Tests for the Jailbreak Bench Judge implementation."""

import logging
import os
import sys
from unittest.mock import patch

import pytest

# Add the parent directory to the path to import judgezoo
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from judgezoo.jailbreakbench import JailbreakBenchJudge


@pytest.fixture(scope="module")
def jailbreak_bench_judge():
    """Create a JailbreakBenchJudge instance for testing.

    This fixture loads the real model and is module-scoped to avoid
    reloading the model for each test.
    """
    judge = JailbreakBenchJudge()
    return judge


@pytest.fixture(scope="module")
def jailbreak_bench_judge_local():
    """Create an JailbreakBenchJudge instance for testing.

    This fixture loads the real model and is module-scoped to avoid
    reloading the model for each test.
    """
    judge = JailbreakBenchJudge(use_local_model=True)
    return judge


class TestJailbreakBenchJudge:
    """Test the Jailbreak Bench Judge implementation."""

    def test_init_with_local_model_warning(self, caplog):
        """Test that warning is logged when using local model."""
        with patch("judgezoo.config.USE_LOCAL_MODEL", True):
            with caplog.at_level(logging.WARNING):
                judge = JailbreakBenchJudge()
                assert (
                    "JailbreakBenchJudge originally used Llama-3-70b-chat-hf"
                    in caplog.text
                )
                assert "None" not in caplog.text
                assert judge.use_local_model is True

    def test_init_with_different_remote_model_warning(self, caplog):
        """Test that warning is logged when using different remote model."""
        with patch("judgezoo.config.USE_LOCAL_MODEL", False), patch(
            "judgezoo.config.REMOTE_FOUNDATION_MODEL", "gpt-3.5-turbo"
        ):
            with caplog.at_level(logging.WARNING):
                judge = JailbreakBenchJudge(remote_foundation_model="gpt-3.5-turbo")
                assert (
                    "JailbreakBenchJudge originally used Llama-3-70b-chat-hf"
                    in caplog.text
                )
                assert "None" not in caplog.text
                assert judge.use_local_model is False

    @pytest.mark.slow
    def test_init(self, jailbreak_bench_judge):
        """Test JailbreakBenchJudge initialization."""
        judge = jailbreak_bench_judge

        # Verify the judge was created successfully
        assert judge is not None
        assert hasattr(judge, "JUDGE_PROMPT")

        # Verify the prompt contains expected elements
        assert "RESPONSE sufficiently addresses the REQUEST" in judge.JUDGE_PROMPT
        assert "either 'safe' or 'unsafe' and nothing else" in judge.JUDGE_PROMPT

    @pytest.mark.slow
    def test_init_local(self, jailbreak_bench_judge_local):
        """Test JailbreakBenchJudge initialization with local model."""
        judge = jailbreak_bench_judge_local
        assert judge is not None
        assert hasattr(judge, "JUDGE_PROMPT")

    @pytest.mark.slow
    def test_judge_valid_chats(self, jailbreak_bench_judge, sample_chats):
        """Test judge method with valid chats."""
        judge = jailbreak_bench_judge

        result = judge.judge(sample_chats)

        assert isinstance(result, dict)
        assert "p_harmful" in result
        assert len(result["p_harmful"]) == len(sample_chats)
        for p in result["p_harmful"]:
            assert 0.0 == p or 1.0 == p or p is None

    def test_judge_invalid_chats(self, jailbreak_bench_judge, invalid_chats):
        """Test judge method with invalid chats."""
        judge = jailbreak_bench_judge

        result = judge.judge(invalid_chats)

        # Should return dict with None values for invalid chats
        assert isinstance(result, dict)
        assert "p_harmful" in result
        assert result["p_harmful"] == [None] * len(invalid_chats)

    @pytest.mark.slow
    def test_judge_safe_conversation(
        self, jailbreak_bench_judge, harmless_conversations
    ):
        """Test judge with a clearly safe conversation."""
        judge = jailbreak_bench_judge

        result = judge.judge(harmless_conversations)

        assert isinstance(result, dict)
        assert "p_harmful" in result
        assert len(result["p_harmful"]) == len(harmless_conversations)

        assert all(p == 0.0 for p in result["p_harmful"])

    @pytest.mark.slow
    def test_judge_potentially_harmful_conversation(
        self, jailbreak_bench_judge, harmful_conversations
    ):
        """Test judge with a potentially harmful conversation."""
        judge = jailbreak_bench_judge

        # This is a refusal case that should get low rating
        result = judge.judge(harmful_conversations)

        assert isinstance(result, dict)
        assert "p_harmful" in result
        assert len(result["p_harmful"]) == len(harmful_conversations)

        assert all(p == 1.0 or p == 0.0 for p in result["p_harmful"])

    @pytest.mark.slow
    def test_judge_multiple_chats_harmless(
        self, jailbreak_bench_judge, harmless_conversations
    ):
        """Test judge with multiple harmless chats."""
        judge = jailbreak_bench_judge

        result = judge.judge(harmless_conversations)

        assert isinstance(result, dict)
        assert "p_harmful" in result
        assert len(result["p_harmful"]) == len(harmless_conversations)

        assert all(p == 0.0 for p in result["p_harmful"])

    @pytest.mark.slow
    def test_judge_multiple_chats_harmless_batched(
        self, jailbreak_bench_judge, harmless_conversations
    ):
        """Test judge with multiple harmless chats using callable interface."""
        judge = jailbreak_bench_judge

        result = judge(harmless_conversations)

        assert isinstance(result, dict)
        assert "p_harmful" in result
        assert len(result["p_harmful"]) == len(harmless_conversations)
        assert all(p == 0.0 for p in result["p_harmful"])

    @pytest.mark.slow
    def test_judge_multiple_chats_harmful(
        self, jailbreak_bench_judge, harmful_conversations, harmless_conversations
    ):
        """Test judge with mixed harmful and safe chats."""
        judge = jailbreak_bench_judge

        result = judge.judge(harmful_conversations + harmless_conversations)

        assert isinstance(result, dict)
        assert "p_harmful" in result
        assert len(result["p_harmful"]) == len(harmful_conversations) + len(
            harmless_conversations
        )

        harmful_probs = result["p_harmful"][: len(harmful_conversations)]
        harmless_probs = result["p_harmful"][len(harmful_conversations) :]
        assert all(p == 1 for p in harmful_probs)
        assert all(p == 0 for p in harmless_probs)

    def test_judge_empty_chats(self, jailbreak_bench_judge):
        """Test judge with empty chat list."""
        judge = jailbreak_bench_judge

        result = judge.judge([])

        assert isinstance(result, dict)
        assert "p_harmful" in result
        assert len(result["p_harmful"]) == 0

    def test_inheritance(self, jailbreak_bench_judge):
        """Test that JailbreakBenchJudge properly inherits from PromptBasedJudge."""
        from judgezoo.base import PromptBasedJudge

        judge = jailbreak_bench_judge
        assert isinstance(judge, PromptBasedJudge)

    @pytest.mark.slow
    def test_consistent_results(self, jailbreak_bench_judge):
        """Test that the same input produces consistent results."""
        judge = jailbreak_bench_judge

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

        # Results should be identical (no sampling in the base model)
        assert result1["p_harmful"] == result2["p_harmful"]

    def test_warning_logging(self, jailbreak_bench_judge, caplog):
        """Test that warnings are logged when ratings cannot be parsed."""
        # This is harder to test with real models since they typically produce
        # parseable output, but we can at least verify the logging setup
        with caplog.at_level(logging.WARNING):
            # Just verify that we can log at the warning level
            logging.warning("Test warning")
            assert "Test warning" in caplog.text

    @pytest.mark.slow
    def test_prompt_formatting(self, jailbreak_bench_judge):
        """Test that the Jailbreak Bench prompt is formatted correctly."""
        judge = jailbreak_bench_judge

        # Test with a simple chat
        test_chat = [
            [
                {"role": "user", "content": "Test task"},
                {"role": "assistant", "content": "Test response"},
            ]
        ]

        # This should run without errors and produce a result
        result = judge.judge(test_chat)

        assert isinstance(result, dict)
        assert "p_harmful" in result
        assert len(result["p_harmful"]) == 1
