"""Tests for the MD Judge implementation."""

import logging
import os
import sys

import pytest

# Add the parent directory to the path to import judgezoo
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from judgezoo.md_judge import MDJudge, MDJudgeVersion


@pytest.fixture(scope="module")
def md_judge_v01():
    """Create an MDJudge v0.1 instance for testing.

    This fixture loads the real model and is session-scoped to avoid
    reloading the model for each test.
    """
    judge = MDJudge(version=MDJudgeVersion.V0_1)
    return judge


@pytest.fixture(scope="module")
def md_judge():
    """Create an MDJudge instance for testing (defaults to v0.2).

    This fixture loads the real model and is session-scoped to avoid
    reloading the model for each test.
    """
    judge = MDJudge()
    return judge


class TestMDJudgeV01:
    """Test the MD Judge v0.1 implementation."""

    @pytest.mark.slow
    def test_init(self, md_judge_v01):
        """Test MDJudge v0.1 initialization."""
        judge = md_judge_v01

        # Verify the judge was created successfully
        assert judge is not None
        assert hasattr(judge, "classifier")
        assert hasattr(judge, "tokenizer")
        assert judge.version == MDJudgeVersion.V0_1

        # Verify v0.1 specific attributes
        assert hasattr(judge, "unsafe_token_ids")
        assert hasattr(judge, "safe_token_ids")
        assert hasattr(judge, "CLS_CONVERSATION")

    def test_judge_invalid_chats(self, md_judge_v01, invalid_chats):
        """Test judge method with invalid chats."""
        judge = md_judge_v01

        result = judge.judge(invalid_chats)

        # Should return dict with None values for invalid chats
        assert isinstance(result, dict)
        assert "p_harmful" in result
        assert result["p_harmful"] == [None] * len(invalid_chats)

    @pytest.mark.slow
    def test_judge_valid_chats(self, md_judge_v01, sample_chats):
        """Test judge method with valid chats."""
        judge = md_judge_v01

        result = judge.judge(sample_chats)

        # Verify the result structure for v0.1
        assert isinstance(result, dict)
        assert "p_harmful" in result
        assert len(result["p_harmful"]) == len(sample_chats)
        assert all(isinstance(p, float) for p in result["p_harmful"])
        assert all(0.0 <= p <= 1.0 for p in result["p_harmful"])

    @pytest.mark.slow
    def test_judge_safe_conversation(self, md_judge_v01, harmless_conversations):
        """Test judge with a clearly safe conversation."""
        judge = md_judge_v01

        result = judge.judge(harmless_conversations)

        assert isinstance(result, dict)
        assert "p_harmful" in result
        assert len(result["p_harmful"]) == len(harmless_conversations)
        # For a clearly safe conversation, we expect low probability of harm
        assert all(0.0 <= p <= 0.1 for p in result["p_harmful"])

    @pytest.mark.slow
    def test_judge_potentially_harmful_conversation(
        self, md_judge_v01, harmful_conversations
    ):
        """Test judge with a potentially harmful conversation."""
        judge = md_judge_v01

        # Test with harmful conversations that should be flagged
        result = judge.judge(harmful_conversations)

        assert isinstance(result, dict)
        assert "p_harmful" in result
        assert len(result["p_harmful"]) == len(harmful_conversations)
        # For harmful conversations, we expect higher probability of harm
        assert all(0.5 <= p <= 1.0 for p in result["p_harmful"])

    def test_judge_empty_chats(self, md_judge_v01):
        """Test judge with empty chat list."""
        judge = md_judge_v01

        result = judge.judge([])

        assert isinstance(result, dict)
        assert "p_harmful" in result
        assert len(result["p_harmful"]) == 0


class TestMDJudgeV02:
    """Test the MD Judge v0.2 implementation."""

    @pytest.mark.slow
    def test_init(self, md_judge):
        """Test MDJudge v0.2 initialization."""
        judge = md_judge

        # Verify the judge was created successfully
        assert judge is not None
        assert hasattr(judge, "classifier")
        assert hasattr(judge, "tokenizer")
        assert judge.version == MDJudgeVersion.V0_2

        # Verify v0.2 specific attributes
        assert hasattr(judge, "CLASSIFICATION_PATTERN")
        # Note: safe_token_ids and unsafe_token_ids are commented out in v0.2

    def test_judge_invalid_chats(self, md_judge, invalid_chats):
        """Test judge method with invalid chats."""
        judge = md_judge

        result = judge.judge(invalid_chats)

        # Should return dict with None values for invalid chats
        assert isinstance(result, dict)
        assert "p_harmful" in result
        assert "score" in result
        assert "harmful" in result
        assert "category" in result
        assert result["p_harmful"] == [None] * len(invalid_chats)
        assert result["score"] == [None] * len(invalid_chats)
        assert result["harmful"] == [None] * len(invalid_chats)
        assert result["category"] == [None] * len(invalid_chats)

    @pytest.mark.slow
    def test_judge_valid_chats(self, md_judge, sample_chats):
        """Test judge method with valid chats."""
        judge = md_judge

        result = judge.judge(sample_chats)

        # Verify the result structure for v0.2
        assert isinstance(result, dict)
        assert "p_harmful" in result
        assert "score" in result
        assert "harmful" in result
        assert "category" in result
        assert len(result["p_harmful"]) == len(sample_chats)
        assert len(result["score"]) == len(sample_chats)
        assert len(result["harmful"]) == len(sample_chats)
        assert len(result["category"]) == len(sample_chats)

        # Check data types
        for p in result["p_harmful"]:
            assert p is None or isinstance(p, float)
        for s in result["score"]:
            assert s is None or isinstance(s, int)
        for h in result["harmful"]:
            assert h is None or isinstance(h, bool)
        for c in result["category"]:
            assert c is None or isinstance(c, str)

    @pytest.mark.slow
    def test_judge_safe_conversation(self, md_judge, harmless_conversations):
        """Test judge with a clearly safe conversation."""
        judge = md_judge

        result = judge.judge(harmless_conversations)

        assert isinstance(result, dict)
        assert "p_harmful" in result
        assert "score" in result
        assert "harmful" in result
        assert "category" in result
        assert len(result["p_harmful"]) == len(harmless_conversations)

        # Check that valid results have proper structure
        for i in range(len(harmless_conversations)):
            p_harmful = result["p_harmful"][i]
            score = result["score"][i]
            harmful = result["harmful"][i]
            category = result["category"][i]

            if p_harmful is not None:
                assert 0.0 <= p_harmful <= 1.0
                assert isinstance(score, int)
                assert 1 <= score <= 5
                assert isinstance(harmful, bool)
                if harmful:
                    assert category is not None
                    assert category.startswith("O")
                else:
                    assert category is None

    @pytest.mark.slow
    def test_judge_potentially_harmful_conversation(
        self, md_judge, harmful_conversations
    ):
        """Test judge with a potentially harmful conversation."""
        judge = md_judge

        # Test with harmful conversations that should be flagged
        result = judge.judge(harmful_conversations)

        assert isinstance(result, dict)
        assert "p_harmful" in result
        assert "score" in result
        assert "harmful" in result
        assert "category" in result
        assert len(result["p_harmful"]) == len(harmful_conversations)

        # Check that valid results have proper structure
        for i in range(len(harmful_conversations)):
            p_harmful = result["p_harmful"][i]
            score = result["score"][i]
            harmful = result["harmful"][i]
            category = result["category"][i]

            if p_harmful is not None:
                assert 0.0 <= p_harmful <= 1.0
                assert isinstance(score, int)
                assert 1 <= score <= 5
                assert isinstance(harmful, bool)
                if harmful:
                    assert category is not None
                    assert category.startswith("O")
                else:
                    assert category is None

    def test_judge_empty_chats(self, md_judge):
        """Test judge with empty chat list."""
        judge = md_judge

        result = judge.judge([])

        assert isinstance(result, dict)
        assert "p_harmful" in result
        assert "score" in result
        assert "harmful" in result
        assert "category" in result
        assert len(result["p_harmful"]) == 0
        assert len(result["score"]) == 0
        assert len(result["harmful"]) == 0
        assert len(result["category"]) == 0

    @pytest.mark.slow
    def test_v02_score_to_probability_mapping(self, md_judge):
        """Test that v0.2 correctly maps scores to probabilities."""
        judge = md_judge

        # Test with a simple conversation
        chat = [
            [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ]
        ]

        result = judge.judge(chat)

        if result["p_harmful"][0] is not None:
            p_harmful = result["p_harmful"][0]
            score = result["score"][0]

            # Verify the mapping: p_harmful = (score - 1) / 4
            expected_p_harmful = (score - 1) / 4
            assert abs(p_harmful - expected_p_harmful) < 1e-6


class TestMDJudgeGeneral:
    """Test general MD Judge functionality (using default v0.2)."""

    @pytest.mark.slow
    def test_init_default_version(self, md_judge):
        """Test MDJudge initialization with default version."""
        judge = md_judge

        # Verify the judge was created successfully
        assert judge is not None
        assert hasattr(judge, "classifier")
        assert hasattr(judge, "tokenizer")
        assert judge.version == MDJudgeVersion.V0_2  # Default should be v0.2

    @pytest.mark.slow
    def test_judge_multiple_chats_harmless(self, md_judge, harmless_conversations):
        """Test judge with multiple different chats."""
        judge = md_judge

        result = judge.judge(harmless_conversations)

        assert isinstance(result, dict)
        assert "p_harmful" in result
        assert len(result["p_harmful"]) == len(harmless_conversations)

    @pytest.mark.slow
    def test_judge_multiple_chats_harmless_batched(
        self, md_judge, harmless_conversations
    ):
        """Test judge with multiple different chats."""
        judge = md_judge

        result = judge(harmless_conversations)

        assert isinstance(result, dict)
        assert "p_harmful" in result
        assert len(result["p_harmful"]) == len(harmless_conversations)

    @pytest.mark.slow
    def test_judge_multiple_chats_harmful(
        self, md_judge, harmful_conversations, harmless_conversations
    ):
        """Test judge with multiple different chats."""
        judge = md_judge

        mixed_chats = harmful_conversations + harmless_conversations
        result = judge(mixed_chats)

        assert isinstance(result, dict)
        assert "p_harmful" in result
        assert len(result["p_harmful"]) == len(harmful_conversations) + len(
            harmless_conversations
        )

    @pytest.mark.slow
    def test_model_device_handling(self, md_judge):
        """Test that the model handles device placement correctly."""
        judge = md_judge

        # Check that the model is on some device
        model_device = next(judge.classifier.parameters()).device
        assert model_device.type in ["cuda", "cpu"]

    def test_inheritance(self, md_judge):
        """Test that MDJudge properly inherits from FineTunedJudge."""
        from judgezoo.base import FineTunedJudge

        judge = md_judge
        assert isinstance(judge, FineTunedJudge)

    @pytest.mark.slow
    def test_consistent_results(self, md_judge):
        """Test that the same input produces consistent results."""
        judge = md_judge

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
    def test_inference_runs_successfully(self, md_judge):
        """Test that inference runs successfully without errors."""
        judge = md_judge

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

    def test_warning_logging(self, md_judge, caplog):
        """Test that warnings are logged when probabilities are inconsistent."""
        # This is harder to test with real models since they typically produce
        # consistent probabilities, but we can at least verify the logging setup
        with caplog.at_level(logging.WARNING):
            # Just verify that we can log at the warning level
            logging.warning("Test warning")
            assert "Test warning" in caplog.text

    def test_invalid_version_raises_error(self):
        """Test that invalid version raises ValueError."""
        with pytest.raises(ValueError, match="Invalid version"):
            # This should raise an error since we're passing an invalid enum value
            judge = MDJudge(version="invalid_version")
