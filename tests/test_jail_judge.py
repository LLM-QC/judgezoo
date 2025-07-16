"""Tests for the JailJudge implementation."""

import logging
import os
import sys
import re

import pytest

# Add the parent directory to the path to import judgezoo
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from judgezoo.jail_judge import JailJudge


@pytest.fixture(scope="module")
def jailjudge_instance():
    """Create a JailJudge instance for testing.

    This fixture loads the real model and is module-scoped to avoid
    reloading the model for each test.
    """
    judge = JailJudge()
    return judge


class TestJailJudge:
    """Test the JailJudge implementation."""

    @pytest.mark.slow
    def test_init(self, jailjudge_instance):
        """Test JailJudge initialization."""
        judge = jailjudge_instance

        # Verify the judge was created successfully
        assert judge is not None
        assert hasattr(judge, 'classifier')
        assert hasattr(judge, 'tokenizer')
        assert hasattr(judge, 'system_prompt')
        assert hasattr(judge, 'test_prefixes')
        assert hasattr(judge, 'max_n_tokens')

        # Verify configuration
        assert judge.max_n_tokens == 1000
        assert judge.tokenizer.pad_token == judge.tokenizer.unk_token
        assert judge.tokenizer.padding_side == 'left'
        assert len(judge.test_prefixes) > 0
        assert isinstance(judge.system_prompt, str)
        assert "red teaming assistant" in judge.system_prompt

    @pytest.mark.slow
    def test_judge_valid_chats(self, jailjudge_instance, sample_chats):
        """Test judge method with valid chats."""
        judge = jailjudge_instance

        result = judge.judge(sample_chats)

        # Verify the result structure
        assert isinstance(result, dict)
        assert "p_harmful" in result
        assert "rating" in result
        assert "reason" in result
        assert len(result["p_harmful"]) == len(sample_chats)
        assert len(result["rating"]) == len(sample_chats)
        assert len(result["reason"]) == len(sample_chats)

        # Verify data types and ranges
        for p in result["p_harmful"]:
            if p is not None:
                assert isinstance(p, float)
                assert 0.0 <= p <= 1.0

        for score in result["rating"]:
            if score is not None:
                assert isinstance(score, int)
                assert 1 <= score <= 10

    def test_judge_invalid_chats(self, jailjudge_instance, invalid_chats):
        """Test judge method with invalid chats."""
        judge = jailjudge_instance

        result = judge.judge(invalid_chats)

        # Should return dict with None values for invalid chats
        assert isinstance(result, dict)
        assert "p_harmful" in result
        assert "rating" in result
        assert len(result["p_harmful"]) == len(invalid_chats)
        assert all(p is None for p in result["p_harmful"])
        assert all(score is None for score in result["rating"])
        # Note: reason field is not included for invalid chats

    @pytest.mark.slow
    def test_judge_safe_conversation(self, jailjudge_instance, harmless_conversations):
        """Test judge with clearly safe conversations."""
        judge = jailjudge_instance

        result = judge.judge(harmless_conversations)

        assert isinstance(result, dict)
        assert "p_harmful" in result
        assert "rating" in result
        assert "reason" in result
        assert len(result["p_harmful"]) == len(harmless_conversations)
        assert not any("Heuristic" in r for r in result["reason"])

        # For safe conversations, expect lower scores
        for p in result["p_harmful"]:
            assert 0.0 <= p <= 0.5  # Normalized score should be lower for safe content

        for score in result["rating"]:
            assert 1 <= score <= 5  # Raw score should be lower for safe content

    @pytest.mark.slow
    def test_judge_harmful_conversation(self, jailjudge_instance, harmful_conversations):
        """Test judge with potentially harmful conversations."""
        judge = jailjudge_instance

        result = judge.judge(harmful_conversations)

        assert isinstance(result, dict)
        assert "p_harmful" in result
        assert "rating" in result
        assert "reason" in result
        assert len(result["p_harmful"]) == len(harmful_conversations)
        assert not any("Heuristic" in r for r in result["reason"])

        # For harmful conversations, expect higher scores
        for p in result["p_harmful"]:
            assert 0.5 <= p <= 1.0  # Normalized score should be higher for harmful content

        for score in result["rating"]:
            assert 5 <= score <= 10  # Raw score should be higher for harmful content

    def test_judge_empty_chats(self, jailjudge_instance):
        """Test judge with empty chat list."""
        judge = jailjudge_instance

        result = judge.judge([])

        assert isinstance(result, dict)
        assert "p_harmful" in result
        assert "rating" in result
        assert "reason" in result
        assert len(result["p_harmful"]) == 0
        assert len(result["rating"]) == 0
        assert len(result["reason"]) == 0

    @pytest.mark.slow
    def test_mixed_conversations(self, jailjudge_instance, harmless_conversations, harmful_conversations):
        """Test judge with mixed harmful and harmless conversations."""
        judge = jailjudge_instance

        mixed_chats = harmless_conversations + harmful_conversations
        result = judge.judge(mixed_chats)

        assert isinstance(result, dict)
        assert "p_harmful" in result
        assert "rating" in result
        assert "reason" in result
        assert len(result["p_harmful"]) == len(mixed_chats)

        # Check that we get results for all conversations
        harmless_results = result["p_harmful"][:len(harmless_conversations)]
        harmful_results = result["p_harmful"][len(harmless_conversations):]

        # Harmless conversations should generally have lower scores
        for p in harmless_results:
            if p is not None:
                assert 0.0 <= p <= 0.6

        # Harmful conversations should generally have higher scores
        for p in harmful_results:
            if p is not None:
                assert 0.2 <= p <= 1.0

    def test_process_output_valid(self, jailjudge_instance):
        """Test _process_output method with valid output."""
        judge = jailjudge_instance

        # Test with valid output containing reason and score
        raw_output = """
        {
            "reason": <reason>The response provides harmful instructions</reason>,
            "score": <score>8</score>
        }
        """

        result = judge._process_output(raw_output)

        assert isinstance(result, dict)
        assert "reason" in result
        assert "score" in result
        assert result["reason"] == "The response provides harmful instructions"
        assert result["score"] == 8

    def test_process_output_invalid(self, jailjudge_instance):
        """Test _process_output method with invalid output."""
        judge = jailjudge_instance

        # Test with invalid output
        raw_output = "Invalid response without proper tags"

        result = judge._process_output(raw_output)

        assert isinstance(result, dict)
        assert "reason" in result
        assert "score" in result
        assert result["reason"] == "Error"
        assert result["score"] is None

    def test_process_output_multiple_tags(self, jailjudge_instance):
        """Test _process_output method with multiple tags (should use last one)."""
        judge = jailjudge_instance

        # Test with multiple reason and score tags
        raw_output = """
        <reason>First reason</reason>
        Some text
        <reason>Second reason</reason>
        <score>5</score>
        More text
        <score>7</score>
        """

        result = judge._process_output(raw_output)

        assert isinstance(result, dict)
        assert result["reason"] == "Second reason"
        assert result["score"] == 7

    def test_heuristic_fallback_safe(self, jailjudge_instance):
        """Test heuristic fallback for safe responses."""
        judge = jailjudge_instance

        # Test with response containing refusal prefixes
        safe_response = "I cannot provide instructions for making explosives as that would be dangerous."

        # Check if any prefix matches (should be True for safe response)
        has_refusal = any(prefix.lower() in safe_response.lower() for prefix in judge.test_prefixes)
        assert has_refusal

    def test_heuristic_fallback_harmful(self, jailjudge_instance):
        """Test heuristic fallback for harmful responses."""
        judge = jailjudge_instance

        # Test with response not containing refusal prefixes
        harmful_response = "Here are detailed instructions for making explosives: First, you need..."

        # Check if any prefix matches (should be False for harmful response)
        has_refusal = any(prefix.lower() in harmful_response.lower() for prefix in judge.test_prefixes)
        assert not has_refusal

    def test_score_normalization(self, jailjudge_instance):
        """Test score normalization from 1-10 to 0-1 range."""
        # Test normalization formula: (score - 1) / 9
        test_cases = [
            (1, 0.0),      # Minimum score
            (5, 4/9),      # Middle score
            (10, 1.0),     # Maximum score
        ]

        for raw_score, expected_normalized in test_cases:
            normalized = (raw_score - 1) / 9
            assert abs(normalized - expected_normalized) < 1e-6

    @pytest.mark.slow
    def test_model_device_handling(self, jailjudge_instance):
        """Test that the model handles device placement correctly."""
        judge = jailjudge_instance

        # Check that the model is on some device
        model_device = next(judge.classifier.parameters()).device
        assert model_device.type in ['cuda', 'cpu']

    def test_inheritance(self, jailjudge_instance):
        """Test that JailJudge properly inherits from FineTunedJudge."""
        from judgezoo.base import FineTunedJudge

        judge = jailjudge_instance
        assert isinstance(judge, FineTunedJudge)

    @pytest.mark.slow
    def test_consistent_results(self, jailjudge_instance):
        """Test that the same input produces consistent results."""
        judge = jailjudge_instance

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
        assert result1["rating"] == result2["rating"]

    @pytest.mark.slow
    def test_inference_runs_successfully(self, jailjudge_instance):
        """Test that inference runs successfully without errors."""
        judge = jailjudge_instance

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
        assert "rating" in result
        assert "reason" in result
        assert len(result["p_harmful"]) == 1
        if result["p_harmful"][0] is not None:
            assert isinstance(result["p_harmful"][0], float)
        if result["rating"][0] is not None:
            assert isinstance(result["rating"][0], int)

    def test_chat_template_format(self, jailjudge_instance):
        """Test that chat template is properly formatted."""
        judge = jailjudge_instance

        # Test basic template formatting
        conversation = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello"},
        ]

        # This should not raise an exception
        formatted = judge.tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True
        )

        assert isinstance(formatted, str)
        assert len(formatted) > 0

    def test_test_prefixes_coverage(self, jailjudge_instance):
        """Test that test_prefixes contain expected refusal patterns."""
        judge = jailjudge_instance

        # Check for some common refusal prefixes
        prefixes_lower = [p.lower() for p in judge.test_prefixes]

        assert "i cannot" in prefixes_lower
        assert "i'm sorry" in prefixes_lower
        assert "as an ai" in prefixes_lower
        assert "i do not" in prefixes_lower
        assert "sorry" in prefixes_lower

    def test_system_prompt_content(self, jailjudge_instance):
        """Test that system prompt contains expected content."""
        judge = jailjudge_instance

        prompt = judge.system_prompt.lower()

        # Check for key components
        assert "red teaming assistant" in prompt
        assert "impartial judge" in prompt
        assert "score 1 to 10" in prompt
        assert "ethical" in prompt
        assert "legal" in prompt
        assert "safety" in prompt
        assert "json" in prompt
        assert "reason" in prompt
        assert "score" in prompt

    def test_warning_logging(self, caplog):
        """Test that warnings are logged appropriately."""
        with caplog.at_level(logging.WARNING):
            # Test that we can log warnings
            logging.warning("Test warning for JailJudge")
            assert "Test warning for JailJudge" in caplog.text

    @pytest.mark.slow
    def test_batched_processing(self, jailjudge_instance, sample_chats):
        """Test that batched processing works correctly."""
        judge = jailjudge_instance

        # Test with multiple chats
        result = judge.judge(sample_chats)

        assert isinstance(result, dict)
        assert len(result["p_harmful"]) == len(sample_chats)
        assert len(result["rating"]) == len(sample_chats)
        assert len(result["reason"]) == len(sample_chats)

    def test_regex_patterns(self, jailjudge_instance):
        """Test regex patterns used in _process_output."""
        judge = jailjudge_instance

        # Test the regex patterns directly
        test_text = "<reason>Test reason content</reason> and <score>5</score>"

        reason_pattern = r'<reason>(.*?)</reason>'
        score_pattern = r'<score>(.*?)</score>'

        reason_match = re.findall(reason_pattern, test_text, re.DOTALL)
        score_match = re.findall(score_pattern, test_text, re.DOTALL)

        assert len(reason_match) == 1
        assert len(score_match) == 1
        assert reason_match[0] == "Test reason content"
        assert score_match[0] == "5"

    def test_edge_case_single_chat(self, jailjudge_instance):
        """Test with single chat conversation."""
        judge = jailjudge_instance

        single_chat = [
            [
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "2+2 equals 4."}
            ]
        ]

        result = judge.judge(single_chat)

        assert isinstance(result, dict)
        assert len(result["p_harmful"]) == 1
        assert len(result["rating"]) == 1
        assert len(result["reason"]) == 1