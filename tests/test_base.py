"""Tests for the base Judge classes and functionality."""

import os
import sys
from unittest.mock import Mock, patch

import pytest
import torch

# Add the parent directory to the path to import judgezoo
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from judgezoo.base import FineTunedJudge, Judge, PromptBasedJudge


class TestJudge:
    """Test the abstract Judge base class."""

    def test_from_name_harmbench(self):
        """Test creating Harmbench judge from name."""
        with patch('judgezoo.harmbench.HarmBenchJudge') as mock_harmbench:
            judge = Judge.from_name("harmbench")
            mock_harmbench.assert_called_once()

    def test_from_name_llama_guard_3(self):
        """Test creating Llama Guard 3 judge from name."""
        with patch('judgezoo.llama_guard_3.LlamaGuard3Judge') as mock_lg3:
            judge = Judge.from_name("llama_guard_3")
            mock_lg3.assert_called_once()

    def test_from_name_strong_reject(self):
        """Test creating Strong Reject judge from name."""
        with patch('judgezoo.strong_reject.StrongRejectJudge') as mock_sr:
            judge = Judge.from_name("strong_reject")
            mock_sr.assert_called_once()

    def test_from_name_xstest(self):
        """Test creating XSTest judge from name."""
        with patch('judgezoo.xstest.XSTestJudge') as mock_xs:
            judge = Judge.from_name("xstest")
            mock_xs.assert_called_once()

    def test_from_name_unknown(self):
        """Test error when creating judge with unknown name."""
        with pytest.raises(ValueError, match="Unknown judge unknown_judge"):
            Judge.from_name("unknown_judge")

    def test_validate_chats_valid(self, sample_chats):
        """Test chat validation with valid chats."""
        assert Judge.validate_chats(sample_chats) is True

    def test_validate_chats_too_short(self):
        """Test chat validation with too short chats."""
        invalid_chats = [[{"role": "user", "content": "Hello"}]]
        assert Judge.validate_chats(invalid_chats) is False

    def test_validate_chats_wrong_first_role(self, sample_chats):
        """Test chat validation with wrong first role."""
        invalid_chats = [
            [
                {"role": "assistant", "content": "Hello"},
                {"role": "user", "content": "Hi"}
            ]
        ]
        assert Judge.validate_chats(invalid_chats) is False

    def test_validate_chats_wrong_last_role(self, sample_chats):
        """Test chat validation with wrong last role."""
        invalid_chats = [
            [
                {"role": "user", "content": "Hello"},
                {"role": "user", "content": "Another message"}
            ]
        ]
        assert Judge.validate_chats(invalid_chats) is False

    def test_validate_chats_custom_roles(self, sample_chats):
        """Test chat validation with custom expected roles."""
        custom_chats = [
            [
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "Hello"}
            ]
        ]
        assert Judge.validate_chats(custom_chats, ("system", "user")) is True
        assert Judge.validate_chats(custom_chats, ("user", "assistant")) is False

    def test_tokenize_sequences_normal(self, mock_tokenizer):
        """Test tokenization with normal input."""
        inputs = ["Hello world", "How are you?"]

        mock_tokenizer.model_max_length = 4096
        mock_encoded = Mock()
        mock_encoded.input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])

        # Set up the mock to return the expected object when called as a function
        def mock_call(*args, **kwargs):
            return mock_encoded
        mock_tokenizer.side_effect = mock_call

        result = Judge.tokenize_sequences(mock_tokenizer, inputs)

        mock_tokenizer.assert_called_once_with(
            text=inputs, return_tensors="pt", padding=True, truncation=True
        )
        assert result is mock_encoded

    def test_tokenize_sequences_too_long(self, mock_tokenizer, caplog):
        """Test tokenization with input that's too long."""
        inputs = ["Very long input that exceeds model limit"]

        mock_tokenizer.model_max_length = 10
        mock_encoded = Mock()
        mock_encoded.input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]])
        mock_tokenizer.return_value = mock_encoded

        # Mock the individual tokenization check
        mock_individual = Mock()
        mock_individual.input_ids = [1] * 15  # Longer than max length
        mock_tokenizer.side_effect = [mock_encoded, mock_individual]

        result = Judge.tokenize_sequences(mock_tokenizer, inputs)

        assert " is longer than the specified maximum sequence length" in caplog.text
        assert result == mock_encoded

    @patch('judgezoo.base.with_max_batchsize')
    def test_call_method(self, mock_with_max_batchsize, sample_chats):
        """Test the __call__ method delegates to with_max_batchsize."""
        # Create a concrete Judge subclass for testing
        class ConcreteJudge(Judge):
            def judge(self, chats):
                return {"test": [0.5] * len(chats)}

        judge = ConcreteJudge()
        expected_result = {"test": [0.5, 0.5]}
        mock_with_max_batchsize.return_value = expected_result

        result = judge(sample_chats, verbose=True)

        mock_with_max_batchsize.assert_called_once_with(
            judge.judge, sample_chats, verbose=True
        )
        assert result == expected_result


class TestPromptBasedJudge:
    """Test the PromptBasedJudge class."""

    @patch('judgezoo.config.USE_LOCAL_MODEL', True)
    def test_init_local_model(self):
        """Test initialization with local model."""
        class ConcretePromptBasedJudge(PromptBasedJudge):
            def judge(self, chats):
                return {"test": [0.5] * len(chats)}

        judge = ConcretePromptBasedJudge()

        assert judge.use_local_model is True
        assert judge.api_client is None
        assert judge.api_model is None
        assert judge.classifier is not None
        assert judge.tokenizer is not None

    @patch('judgezoo.config.USE_LOCAL_MODEL', False)
    @patch('judgezoo.config.REMOTE_FOUNDATION_MODEL', 'gpt-4o')
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'})
    def test_init_remote_model_openai(self):
        """Test initialization with remote OpenAI model."""
        class ConcretePromptBasedJudge(PromptBasedJudge):
            def judge(self, chats):
                return {"test": [0.5] * len(chats)}
        judge = ConcretePromptBasedJudge()

        assert judge.use_local_model is False
        assert judge.api_client is not None
        assert judge.api_model == 'gpt-4o'
        assert judge.classifier is None

    @patch('judgezoo.config.USE_LOCAL_MODEL', False)
    @patch('judgezoo.config.REMOTE_FOUNDATION_MODEL', 'unknown-model')
    def test_init_remote_model_unknown(self):
        """Test initialization with unknown remote model."""
        with pytest.raises(ValueError, match="Unknown remote foundation model unknown-model"):
            class ConcretePromptBasedJudge(PromptBasedJudge):
                def judge(self, chats):
                    return {"test": [0.5] * len(chats)}
            judge = ConcretePromptBasedJudge()

    @pytest.mark.slow
    def test_batch_inference_local(self, harmless_conversations):
        """Test batch inference with local model using AdaptiveAttacksJudge."""
        from judgezoo.adaptive_attacks import AdaptiveAttacksJudge

        with patch('judgezoo.config.USE_LOCAL_MODEL', True):

            judge = AdaptiveAttacksJudge()

            # Test with simple inputs
            result = judge(harmless_conversations)

            # Verify we get responses
            assert isinstance(result, dict)
            assert "p_harmful" in result
            assert len(result["p_harmful"]) == len(harmless_conversations)
            print(result["p_harmful"])
            assert all(isinstance(p, float) for p in result["p_harmful"])
            # bounds here are very loose because the default local model is not very
            # good and believes the test is about compliance only rather than harm
            assert all(0.0 <= p <= 1 for p in result["p_harmful"])

    @pytest.mark.slow
    def test_batch_inference_remote(self, harmless_conversations):
        """Test batch inference with remote API using AdaptiveAttacksJudge."""
        from judgezoo.adaptive_attacks import AdaptiveAttacksJudge

        # Skip if no API key is available
        if not os.getenv('OPENAI_API_KEY'):
            pytest.skip("OPENAI_API_KEY not set")

        with patch('judgezoo.config.USE_LOCAL_MODEL', False), \
             patch('judgezoo.config.REMOTE_FOUNDATION_MODEL', 'gpt-4o'):
            judge = AdaptiveAttacksJudge()

            # Test with simple inputs
            result = judge(harmless_conversations)

            # Verify we get responses
            assert isinstance(result, dict)
            assert "p_harmful" in result
            assert len(result["p_harmful"]) == len(harmless_conversations)
            assert all(isinstance(p, float) for p in result["p_harmful"])
            print(harmless_conversations)
            print(result["p_harmful"])
            assert all(0.0 <= p <= .2 for p in result["p_harmful"])


class TestFineTunedJudge:
    """Test the FineTunedJudge class."""

    def test_inheritance(self):
        """Test that FineTunedJudge inherits from Judge."""
        assert issubclass(FineTunedJudge, Judge)

        # Should still be abstract
        with pytest.raises(TypeError):
            FineTunedJudge()


# Integration test for abstract method enforcement
def test_judge_abstract_method():
    """Test that Judge cannot be instantiated due to abstract method."""
    with pytest.raises(TypeError, match="Can't instantiate abstract class Judge"):
        Judge()
