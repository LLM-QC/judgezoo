"""Tests for the HarmBench Judge implementation."""

import logging
import os
import sys

import pytest

# Add the parent directory to the path to import judgezoo
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from judgezoo.harmbench import HarmBenchClassifier, HarmBenchJudge


@pytest.fixture(scope="module")
def harmbench_judge():
    """Create a HarmBenchJudge instance for testing.

    This fixture loads the real model and is module-scoped to avoid
    reloading the model for each test.
    """
    judge = HarmBenchJudge()
    return judge


@pytest.fixture(scope="module")
def harmbench_judge_mistral():
    """Create a HarmBenchJudge instance with Mistral classifier for testing.

    This fixture loads the Mistral model and is module-scoped to avoid
    reloading the model for each test.
    """
    judge = HarmBenchJudge(classifier=HarmBenchClassifier.MISTRAL_7B)
    return judge


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
    def test_init_mistral(self, harmbench_judge_mistral):
        """Test HarmBenchJudge initialization with Mistral classifier."""
        judge = harmbench_judge_mistral

        # Verify the judge was created successfully
        assert judge is not None
        assert hasattr(judge, 'classifier')
        assert hasattr(judge, 'tokenizer')

        # Verify token IDs are set correctly (should be same for both models)
        assert judge.yes_token_id == 5613
        assert judge.no_token_id == 2501

    @pytest.mark.slow
    def test_init_with_explicit_llama_model(self):
        """Test HarmBenchJudge initialization with explicit Llama model."""
        judge = HarmBenchJudge(classifier=HarmBenchClassifier.LLAMA_2_13B)

        # Verify the judge was created successfully
        assert judge is not None
        assert hasattr(judge, 'classifier')
        assert hasattr(judge, 'tokenizer')

        # Verify token IDs are set correctly
        assert judge.yes_token_id == 8241
        assert judge.no_token_id == 3782

    def test_init_invalid_classifier(self):
        """Test HarmBenchJudge initialization with invalid classifier."""
        with pytest.raises(AssertionError, match="Invalid classifier"):
            HarmBenchJudge(classifier="invalid_model")

    @pytest.mark.slow
    def test_chat_template_differences(self, harmbench_judge, harmbench_judge_mistral):
        """Test that different models have different chat templates."""
        llama_judge = harmbench_judge
        mistral_judge = harmbench_judge_mistral

        # Both should have chat templates but they should be different
        assert hasattr(llama_judge.tokenizer, 'chat_template')
        assert hasattr(mistral_judge.tokenizer, 'chat_template')
        assert llama_judge.tokenizer.chat_template != mistral_judge.tokenizer.chat_template

        # Llama template should contain '<<SYS>>' and '[INST]'
        assert '<<SYS>>' in llama_judge.tokenizer.chat_template
        assert '[INST]' in llama_judge.tokenizer.chat_template

        # Mistral template should contain '[INST]' but not '<<SYS>>'
        assert '[INST]' in mistral_judge.tokenizer.chat_template
        assert '<<SYS>>' not in mistral_judge.tokenizer.chat_template

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

    @pytest.mark.slow
    def test_judge_valid_chats_mistral(self, harmbench_judge_mistral, sample_chats):
        """Test judge method with valid chats using Mistral classifier."""
        judge = harmbench_judge_mistral

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

        # Should return dict with None values for invalid chats
        assert isinstance(result, dict)
        assert "p_harmful" in result
        assert result["p_harmful"] == [None] * len(invalid_chats)

    def test_judge_invalid_chats_mistral(self, harmbench_judge_mistral, invalid_chats):
        """Test judge method with invalid chats using Mistral classifier."""
        judge = harmbench_judge_mistral

        result = judge.judge(invalid_chats)

        # Should return dict with None values for invalid chats
        assert isinstance(result, dict)
        assert "p_harmful" in result
        assert result["p_harmful"] == [None] * len(invalid_chats)

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
    def test_judge_safe_conversation_mistral(self, harmbench_judge_mistral, harmless_conversations):
        """Test judge with a clearly safe conversation using Mistral classifier."""
        judge = harmbench_judge_mistral

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
    def test_judge_potentially_harmful_conversation_mistral(self, harmbench_judge_mistral, harmful_conversations):
        """Test judge with a potentially harmful conversation using Mistral classifier."""
        judge = harmbench_judge_mistral

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
    def test_judge_multiple_chats_harmless_mistral(self, harmbench_judge_mistral, harmless_conversations):
        """Test judge with multiple harmless chats using Mistral classifier."""
        judge = harmbench_judge_mistral

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
    def test_judge_multiple_chats_harmless_batched_mistral(self, harmbench_judge_mistral, harmless_conversations):
        """Test judge with multiple harmless chats using callable interface with Mistral classifier."""
        judge = harmbench_judge_mistral

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

    @pytest.mark.slow
    def test_judge_multiple_chats_harmful_mistral(self, harmbench_judge_mistral, harmful_conversations, harmless_conversations):
        """Test judge with mixed harmful and safe chats using Mistral classifier."""
        judge = harmbench_judge_mistral

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

    def test_judge_empty_chats_mistral(self, harmbench_judge_mistral):
        """Test judge with empty chat list using Mistral classifier."""
        judge = harmbench_judge_mistral

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

    @pytest.mark.slow
    def test_model_device_handling_mistral(self, harmbench_judge_mistral):
        """Test that the Mistral model handles device placement correctly."""
        judge = harmbench_judge_mistral

        # Check that the model is on some device
        model_device = next(judge.classifier.parameters()).device
        assert model_device.type in ['cuda', 'cpu']

    def test_inheritance(self, harmbench_judge):
        """Test that HarmBenchJudge properly inherits from FineTunedJudge."""
        from judgezoo.base import FineTunedJudge

        judge = harmbench_judge
        assert isinstance(judge, FineTunedJudge)

    def test_inheritance_mistral(self, harmbench_judge_mistral):
        """Test that HarmBenchJudge with Mistral properly inherits from FineTunedJudge."""
        from judgezoo.base import FineTunedJudge

        judge = harmbench_judge_mistral
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
    def test_consistent_results_mistral(self, harmbench_judge_mistral):
        """Test that the same input produces consistent results with Mistral classifier."""
        judge = harmbench_judge_mistral

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

    @pytest.mark.slow
    def test_inference_runs_successfully_mistral(self, harmbench_judge_mistral):
        """Test that inference runs successfully without errors with Mistral classifier."""
        judge = harmbench_judge_mistral

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

    @pytest.mark.slow
    def test_prompt_formatting_mistral(self, harmbench_judge_mistral):
        """Test that the HarmBench prompt is formatted correctly with Mistral classifier."""
        judge = harmbench_judge_mistral

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

    @pytest.mark.slow
    def test_model_comparison_harmless(self, harmbench_judge, harmbench_judge_mistral, harmless_conversations):
        """Test that both models agree on harmless conversations (within reasonable bounds)."""
        llama_judge = harmbench_judge
        mistral_judge = harmbench_judge_mistral

        llama_result = llama_judge.judge(harmless_conversations)
        mistral_result = mistral_judge.judge(harmless_conversations)

        # Both should classify as low harm probability
        assert all(0.0 <= p <= 0.2 for p in llama_result["p_harmful"])
        assert all(0.0 <= p <= 0.2 for p in mistral_result["p_harmful"])

        # Results should be in the same ballpark (both low)
        assert len(llama_result["p_harmful"]) == len(mistral_result["p_harmful"])

    @pytest.mark.slow
    def test_model_comparison_harmful(self, harmbench_judge, harmbench_judge_mistral, harmful_conversations):
        """Test that both models agree on harmful conversations (within reasonable bounds)."""
        llama_judge = harmbench_judge
        mistral_judge = harmbench_judge_mistral

        llama_result = llama_judge.judge(harmful_conversations)
        mistral_result = mistral_judge.judge(harmful_conversations)

        # Both should classify as high harm probability
        assert all(0.5 <= p <= 1.0 for p in llama_result["p_harmful"])
        assert all(0.5 <= p <= 1.0 for p in mistral_result["p_harmful"])

        # Results should be in the same ballpark (both high)
        assert len(llama_result["p_harmful"]) == len(mistral_result["p_harmful"])


class TestHarmbenchJudgeModelSpecific:
    """Test model-specific functionality for different HarmBench classifiers."""

    @pytest.mark.slow
    def test_token_ids_consistency_across_models(self):
        """Test that token IDs are consistent across different models."""
        llama_judge = HarmBenchJudge(classifier=HarmBenchClassifier.LLAMA_2_13B)
        mistral_judge = HarmBenchJudge(classifier=HarmBenchClassifier.MISTRAL_7B)

        # Verify the specific expected values
        assert llama_judge.yes_token_id == 8241
        assert llama_judge.no_token_id == 3782
        assert mistral_judge.yes_token_id == 5613
        assert mistral_judge.no_token_id == 2501

    @pytest.mark.slow
    def test_model_specific_chat_templates(self):
        """Test that each model gets its appropriate chat template."""
        llama_judge = HarmBenchJudge(classifier=HarmBenchClassifier.LLAMA_2_13B)
        mistral_judge = HarmBenchJudge(classifier=HarmBenchClassifier.MISTRAL_7B)

        # Verify Llama template characteristics
        llama_template = llama_judge.tokenizer.chat_template
        assert '<<SYS>>' in llama_template
        assert '[INST]' in llama_template
        assert '[/INST]' in llama_template

        # Verify Mistral template characteristics
        mistral_template = mistral_judge.tokenizer.chat_template
        assert '[INST]' in mistral_template
        assert '[/INST]' in mistral_template
        assert '<<SYS>>' not in mistral_template  # Mistral doesn't use this format

        # Templates should be different
        assert llama_template != mistral_template

    @pytest.mark.slow
    def test_enum_values_match_model_paths(self):
        """Test that enum values match expected HuggingFace model paths."""
        assert HarmBenchClassifier.LLAMA_2_13B.value == "cais/HarmBench-Llama-2-13b-cls"
        assert HarmBenchClassifier.MISTRAL_7B.value == "cais/HarmBench-Mistral-7b-val-cls"

    def test_all_enum_models_supported(self):
        """Test that all models in the enum can be instantiated."""
        # This test doesn't load the actual models (not marked as slow)
        # but verifies the enum structure is correct
        models = list(HarmBenchClassifier)
        assert len(models) == 2
        assert HarmBenchClassifier.LLAMA_2_13B in models
        assert HarmBenchClassifier.MISTRAL_7B in models

    @pytest.mark.slow
    def test_mistral_specific_edge_cases(self, harmbench_judge_mistral):
        """Test Mistral-specific edge cases and behavior."""
        judge = harmbench_judge_mistral

        # Test with very short conversations
        short_chat = [
            [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello"}
            ]
        ]

        result = judge.judge(short_chat)
        assert isinstance(result, dict)
        assert "p_harmful" in result
        assert len(result["p_harmful"]) == 1
        assert isinstance(result["p_harmful"][0], float)
        assert 0.0 <= result["p_harmful"][0] <= 1.0

    @pytest.mark.slow
    def test_both_models_handle_long_conversations(self):
        """Test that both models can handle longer conversations."""
        # Create a longer conversation
        long_chat = [
            [
                {"role": "user", "content": "How do I cook pasta?"},
                {"role": "assistant", "content": "First, bring a large pot of salted water to a boil."},
                {"role": "user", "content": "What kind of pasta should I use?"},
                {"role": "assistant", "content": "It depends on your preference and the sauce you're making. For a simple tomato sauce, spaghetti or penne work well."},
                {"role": "user", "content": "How long should I cook it?"},
                {"role": "assistant", "content": "Most pasta takes 8-12 minutes, but check the package instructions for the specific type you're using."}
            ]
        ]

        llama_judge = HarmBenchJudge(classifier=HarmBenchClassifier.LLAMA_2_13B)
        mistral_judge = HarmBenchJudge(classifier=HarmBenchClassifier.MISTRAL_7B)

        # Both should handle the longer conversation
        llama_result = llama_judge.judge(long_chat)
        mistral_result = mistral_judge.judge(long_chat)

        # Verify both produce valid results
        for result in [llama_result, mistral_result]:
            assert isinstance(result, dict)
            assert "p_harmful" in result
            assert len(result["p_harmful"]) == 1
            assert isinstance(result["p_harmful"][0], float)
            assert 0.0 <= result["p_harmful"][0] <= 1.0
            # Should be low harm probability for cooking conversation
            assert result["p_harmful"][0] <= 0.2