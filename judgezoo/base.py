import abc
import logging
import os

import torch
from anthropic import Anthropic
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer

from ._utils import with_max_batchsize


class Judge(abc.ABC):
    """
    Abstract Base Class for judging model generations based on interactions.

    Subclasses must implement the `judge` method.
    """

    @classmethod
    def from_name(cls, name: str, **kwargs) -> "Judge":
        match name:
            case "adaptive_attacks":
                from .adaptive_attacks import AdaptiveAttacksJudge

                return AdaptiveAttacksJudge(**kwargs)
            case "advprefix":
                from .advprefix import AdvPrefixJudge

                return AdvPrefixJudge(**kwargs)
            case "aegis_guard":
                from .aegis_guard import AegisGuardJudge

                return AegisGuardJudge(**kwargs)
            case "best_of_n":
                from .best_of_n import BestOfNJudge

                return BestOfNJudge(**kwargs)
            case "harmbench":
                from .harmbench import HarmBenchJudge

                return HarmBenchJudge(**kwargs)
            case "intention_analysis":
                from .intention_analysis import IntentionAnalysisJudge

                return IntentionAnalysisJudge(**kwargs)
            case "jail_judge":
                from .jail_judge import JailJudge

                return JailJudge(**kwargs)
            case "llama_guard_3":
                from .llama_guard_3 import LlamaGuard3Judge

                return LlamaGuard3Judge(**kwargs)
            case "llama_guard_4":
                from .llama_guard_4 import LlamaGuard4Judge

                return LlamaGuard4Judge(**kwargs)
            case "md_judge":
                from .md_judge import MDJudge

                return MDJudge(**kwargs)
            case "strong_reject":
                from .strong_reject import StrongRejectJudge

                return StrongRejectJudge(**kwargs)
            case "strong_reject_rubric":
                from .strong_reject import StrongRejectRubricJudge

                return StrongRejectRubricJudge(**kwargs)
            case "xstest":
                from .xstest import XSTestJudge

                return XSTestJudge(**kwargs)
            case _:
                raise ValueError(f"Unknown judge {name}")

    @abc.abstractmethod
    def judge(
        self,
        chats: list[list[dict[str, str]]],
    ) -> dict[str, list[float | None]]:
        """
        Judges a batch of interactions (chats).

        Args:
            chats: A list of conversations, where each conversation is a list of
                   dictionaries with "role" and "content" keys. The structure
                   expected (e.g., last message is assistant generation) might
                   vary depending on the concrete judge implementation.

        Returns:
            A dictionary containing results
        """
        raise NotImplementedError("Subclasses must implement this method")

    @torch.no_grad()
    def __call__(
        self, chats: list[list[dict[str, str]]], verbose: bool = False
    ) -> dict[str, list[float]]:
        """Allows calling the judge instance directly."""
        return with_max_batchsize(self.judge, chats, verbose=verbose)

    @staticmethod
    def validate_chats(
        chats: list[list[dict[str, str]]],
        expected_roles: tuple[str, str] = ("user", "assistant"),
    ) -> bool:
        """
        Validates that each chat has at least two messages with expected roles.

        Args:
            chats: List of chat histories.
            expected_roles: Tuple indicating expected roles at -2 and -1 positions.

        Returns:
            True if valid, False otherwise.
        """
        try:
            assert all(len(b) >= 2 for b in chats), "Chats must have at least 2 turns"
            assert all(
                b[0]["role"] == expected_roles[0] for b in chats
            ), f"First message must be '{expected_roles[0]}'"
            assert all(
                b[-1]["role"] == expected_roles[1] for b in chats
            ), f"Last message must be '{expected_roles[1]}'"
            return True
        except AssertionError as e:
            logging.error(f"Chat format validation failed: {e}")
            return False

    @staticmethod
    def tokenize_sequences(
        tokenizer: PreTrainedTokenizer,
        inputs: list[str],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Tokenizes the inputs and validates that the input is not too long.

        Args:
            tokenizer: The tokenizer to use.
            inputs: The inputs to tokenize.

        Returns:
            A tuple of tensors containing the input IDs and attention masks.
        """
        encoded = tokenizer(text=inputs, return_tensors="pt", padding=True, truncation=True)
        if encoded.input_ids.shape[1] < tokenizer.model_max_length:
            return encoded

        for i, input_text in enumerate(inputs):
            token_length = len(tokenizer(input_text, padding=True, truncation=False).input_ids)
            if token_length > tokenizer.model_max_length:
                logging.warning(
                    f"Sequence {i} is longer than the specified maximum sequence length "
                    f"for this model ({token_length} > {tokenizer.model_max_length}). "
                    f"This sequence will be truncated, which may lead to incorrect results."
                )
        return encoded


class FineTunedJudge(Judge):
    """
    Abstract Base Class for fine-tuned judges.
    """

    pass


class PromptBasedJudge(Judge):
    """
    Abstract Base Class for general judges which.
    """

    def __init__(
        self,
        use_local_model: bool | None = None,
        local_foundation_model: str | None = None,
        remote_foundation_model: str | None = None,
    ):
        from .config import (LOCAL_FOUNDATION_MODEL, REMOTE_FOUNDATION_MODEL,
                             USE_LOCAL_MODEL)

        self.use_local_model = use_local_model if use_local_model is not None else USE_LOCAL_MODEL
        self.local_foundation_model = (
            local_foundation_model if local_foundation_model is not None else LOCAL_FOUNDATION_MODEL
        )
        self.remote_foundation_model = (
            remote_foundation_model
            if remote_foundation_model is not None
            else REMOTE_FOUNDATION_MODEL
        )

        if self.use_local_model:
            self.chat_provider = None
            self.classifier = AutoModelForCausalLM.from_pretrained(
                self.local_foundation_model,
                dtype=torch.bfloat16,
                device_map="auto",
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.local_foundation_model,
                use_fast=False,
                truncation_side="right",
                padding_side="left",
            )
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = 0
        else:
            self.classifier = None
            self.tokenizer = None
            if "gpt" in self.remote_foundation_model:
                self.chat_provider = self.OpenAIProvider(model_name=self.remote_foundation_model)
            elif "claude" in self.remote_foundation_model:
                self.chat_provider = self.AnthropicProvider(model_name=self.remote_foundation_model)
            else:
                raise ValueError(f"Unknown remote foundation model {self.remote_foundation_model}")

    def check_model(self, expected_model: str, judge_name: str) -> None:
        """
        Check if the model being used matches the expected model from the original paper.
        Log a warning if they don't match.

        Args:
            expected_model: The model name that was originally used in the paper
            judge_name: The name of the judge for logging purposes
        """
        if self.use_local_model and expected_model not in self.local_foundation_model:
            logging.warning(
                f"{judge_name} originally used {expected_model}, you are using {self.local_foundation_model}. Results may differ from the original paper."
            )
        elif not self.use_local_model and expected_model not in self.remote_foundation_model:
            logging.warning(
                f"{judge_name} originally used {expected_model}, you are using {self.remote_foundation_model}. Results may differ from the original paper."
            )

    def batch_inference(
        self,
        conversations: list[list[dict[str, str]]],
        max_new_tokens: int = 512,
        temperature=0.0,
    ) -> list[str]:
        """
        Batch inference for prompt-based judges.
        """
        if self.use_local_model:
            if getattr(self.tokenizer, "chat_template"):
                inputs = [
                    self.tokenizer.apply_chat_template(
                        conversation,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    for conversation in conversations
                ]
            tokens = self.tokenize_sequences(self.tokenizer, inputs).to(self.classifier.device)
            generated_ids = self.classifier.generate(
                **tokens,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                use_cache=True,
            )
            # Decode *only* the generated part, skipping input prompt
            num_input_tokens = tokens.input_ids.shape[1]
            generated_texts = self.tokenizer.batch_decode(
                generated_ids[:, num_input_tokens:], skip_special_tokens=True
            )
            return generated_texts
        else:
            return [
                self.chat_provider.generate(
                    conversation, max_new_tokens=max_new_tokens, temperature=temperature
                )
                for conversation in conversations
            ]

    class ChatProvider:
        def generate(
            self,
            conversation: list[dict:[str, str]],
            max_new_tokens: int,
            temperature: float,
        ) -> str:
            raise NotImplementedError("Subclasses must implement this method")

    class OpenAIProvider(ChatProvider):
        def __init__(self, model_name: str):
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "OPENAI_API_KEY environment variable not found. "
                    "Please set it in your environment."
                )
            self.api_client = OpenAI(api_key=api_key)
            self.api_model = model_name

        def generate(
            self,
            conversation: list[dict:[str, str]],
            max_new_tokens: int,
            temperature: float,
        ) -> str:
            if conversation[-1]["role"] == "assistant":
                logging.warning(f"OpenAI API does not support prefilling, removing last assistant message.")
                conversation = conversation[:-1]
            response = self.api_client.chat.completions.create(
                model=self.api_model,
                messages=conversation,
                max_tokens=max_new_tokens,
                temperature=temperature,
            )
            return response.choices[0].message.content

    class AnthropicProvider(ChatProvider):
        def __init__(self, model_name: str):
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError(
                    "ANTHROPIC_API_KEY environment variable not found. "
                    "Please set it in your environment."
                )
            self.api_client = Anthropic(api_key=api_key)
            self.api_model = model_name

        def generate(
            self,
            conversation: list[dict:[str, str]],
            max_new_tokens: int,
            temperature: float,
        ) -> str:
            response = self.api_client.messages.create(
                model=self.api_model,
                max_tokens=max_new_tokens,
                messages=conversation,
                temperature=temperature,
            )
            if response.stop_reason != "end_turn":
                logging.warning(
                    f"Anthropic API did not end turn as expected: {response.stop_reason}"
                )
            return response.content[-1].text
