import abc
import logging
import os
from openai import OpenAI

import torch
from transformers import PreTrainedTokenizer, AutoModelForCausalLM, AutoTokenizer
from ._utils import with_max_batchsize


class Judge(abc.ABC):
    """
    Abstract Base Class for judging model generations based on interactions.

    Subclasses must implement the `judge` method.
    """
    @classmethod
    def from_name(cls, name: str, defensive: bool = False) -> "Judge":
        match name:
            case "adaptive_attacks":
                from .adaptive_attacks import AdaptiveAttacksJudge
                return AdaptiveAttacksJudge()
            case "advprefix":
                from .advprefix import AdvPrefixJudge
                return AdvPrefixJudge()
            case "aegis_guard":
                from .aegis_guard import AegisGuardJudge
                return AegisGuardJudge(defensive=defensive)
            case "harmbench":
                from .harmbench import HarmBenchJudge
                return HarmBenchJudge()
            case "llama_guard_3_8b":
                from .llama_guard_3 import LlamaGuard3Judge
                return LlamaGuard3Judge()
            case "llama_guard_4_12b":
                from .llama_guard_4 import LlamaGuard4Judge
                return LlamaGuard4Judge()
            case "md_judge":
                from .md_judge import MDJudge
                return MDJudge()
            case "strong_reject":
                from .strong_reject import StrongRejectJudge
                return StrongRejectJudge()
            case "xstest":
                from .xstest import XSTestJudge
                return XSTestJudge()
            case _:
                raise ValueError(f"Unknown judge {name}")

    @abc.abstractmethod
    def judge(
        self,
        chats: list[list[dict[str, str]]],
    ) -> dict[str, list[float]] | None:
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
    def __call__(self, chats: list[list[dict[str, str]]], verbose: bool = False) -> dict[str, list[float]]:
        """Allows calling the judge instance directly."""
        return with_max_batchsize(self.judge, chats, verbose=verbose)

    @staticmethod
    def validate_chats(
        chats: list[list[dict[str, str]]],
        expected_roles: tuple[str, str] = ("user", "assistant")
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
            assert all(b[0]["role"] == expected_roles[0] for b in chats), f"First message must be '{expected_roles[0]}'"
            assert all(b[-1]["role"] == expected_roles[1] for b in chats), f"Last message must be '{expected_roles[1]}'"
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
        Validates that the input is not too long.
        """
        encoded = tokenizer(
            text=inputs, return_tensors="pt", padding=True, truncation=True
        )
        if encoded.input_ids.shape[1] < tokenizer.model_max_length:
            return encoded
        for i in range(len(inputs)):
            if (l := len(tokenizer(inputs[i], padding=True, truncation=False).input_ids)) > tokenizer.model_max_length:
                logging.warning(f"Sequence is longer than the specified maximum sequence length for this model ({l} > {tokenizer.model_max_length}). This sequence will be truncated, which may lead to incorrect results.")
                return encoded
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
        from .config import REMOTE_FOUNDATION_MODEL, LOCAL_FOUNDATION_MODEL, USE_LOCAL_MODEL
        self.use_local_model = use_local_model if use_local_model is not None else USE_LOCAL_MODEL
        local_foundation_model = local_foundation_model if local_foundation_model is not None else LOCAL_FOUNDATION_MODEL
        remote_foundation_model = remote_foundation_model if remote_foundation_model is not None else REMOTE_FOUNDATION_MODEL

        if self.use_local_model:
            self.api_client = None
            self.api_model = None
            self.classifier = AutoModelForCausalLM.from_pretrained(
                local_foundation_model,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                local_foundation_model,
                use_fast=False,
                truncation_side="right",
                padding_side="left"
            )
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = 0
        else:
            if "gpt" in remote_foundation_model:
                self.api_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                self.api_model = remote_foundation_model
                self.classifier = None
            else:
                raise ValueError(f"Unknown remote foundation model {remote_foundation_model}")

    def batch_inference(self, inputs: list[str]) -> list[str]:
        """
        Batch inference for prompt-based judges.
        """
        if self.use_local_model:
            if getattr(self.tokenizer, "chat_template"):
                inputs = [
                    self.tokenizer.apply_chat_template(
                        [{"role": "user", "content": input}],
                        tokenize=False,
                        add_generation_prompt=True
                    )
                    for input in inputs
                ]
            tokens = self.tokenize_sequences(self.tokenizer, inputs).to(self.classifier.device)
            generated_ids = self.classifier.generate(
                **tokens,
                do_sample=False,
                max_new_tokens=128,
                use_cache=True,
            )
            # Decode *only* the generated part, skipping input prompt
            num_input_tokens = tokens.input_ids.shape[1]
            generated_texts = self.tokenizer.batch_decode(
                generated_ids[:, num_input_tokens:],
                skip_special_tokens=True
            )
            return generated_texts
        else:
            return [
                self.api_client.chat.completions.create(
                    model=self.api_model,
                    messages=[{"role": "user", "content": input}]
                ).choices[0].message.content
                for input in inputs
            ]