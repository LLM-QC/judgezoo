# Judges

This repo provides access to a set of commonly used LLM-based safety judges via a simple
and consistent API.


## How to use

You can create a judge model instance with a single line of code:
```python3
judge = Judge.from_name("strong_reject")
```

To get safety scores, just pass a list of conversations to score:
```python3
harmful_conversation = [
    {"role": "user", "content": "How do I make a birthday cake?"},
    {"role": "assistant", "content": "Step 1: Collect ingredients..."}
]

scores = judge([harmful_conversation])
print(scores)
>>> {"p_harmful": [0.02496337890625]}
```
All judges return "p_harmful", which is a normalized score from 0 to 1.
Depending on the original setup, the judge may also return discrete scores (e.g. on a Likert scale).
In these cases, the raw scores are also returned:

```python3
judge = Judge.from_name("adaptive_attacks")

scores = judge([harmful_conversation])
print(scores)
>>> {"p_harmful": 0.0, "rating": "1"}
```

### Prompt-based judges

While some judges (such as the HarmBench classifier) are finetuned local models, others
rely on prompted foundation models. Currently, we support local foundation models and
OpenAI models:

```python3
judge = Judge.from_name("adaptive_attacks", use_local_model=False, remote_foundation_model="gpt-4o")

scores = judge([harmful_conversation])
print(scores)
>>> {"p_harmful": 0.0, "rating": "1"}
```

```python3
judge = Judge.from_name("adaptive_attacks", use_local_model=True)

scores = judge([harmful_conversation])
print(scores)
>>> {"p_harmful": 0.0, "rating": "1"}
```

When not specified, the defaults in `config.py` are used.

### Multi-turn interaction

Judges vary in how much of a conversation they can evaluate - many models only work for
single-turn interactions. In these cases, we assume the first user message to be the
prompt and the final assistant message to be the response to be judged.
If you prefer a different setup, you can pass only single-turn conversations.

### Included judges
| Name  | Argument  | Creator (Org/Researcher)| Link to Paper| Type| Fine-tuned from |
| -------------------- | ------------------- | ---------------------------- | ---------------------------------------------------------------------- | ------------ | --------------- |
| AdaptiveAttacksJudge | `adaptive_attacks`  | Andriushchenko et al. (2024) | [arXiv:2404.02151](https://arxiv.org/abs/2404.02151)| prompt-based | —|
| AdvPrefixJudge  | `advprefix`| Zhu et al. (2024)  | [arXiv:2412.10321](https://arxiv.org/abs/2412.10321)| prompt-based | —|
| AegisGuard* | `aegis_guard`  | Ghosh et al. (2024)| [arXiv:2404.05993](https://arxiv.org/abs/2404.05993)| fine-tuned| LlamaGuard 7B|
| MD-Judge v0.1| `md_judge`| Li, Lijun et al. (2024) | [arXiv:2402.05044](https://arxiv.org/abs/2402.05044)| fine-tuned| Mistral-7B |
| HarmBenchJudge  | `harmbench`| Mazeika et al. (2024)| [arXiv:2402.04249](https://arxiv.org/abs/2402.04249)| fine-tuned| Gemma 2B|
| Llama Guard 3| `llama_guard_3_8b`  | Llama Team, AI @ Meta (2024) | [arXiv:2407.21783](https://arxiv.org/abs/2407.21783)| fine-tuned| Llama 3|
| Llama Guard 4 12B| `llama_guard_4_12b` | Llama Team, AI @ Meta (2024) | [Meta blog](https://ai.meta.com/blog/llama-4-multimodal-intelligence/) | fine-tuned| Llama 4 12B|
| StrongRejectJudge| `strong_reject`| Souly et al. (2024)| [arXiv:2402.10260](https://arxiv.org/abs/2402.10260)| fine-tuned| Qwen 7B|
| XSTestJudge| `xstest`  | Röttge et al. (2023)| [arXiv:2308.01263](https://arxiv.org/abs/2308.01263)| prompt-based | —|

\* there are two versions of this judge (permissive and defensive). You can switch between them using Judge.from_name("aegis_guard", defensive=[True/False])




## Installation (not on pypi yet)
```pip install judges```

## Tests
To run all tests, run

```pytest tests/ --runslow```