# JudgeZoo

This repo provides access to a set of commonly used LLM-based safety judges via a simple and consistent API.
Our main focus is ease-of-use, correctness, and reproducibility.


## How to use

You can create a judge model instance with a single line of code:
```python3
judge = Judge.from_name("strong_reject")
```

To get safety scores, just pass a list of conversations to score:
```python3
harmless_conversation = [
    {"role": "user", "content": "How do I make a birthday cake?"},
    {"role": "assistant", "content": "Step 1: Collect ingredients..."}
]

scores = judge([harmless_conversation])
print(scores)
>>> {"p_harmful": [0.02496337890625]}
```
All judges return `"p_harmful"`, which is a normalized score from 0 to 1.
Depending on the original setup, the judge may also return discrete scores or harm categories (e.g. on a Likert scale).
In these cases, the raw scores are also returned:

```python3
judge = Judge.from_name("adaptive_attacks")

scores = judge([harmful_conversation])
print(scores)
>>> {"p_harmful": 0.0, "rating": "1"}
```


## Included judges
| Name                   | Argument              | Creator (Org/Researcher)     | Link to Paper                                       | Type         | Fine-tuned from        |
| ---------------------- | --------------------- | ---------------------------- | ----------------------------------------------------| ------------ | ---------------------- |
| Adaptive Attacks       | `adaptive_attacks`    | Andriushchenko et al. (2024) | [arXiv:2404.02151](https://arxiv.org/abs/2404.02151)| prompt-based | —                      |
| AdvPrefix              | `advprefix`           | Zhu et al. (2024)            | [arXiv:2412.10321](https://arxiv.org/abs/2412.10321)| prompt-based | —                      |
| AegisGuard*            | `aegis_guard`         | Ghosh et al. (2024)          | [arXiv:2404.05993](https://arxiv.org/abs/2404.05993)| fine-tuned   | LlamaGuard 7B          |
| HarmBench              | `harmbench`           | Mazeika et al. (2024)        | [arXiv:2402.04249](https://arxiv.org/abs/2402.04249)| fine-tuned   | Gemma 2B               |
| JailJudge              | `jail_judge`          | Liu et al. (2024)            | [arXiv:2410.12855](https://arxiv.org/abs/2410.12855)| fine-tuned   | Llama 2 7B
| Llama Guard 3          | `llama_guard_3`       | Llama Team, AI @ Meta (2024) | [arXiv:2407.21783](https://arxiv.org/abs/2407.21783)| fine-tuned   | Llama 3 8B             |
| Llama Guard 4          | `llama_guard_4`       | Llama Team, AI @ Meta (2024) | [Meta blog](https://ai.meta.com/blog/llama-4-multimodal-intelligence/) | fine-tuned | Llama 4 12B |
| MD-Judge (v0.1 & v0.2) | `md_judge`            | Li, Lijun et al. (2024)      | [arXiv:2402.05044](https://arxiv.org/abs/2402.05044)| fine-tuned   | Mistral-7B/LMintern2 7B|
| StrongREJECT           | `strong_reject`       | Souly et al. (2024)          | [arXiv:2402.10260](https://arxiv.org/abs/2402.10260)| fine-tuned   | Gemma 2b               |
| StrongREJECT (rubric)  | `strong_reject_rubric`| Souly et al. (2024)          | [arXiv:2402.10260](https://arxiv.org/abs/2402.10260)| prompt-based | -                      |
| XSTestJudge            | `xstest`              | Röttge et al. (2023)         | [arXiv:2308.01263](https://arxiv.org/abs/2308.01263)| prompt-based | —                      |

\* there are two versions of this judge (permissive and defensive). You can switch between them using `Judge.from_name("aegis_guard", defensive=[True/False])`


## Other
### Prompt-based judges

While some judges (such as the HarmBench classifier) are finetuned local models, others rely on prompted foundation models.
Currently, we support local foundation models and OpenAI models:

```python3
judge = Judge.from_name("adaptive_attacks", use_local_model=False, remote_foundation_model="gpt-4o")

scores = judge([harmless_conversation])
print(scores)
>>> {"p_harmful": 0.0, "rating": "1"}
```

```python3
judge = Judge.from_name("adaptive_attacks", use_local_model=True)

scores = judge([harmless_conversation])
print(scores)
>>> {"p_harmful": 0.0, "rating": "1"}
```

When not specified, the defaults in `config.py` are used.

### Multi-turn interaction

Judges vary in how much of a conversation they can evaluate - many models only work for single-turn interactions.
In these cases, we assume the first user message to be the prompt and the final assistant message to be the response to be judged.
If you prefer a different setup, you can pass only single-turn conversations.

### Reproducibility

Wherever possible, we use official code directly provided by the original authors to ensure correctness.

Finally, we warn if a user's setup diverges from the original implementation:

```python3
from judgezoo import Judge

judge = Judge.from_name("intention_analysis")
>>> WARNING:root:IntentionAnalysisJudge originally used gpt-3.5-turbo-0613, you are using gpt-4o. Results may differ from the original paper.
```


## Installation
```pip install judgezoo```


## Tests
To run all tests, run

```pytest tests/ --runslow```
