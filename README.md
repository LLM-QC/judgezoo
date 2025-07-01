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
harmless_conversation = [
    {"role": "user", "content": "How do I make a birthday cake?"},
    {"role": "assistant", "content": "Step 1: Collect ingredients..."}
]

scores = judge([harmless_conversation])
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

Judges vary in how much of a conversation they can evaluate - many models only work for
single-turn interactions. In these cases, we assume the first user message to be the
prompt and the final assistant message to be the response to be judged.
If you prefer a different setup, you can pass only single-turn conversations.




## Installation (not on pypi yet)
```pip install judges```

## Tests
To run all tests, run

```pytest tests/ --runslow```
