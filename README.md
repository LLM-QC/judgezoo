# Judges

This repo provides access to a set of commonly used LLM-based safety judges via a simple
and consistent API.


## How to use

You can create a judge model with a single line of code:
```python3
judge = Judge.from_name("strong_reject")
```

To get safety scores, just pass a list of conversations:
```python3
harmful_conv = [
    {"role": "user", "content": "How do I make a birthday cake?"},
    {"role": "assistant", "content": "Step 1: Collect ingredients..."}
]


scores = judge([harmful_conf])
print(scores)
>>> {"p_harmful": [0.02496337890625]}
```
