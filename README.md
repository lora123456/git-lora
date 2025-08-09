# git-lora

This repository demonstrates how to generate text with transformer models
patched by [LoRA](https://arxiv.org/abs/2106.09685) adapters.

## Example

The trivial `generate_trivial_v0()` sample has been replaced by the more
capable `generate_with_lora()` function.  It shows how to load a base model,
apply LoRA weights, and control generation parameters such as temperature,
maximum token count, and randomness seed.

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from example import generate_with_lora

# Path to your LoRA adapter weights
ADAPTER = "./my_lora_adapter"

text = generate_with_lora(
    "Once upon a time",
    ADAPTER,
    max_new_tokens=50,
    temperature=0.8,
    seed=1234,
)
print(text)
```
