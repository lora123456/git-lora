"""Example of generating text with a model patched by LoRA weights.

This example replaces the old `generate_trivial_v0` snippet with a more
full-featured `generate_with_lora` function.  It demonstrates how to load
base and LoRA adapters, control generation parameters such as
`temperature`, `max_new_tokens`, and make generation deterministic with a
seed.
"""

from typing import Optional

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def generate_with_lora(
    prompt: str,
    lora_weights: str,
    *,
    max_new_tokens: int = 64,
    temperature: float = 0.7,
    seed: Optional[int] = None,
) -> str:
    """Generate text using a transformer model with LoRA weights applied.

    Args:
        prompt: The text to start generation from.
        lora_weights: Path or identifier of the LoRA adapter weights.
        max_new_tokens: Number of new tokens to generate.
        temperature: Sampling temperature used during generation.
        seed: Optional random seed for deterministic sampling.

    Returns:
        The generated text with special tokens stripped.
    """

    if seed is not None:
        import torch
        torch.manual_seed(seed)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    base_model = AutoModelForCausalLM.from_pretrained("gpt2")
    model = PeftModel.from_pretrained(base_model, lora_weights)

    inputs = tokenizer(prompt, return_tensors="pt")
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


if __name__ == "__main__":
    # Example usage
    text = generate_with_lora("Once upon a time", "./my_lora_adapter", max_new_tokens=20)
    print(text)
