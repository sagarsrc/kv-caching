# %% [markdown]
# # TinyLlama Model Loading and Inference
# This notebook demonstrates loading the TinyLlama model and running basic inference and getting Q, K, V, O projections

# %% [markdown]
# ## Import required libraries
import torch
import pickle
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, Any
import pandas as pd

# custom utils
from attention_helpers.qkvo_hooks import capture_model_attention_internals

# %% [markdown]
# ## Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0", padding=False
)
model = AutoModelForCausalLM.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    device_map="cpu",
    torch_dtype=torch.float16,
)

print(model)

# %% [markdown]
# ## Model Information
print(f"Model Parameters: {model.num_parameters():,}")
print(f"Model Architecture: {model.config.architectures[0]}")
print(f"Model Context Length: {model.config.max_position_embeddings}")


# %% [markdown]
# ## Generate Text
def generate_text(
    messages,
    model,
    tokenizer,
    max_tokens=100,
    temperature=0,
    verbose=True,
    padding=True,
    truncation=True,
):
    """
    Generate text using TinyLlama model with chat template

    Args:
        messages (list): List of message dictionaries with 'role' and 'content'
        model: The language model to use
        tokenizer: The tokenizer to use
        max_tokens (int): Maximum number of tokens to generate
        temperature (float): Sampling temperature (0.0 = deterministic)

    Returns:
        str: Generated text
    """
    # Apply chat template to format messages
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        padding=padding,
        truncation=truncation,
    )

    # Encode the prompt with attention mask
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=padding,
        truncation=truncation,
        return_attention_mask=True,  # Explicitly request attention mask
    )

    if verbose:
        # Print tokenization information
        print("\n\nTokenization Information:")
        text = tokenizer.apply_chat_template(messages, tokenize=False, padding=False)
        tokens = tokenizer.encode(
            text, padding=False, truncation=True, return_tensors="pt"
        )
        print(f"Input sequence length: {tokens.shape[1]} tokens")

        # Create DataFrame for token visualization
        token_data = {
            "token_index": range(len(tokens[0])),
            "token": tokens[0].tolist(),
            "decoded_token": [tokenizer.decode([t]) for t in tokens[0]],
        }
        df = pd.DataFrame(token_data)
        print("\nToken Details:")
        print(df.to_string(index=False))

    # Generate with attention mask
    outputs = model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,  # Add attention mask
        max_new_tokens=max_tokens,
        do_sample=(temperature > 0),
        pad_token_id=tokenizer.eos_token_id,
    )

    # Decode and return the generated text, keeping special tokens
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=False)
    if verbose:
        print(
            f"\nGenerated tokens: {len(outputs[0]) - len(inputs.input_ids[0])} tokens"
        )
        print(f"Total tokens in final sequence: {len(outputs[0])}")
        print(f"\nGenerated Text:\n{decoded_output}")
        print("--------------------------------")
    return decoded_output


# %% [markdown]
# ## Example Usage
messages = [
    {
        "role": "system",
        "content": "You are a friendly chat assistant",
    },
    {
        "role": "user",
        "content": "What is the capital of India?",
    },
]

generated_text = generate_text(messages, model, tokenizer, verbose=True)


# %% [markdown]
# ## Get Q, K, V, O projections

SYSTEM_PROMPT = "You are a helpful assistant."

# Example usage with two different queries
msg1 = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {
        "role": "user",
        "content": "What is the capital of India?",
    },
]

msg2 = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {
        "role": "user",
        "content": "What is the capital of France? Answer the question in just one word not more than that.",
    },
]


# Analyze attention patterns for both inputs
data_obj1 = capture_model_attention_internals(
    messages=msg1,
    model=model,
    tokenizer=tokenizer,
    padding=True,
    truncation=True,
    return_attention_mask=True,
    verbose=True,
)
data_obj2 = capture_model_attention_internals(
    messages=msg2,
    model=model,
    tokenizer=tokenizer,
    padding=True,
    truncation=True,
    return_attention_mask=True,
    verbose=True,
)
# %% [markdown]
# ## Store Q, K, V, O projections


Path("../data").mkdir(exist_ok=True)

with open("../data/msg1-qkvo.pkl", "wb") as f:
    pickle.dump(data_obj1, f)

with open("../data/msg2-qkvo.pkl", "wb") as f:
    pickle.dump(data_obj2, f)

# %%
