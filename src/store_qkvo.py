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

# custom utils
from attention_helpers.qkvo_hooks import get_all_qkvo

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
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, padding=True)

    # Encode the prompt
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)

    # Generate
    outputs = model.generate(
        inputs.input_ids,
        max_new_tokens=max_tokens,
        do_sample=(temperature > 0),
        pad_token_id=tokenizer.eos_token_id,
    )

    # Decode and return the generated text, keeping special tokens
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=False)
    if verbose:
        for token_num, token in enumerate(decoded_output.split()):
            print(f"Token {token_num}: {token}")

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
print(f"\nGenerated Text:\n{generated_text}")


# %% [markdown]
# ## Get Q, K, V, O projections


def analyze_attention_patterns(
    messages,
    model,
    tokenizer,
    analysis_name="",
    padding=True,
    truncation=True,
    return_attention_mask=True,
) -> Dict[str, Any]:
    """
    Analyze attention patterns for a given input message.

    Args:
        messages (list): List of message dictionaries with 'role' and 'content'
        model: The language model to use
        tokenizer: The tokenizer to use
        analysis_name (str): Name identifier for the analysis

    Returns:
        dict: Dictionary containing all Q, K, V, O projections across layers and the input text
    """
    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        padding=True,
    )
    input_tokens = tokenizer.encode(
        input_text,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    attention_matrices = get_all_qkvo(
        model=model,
        input_text=input_text,
        tokenizer=tokenizer,
        padding=True,
        truncation=True,
    )

    # Print shapes for first and last layer as example
    first_layer = 0

    print(f"\nAttention Analysis for {analysis_name}")
    print("Attention layer shapes:")
    print(f"Q projection: {attention_matrices['q'][first_layer].shape}")
    print(f"K projection: {attention_matrices['k'][first_layer].shape}")
    print(f"V projection: {attention_matrices['v'][first_layer].shape}")
    print(f"O projection: {attention_matrices['o'][first_layer].shape}")

    return {
        "input_text": input_text,
        "input_tokens": input_tokens,
        "attention_matrices": attention_matrices,
    }


# Example usage with two different queries
msg1 = [
    {"role": "system", "content": "You are a friendly chat assistant"},
    {
        "role": "user",
        "content": "What is the capital of India?",
    },
]

msg2 = [
    {"role": "system", "content": "You are a friendly chat assistant"},
    {
        "role": "user",
        "content": "What is the capital of France? Answer the question in just one word not more than that.",
    },
]

# Print tokenization information for both messages
print("\nTokenization Information:")
for i, msg in enumerate([msg1, msg2], 1):
    text = tokenizer.apply_chat_template(msg, tokenize=False, padding=False)
    tokens = tokenizer.encode(text, padding=False, truncation=True, return_tensors="pt")
    print(f"\nMessage {i}:")
    print(f"Text length: {len(text)}")
    print(f"Token sequence length: {tokens.shape[1]}")
    print(f"Tokens: {tokens.tolist()[0]}")
    print(f"Decoded tokens: {[tokenizer.decode([t]) for t in tokens[0]]}")


# Analyze attention patterns for both inputs
data_obj1 = analyze_attention_patterns(
    messages=msg1,
    model=model,
    tokenizer=tokenizer,
    analysis_name="Query 1",
    padding=True,
    truncation=True,
    return_attention_mask=True,
)
data_obj2 = analyze_attention_patterns(
    messages=msg2,
    model=model,
    tokenizer=tokenizer,
    analysis_name="Query 2",
    padding=True,
    truncation=True,
    return_attention_mask=True,
)
# %% [markdown]
# ## Store Q, K, V, O projections


Path("../data").mkdir(exist_ok=True)

with open("../data/msg1-qkvo.pkl", "wb") as f:
    pickle.dump(data_obj1, f)

with open("../data/msg2-qkvo.pkl", "wb") as f:
    pickle.dump(data_obj2, f)

# %%
