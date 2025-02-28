# %%
import pickle
import torch
import numpy as np
from pathlib import Path
from plot_helpers.plotter import plot_single_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from attention_helpers.gqa import reshape_llama_attention, compute_multihead_attention
from plot_helpers.plotter import visualize_gqa_attention

# %%
# Load the data
path1 = Path("../data/msg1-qkvo.pkl")
path2 = Path("../data/msg2-qkvo.pkl")

with open(path1, "rb") as f:
    data_obj1 = pickle.load(f)

with open(path2, "rb") as f:
    data_obj2 = pickle.load(f)

input_text1 = data_obj1["input_text"]
input_tokens1 = data_obj1["input_tokens"]
print(len(input_tokens1))

input_text2 = data_obj2["input_text"]
input_tokens2 = data_obj2["input_tokens"]
print(len(input_tokens2))

attn_matrices1 = data_obj1["attention_matrices"]
attn_matrices2 = data_obj2["attention_matrices"]

# %%
# Select a layer for visualization
layer_to_visualize = 10

# Extract raw Q, K, V matrices
q1 = attn_matrices1["q"][layer_to_visualize]
k1 = attn_matrices1["k"][layer_to_visualize]
v1 = attn_matrices1["v"][layer_to_visualize]

q2 = attn_matrices2["q"][layer_to_visualize]
k2 = attn_matrices2["k"][layer_to_visualize]
v2 = attn_matrices2["v"][layer_to_visualize]

print(f"Q1 shape: {q1.shape} \nK1 shape: {k1.shape} \nV1 shape: {v1.shape}")

# %% [markdown]
# ## Understanding LLaMA Attention Projection Shapes
#
# In LLaMA models, the raw projections have these shapes after squeezing:
# - Q: [seq_len, hidden_size] where hidden_size = num_heads * head_dim for Q
# - K: [seq_len, kv_dim] where kv_dim = num_heads * head_dim for K/V
#
# For proper KV cache visualization, we need to understand how these matrices are used in the model.

# %%
# First, let's examine the shapes more carefully
q1_2d = torch.squeeze(q1)  # [seq_len, q_dim]
k1_2d = torch.squeeze(k1)  # [seq_len, k_dim]
v1_2d = torch.squeeze(v1)  # [seq_len, v_dim]

print(f"Q1 shape after squeeze: {q1_2d.shape}")
print(f"K1 shape after squeeze: {k1_2d.shape}")
print(f"V1 shape after squeeze: {v1_2d.shape}")

# %% [markdown]
# ## Approach 1: Visualizing Raw Projections (KV Cache Components)
#
# To visualize what's actually stored in the KV cache, we'll look at the raw K and V matrices.

# %%
# Visualize raw K matrix (what would be cached)
plot_single_matrix(
    k1_2d,
    matrix_type="K",
    plot_title=f"Raw Key Matrix (Cached in KV Cache) for Layer {layer_to_visualize}",
    cmap="coolwarm",
)

# Visualize raw V matrix (what would be cached)
plot_single_matrix(
    v1_2d,
    matrix_type="V",
    plot_title=f"Raw Value Matrix (Cached in KV Cache) for Layer {layer_to_visualize}",
    cmap="coolwarm",
)

# %% [markdown]
# ## Approach 2: Proper Reshaping for Multi-Head Attention
#
# The actual attention calculation requires reshaping into multiple heads.


# %%
# For LLaMA, we need to reshape properly

# %%
# Reshape into multi-head format
q1_mh, k1_mh, v1_mh = reshape_llama_attention(q1, k1, v1, verbose=True)
q2_mh, k2_mh, v2_mh = reshape_llama_attention(q2, k2, v2, verbose=False)

# %%

# %% [markdown]
# ## Visualizing Attention Scores for a Single Head

# %% [markdown]
# ## Compute attention scores for all heads
# Initialize dictionaries to store attention scores for all heads


# Compute attention separately for each message
attention_msg1 = compute_multihead_attention(q1_mh, k1_mh, v1_mh)
attention_msg2 = compute_multihead_attention(q2_mh, k2_mh, v2_mh)

# Print shapes for verification
print("\nMessage 1 attention shapes:")
print(
    f"Attention scores shape: {attention_msg1['heads']['q_head_0_kv_head_0']['attention_scores'].shape}"
)
print(
    f"Attention probs shape: {attention_msg1['heads']['q_head_0_kv_head_0']['attention_probs'].shape}"
)
print(
    f"Attention output shape: {attention_msg1['heads']['q_head_0_kv_head_0']['attention_output'].shape}"
)

print("\nMessage 2 attention shapes:")
print(
    f"Attention scores shape: {attention_msg2['heads']['q_head_0_kv_head_0']['attention_scores'].shape}"
)
print(
    f"Attention probs shape: {attention_msg2['heads']['q_head_0_kv_head_0']['attention_probs'].shape}"
)
print(
    f"Attention output shape: {attention_msg2['heads']['q_head_0_kv_head_0']['attention_output'].shape}"
)


# %%


# Visualize attention patterns for both messages
visualize_gqa_attention(attention_msg1, title_prefix="Message 1 -")
visualize_gqa_attention(attention_msg2, title_prefix="Message 2 -")
