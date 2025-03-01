# %% [markdown]
# # KV Cache Visualization Demo
# This notebook demonstrates the visualization of attention patterns with and without KV caching.
# We'll analyze how caching affects attention computation and visualize the differences.

# %% [markdown]
# ## Setup and Imports
# First, let's import the necessary libraries and set up our environment.

# %%
import pickle
import numpy as np
from pathlib import Path
from attention_helpers.gqa import reshape_llama_attention, compute_multihead_attention
import matplotlib.pyplot as plt
from plot_helpers.plotter import (
    plot_attention_matrices,
    get_axis_limits,
    plot_kv_cache_verification,
    plot_hybrid_verification,
)

# %% [markdown]
# ## Data Loading
# Load two different message data files containing attention matrices and token information.

# %%
# Load the data
path1 = Path("../data/msg1-qkvo.pkl")
path2 = Path("../data/msg2-qkvo.pkl")

with open(path1, "rb") as f:
    data_obj1 = pickle.load(f)

with open(path2, "rb") as f:
    data_obj2 = pickle.load(f)

# Print basic information about the data
print("Message 1 tokens:", len(data_obj1["input_tokens"][0]))
print("Message 2 tokens:", len(data_obj2["input_tokens"][0]))

# %% [markdown]
# ## Helper Functions
# Define utility functions for finding common prefixes and creating hybrid KV caches.


# %%
def find_common_prefix_length(tokens1, tokens2):
    """Find the length of common prefix between two token sequences."""
    common_prefix_length = 0
    for i in range(min(len(tokens1), len(tokens2))):
        if tokens1[i] == tokens2[i]:
            common_prefix_length += 1
        else:
            break
    return common_prefix_length


def create_hybrid_kv_cache(q2_mh, k1_mh, k2_mh, v1_mh, v2_mh, common_prefix_length):
    """Create hybrid K,V matrices using cached values for common prefix."""
    if common_prefix_length == 0:
        return k2_mh, v2_mh

    hybrid_k = k2_mh.clone()
    hybrid_v = v2_mh.clone()

    # Only copy the common prefix!
    hybrid_k[:, :common_prefix_length, :] = k1_mh[:, :common_prefix_length, :]
    hybrid_v[:, :common_prefix_length, :] = v1_mh[:, :common_prefix_length, :]

    # Let's add some verification
    print("Verifying cache creation:")
    print(f"Common prefix length: {common_prefix_length}")
    print(f"Total sequence length: {k2_mh.shape[1]}")
    print("For prefix tokens: hybrid_k should match k1_mh")
    print("For non-prefix tokens: hybrid_k should match k2_mh")

    return hybrid_k, hybrid_v


# %% [markdown]
# ## Analysis Setup
# Prepare the data for analysis by extracting relevant matrices and computing common prefix.

# %%
# Select a layer for visualization
layer_to_use = 21

# Extract matrices for the specified layer
q1, k1, v1 = [
    data_obj1["attention_matrices"][key][layer_to_use] for key in ["q", "k", "v"]
]
q2, k2, v2 = [
    data_obj2["attention_matrices"][key][layer_to_use] for key in ["q", "k", "v"]
]

# Convert input tokens and find common prefix
tokens1 = data_obj1["input_tokens"][0].tolist()
tokens2 = data_obj2["input_tokens"][0].tolist()
common_prefix_length = find_common_prefix_length(tokens1, tokens2)
print(f"Common prefix length: {common_prefix_length} tokens")

# %% [markdown]
# ## Compute Attention
# Process the attention matrices and create hybrid cached version.

# %%
# Reshape into multi-head format
q1_mh, k1_mh, v1_mh = reshape_llama_attention(q1, k1, v1, verbose=False)
q2_mh, k2_mh, v2_mh = reshape_llama_attention(q2, k2, v2, verbose=False)

# Create hybrid cached version and compute attention
hybrid_k, hybrid_v = create_hybrid_kv_cache(
    q2_mh, k1_mh, k2_mh, v1_mh, v2_mh, common_prefix_length
)
original_attention = compute_multihead_attention(q2_mh, k2_mh, v2_mh)
cached_attention = compute_multihead_attention(q2_mh, hybrid_k, hybrid_v)

# %% [markdown]
# ## Visualize Results
# Plot attention patterns and compute statistics.

# %%
# Visualize results
max_tokens = min(15, len(tokens2))
diff_attn = plot_attention_matrices(
    original_attention, cached_attention, common_prefix_length, max_tokens
)

# Print statistics and savings
print(f"Maximum difference in attention values: {np.max(diff_attn):.8f}")
print(f"Mean difference: {np.mean(diff_attn):.8f}")

if common_prefix_length > 0:
    token_savings = common_prefix_length / len(tokens2)
    print(
        f"By caching KV for the common prefix, you save {token_savings:.1%} of KV computations"
    )

# %% [markdown]
# ## KV Vector Analysis
# Compare individual K,V vectors for tokens in the common prefix.


# %%
if common_prefix_length > 0:
    # Print some debug info
    print(f"Common prefix length: {common_prefix_length}")
    print(f"Input 1 sequence length: {k1_mh.shape[1]}")
    print(f"Input 2 sequence length: {k2_mh.shape[1]}")

    # Get common axis limits
    head_idx = 5
    k_limits, v_limits = get_axis_limits(
        k1_mh, k2_mh, v1_mh, v2_mh, hybrid_k, hybrid_v, head_idx, k1_mh.shape[-1] // 2
    )

    # Calculate maximum sequence length
    max_seq_len = max(k1_mh.shape[1], k2_mh.shape[1])

    # Plot input1's KV values (what gets cached)
    fig1 = plot_kv_cache_verification(
        k1_mh,
        v1_mh,
        common_prefix_length,
        k_limits,
        v_limits,
        max_seq_len,
        input_name="input1",
    )
    plt.show()

    # Plot input2's verification with hybrid cache
    fig2 = plot_hybrid_verification(
        k2_mh,
        v2_mh,
        hybrid_k,
        hybrid_v,
        common_prefix_length,
        k_limits,
        v_limits,
        max_seq_len,
    )
    plt.show()
