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
    print(f"Verifying cache creation:")
    print(f"Common prefix length: {common_prefix_length}")
    print(f"Total sequence length: {k2_mh.shape[1]}")
    print(f"For prefix tokens: hybrid_k should match k1_mh")
    print(f"For non-prefix tokens: hybrid_k should match k2_mh")

    return hybrid_k, hybrid_v


# %% [markdown]
# ## Additional Visualization Functions
# Define functions for comparing and visualizing KV vectors.


# %%
def plot_kv_vectors_comparison(
    k1_mh, k2_mh, v1_mh, v2_mh, hybrid_k, hybrid_v, token_pos, head_idx=5
):
    """Plot comparison of K,V vectors for a specific token and head."""
    # First figure: K values side by side
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # K values comparison
    k1_vector = k1_mh[head_idx, token_pos, :].cpu().numpy()
    k2_vector = k2_mh[head_idx, token_pos, :].cpu().numpy()
    k_hybrid_vector = hybrid_k[head_idx, token_pos, :].cpu().numpy()

    # Original K comparison
    ax1.plot(k1_vector, label="K from input1", color="blue", linewidth=2)
    ax1.plot(
        k2_vector, label="K from input2", linestyle="--", color="orange", linewidth=2
    )
    ax1.set_title(f"K values comparison\ntoken {token_pos} (Head {head_idx})")
    ax1.set_xlabel("Dimension")
    ax1.set_ylabel("Value")
    ax1.legend()

    # K hybrid check
    ax2.plot(k1_vector, label="K from input1", color="blue", linewidth=2)
    ax2.plot(k_hybrid_vector, label="K hybrid", color="red", linewidth=2)
    ax2.set_title(f"K values hybrid check\ntoken {token_pos} (Head {head_idx})")
    ax2.set_xlabel("Dimension")
    ax2.legend()

    plt.tight_layout()

    # Second figure: V values side by side
    fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(15, 5))

    # V values comparison
    v1_vector = v1_mh[head_idx, token_pos, :].cpu().numpy()
    v2_vector = v2_mh[head_idx, token_pos, :].cpu().numpy()
    v_hybrid_vector = hybrid_v[head_idx, token_pos, :].cpu().numpy()

    # Original V comparison
    ax3.plot(v1_vector, label="V from input1", color="blue", linewidth=2)
    ax3.plot(
        v2_vector, label="V from input2", linestyle="--", color="orange", linewidth=2
    )
    ax3.set_title(f"V values comparison\ntoken {token_pos} (Head {head_idx})")
    ax3.set_xlabel("Dimension")
    ax3.set_ylabel("Value")
    ax3.legend()

    # V hybrid check
    ax4.plot(v1_vector, label="V from input1", color="blue", linewidth=2)
    ax4.plot(v_hybrid_vector, label="V hybrid", color="red", linewidth=2)
    ax4.set_title(f"V values hybrid check\ntoken {token_pos} (Head {head_idx})")
    ax4.set_xlabel("Dimension")
    ax4.legend()

    plt.tight_layout()
    return fig1, fig2


# %% [markdown]
# ## Visualization Functions
# Define functions for plotting attention matrices and KV vector comparisons.


# %%
def _plot_attention_row(
    axes_row,
    head_idx,
    original_attn,
    cached_attn,
    diff_attn,
    common_prefix_length,
    show_xlabel,
):
    """Plot a row of attention visualizations for a single head.

    Args:
        axes_row: Row of matplotlib axes to plot on
        head_idx: Index of attention head being visualized
        original_attn: Original attention matrix
        cached_attn: Attention matrix with KV caching
        diff_attn: Difference between original and cached attention
        common_prefix_length: Length of common prefix between inputs
        show_xlabel: Whether to show x-axis labels
    """
    # Plot original attention
    im0 = axes_row[0].imshow(original_attn, cmap="coolwarm")
    axes_row[0].set_title(f"Head {head_idx}: Original Attention")
    if show_xlabel:
        axes_row[0].set_xlabel("Key Position")
    axes_row[0].set_ylabel("Query Position")
    plt.colorbar(im0, ax=axes_row[0], fraction=0.046, pad=0.04)

    # Plot cached attention
    im1 = axes_row[1].imshow(cached_attn, cmap="coolwarm")
    axes_row[1].set_title(f"Head {head_idx}: With KV Caching")
    if show_xlabel:
        axes_row[1].set_xlabel("Key Position")
    plt.colorbar(im1, ax=axes_row[1], fraction=0.046, pad=0.04)

    # Plot difference
    im2 = axes_row[2].imshow(diff_attn, cmap="hot", vmin=0, vmax=0.001)
    axes_row[2].set_title(f"Head {head_idx}: Difference")
    if show_xlabel:
        axes_row[2].set_xlabel("Key Position")
    plt.colorbar(im2, ax=axes_row[2], fraction=0.046, pad=0.04)

    # Add common prefix box if applicable
    if common_prefix_length > 0:
        for j in range(3):
            rect = plt.Rectangle(
                (0, 0),
                common_prefix_length,
                common_prefix_length,
                fill=False,
                edgecolor="white",
                linewidth=2,
            )
            axes_row[j].add_patch(rect)


def plot_attention_matrices(
    original_attention, cached_attention, common_prefix_length, max_tokens
):
    """Plot attention matrices comparison for multiple heads."""
    heads_to_viz = [0, 1, 2, 3]
    fig, axes = plt.subplots(len(heads_to_viz), 3, figsize=(18, 4 * len(heads_to_viz)))

    for i, head_idx in enumerate(heads_to_viz):
        original_attn = original_attention["heads"][
            f"q_head_{head_idx}_kv_head_{head_idx//4}"
        ]["attention_probs"]
        cached_attn = cached_attention["heads"][
            f"q_head_{head_idx}_kv_head_{head_idx//4}"
        ]["attention_probs"]

        original_attn_viz = original_attn[:max_tokens, :max_tokens].cpu().numpy()
        cached_attn_viz = cached_attn[:max_tokens, :max_tokens].cpu().numpy()
        diff_attn = np.abs(original_attn_viz - cached_attn_viz)

        _plot_attention_row(
            axes[i],
            head_idx,
            original_attn_viz,
            cached_attn_viz,
            diff_attn,
            common_prefix_length,
            i == len(heads_to_viz) - 1,
        )

    plt.tight_layout()
    return diff_attn


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
def get_axis_limits(k1_mh, k2_mh, v1_mh, v2_mh, hybrid_k, hybrid_v, head_idx, dim):
    """Get common axis limits for all plots."""
    # Get all K values
    k1_vals = k1_mh[head_idx, :, dim].cpu().numpy()
    k2_vals = k2_mh[head_idx, :, dim].cpu().numpy()
    k_hybrid_vals = hybrid_k[head_idx, : k2_mh.shape[1], dim].cpu().numpy()

    # Get all V values
    v1_vals = v1_mh[head_idx, :, dim].cpu().numpy()
    v2_vals = v2_mh[head_idx, :, dim].cpu().numpy()
    v_hybrid_vals = hybrid_v[head_idx, : v2_mh.shape[1], dim].cpu().numpy()

    # Calculate limits with some padding
    k_min = min(k1_vals.min(), k2_vals.min(), k_hybrid_vals.min())
    k_max = max(k1_vals.max(), k2_vals.max(), k_hybrid_vals.max())
    v_min = min(v1_vals.min(), v2_vals.min(), v_hybrid_vals.min())
    v_max = max(v1_vals.max(), v2_vals.max(), v_hybrid_vals.max())

    # Add 10% padding
    k_range = k_max - k_min
    v_range = v_max - v_min
    k_limits = (k_min - 0.1 * k_range, k_max + 0.1 * k_range)
    v_limits = (v_min - 0.1 * v_range, v_max + 0.1 * v_range)

    return k_limits, v_limits


def plot_kv_cache_verification(
    k_mh,
    v_mh,
    common_prefix_length,
    k_limits,
    v_limits,
    max_seq_len,
    head_idx=5,
    input_name="input1",
):
    """Plot K and V values with cached and non-cached regions clearly marked."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Get full sequence K and V vectors for a single dimension
    dim = k_mh.shape[-1] // 2  # Use middle dimension for clearer visualization
    k_vector = k_mh[head_idx, :, dim].cpu().numpy()  # [seq_len]
    v_vector = v_mh[head_idx, :, dim].cpu().numpy()

    # Create x-axis positions
    positions = np.arange(len(k_vector))

    # Plot K values
    ax1.plot(positions, k_vector, label="K values", color="blue", linewidth=2)
    ax1.set_ylim(k_limits)
    ax1.set_xlim(0, max_seq_len)  # Set consistent x-axis limit

    # Add background shading for cached region
    ax1.axvspan(
        0, common_prefix_length - 1, color="yellow", alpha=0.2, label="Cached region"
    )
    ax1.axvline(x=common_prefix_length - 1, color="black", linestyle="--", alpha=0.5)
    ax1.set_title(f"K values - Head {head_idx}, Dimension {dim}")
    ax1.set_xlabel("Token Position")
    ax1.set_ylabel("Value")
    ax1.legend()

    # Plot V values
    ax2.plot(positions, v_vector, label="V values", color="blue", linewidth=2)
    ax2.set_ylim(v_limits)
    ax2.set_xlim(0, max_seq_len)  # Set consistent x-axis limit

    # Add background shading for cached region
    ax2.axvspan(
        0, common_prefix_length - 1, color="yellow", alpha=0.2, label="Cached region"
    )
    ax2.axvline(x=common_prefix_length - 1, color="black", linestyle="--", alpha=0.5)
    ax2.set_title(f"V values - Head {head_idx}, Dimension {dim}")
    ax2.set_xlabel("Token Position")
    ax2.legend()

    plt.suptitle(f"{input_name} KV Values\nYellow region shows what gets cached")
    plt.tight_layout()
    return fig


def plot_hybrid_verification(
    k_mh,
    v_mh,
    hybrid_k,
    hybrid_v,
    common_prefix_length,
    k_limits,
    v_limits,
    max_seq_len,
    head_idx=5,
):
    """Plot K and V values with hybrid cache verification for input2."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Get full sequence K and V vectors for a single dimension
    dim = k_mh.shape[-1] // 2
    k_vector = k_mh[head_idx, :, dim].cpu().numpy()
    k_hybrid_vector = hybrid_k[head_idx, : k_vector.shape[0], dim].cpu().numpy()
    v_vector = v_mh[head_idx, :, dim].cpu().numpy()
    v_hybrid_vector = hybrid_v[head_idx, : v_vector.shape[0], dim].cpu().numpy()

    positions = np.arange(len(k_vector))

    # Plot K values
    ax1.plot(positions, k_vector, label="K from input2", color="blue", linewidth=2)
    ax1.plot(
        positions,
        k_hybrid_vector,
        label="K hybrid",
        color="red",
        linewidth=2,
        linestyle="--",
    )
    ax1.set_ylim(k_limits)
    ax1.set_xlim(0, max_seq_len)  # Set consistent x-axis limit

    ax1.axvspan(
        0, common_prefix_length - 1, color="yellow", alpha=0.2, label="Cached region"
    )
    ax1.axvline(x=common_prefix_length - 1, color="black", linestyle="--", alpha=0.5)
    ax1.set_title(f"K values - Head {head_idx}, Dimension {dim}")
    ax1.set_xlabel("Token Position")
    ax1.set_ylabel("Value")
    ax1.legend()

    # Plot V values
    ax2.plot(positions, v_vector, label="V from input2", color="blue", linewidth=2)
    ax2.plot(
        positions,
        v_hybrid_vector,
        label="V hybrid",
        color="red",
        linewidth=2,
        linestyle="--",
    )
    ax2.set_ylim(v_limits)
    ax2.set_xlim(0, max_seq_len)  # Set consistent x-axis limit

    ax2.axvspan(
        0, common_prefix_length - 1, color="yellow", alpha=0.2, label="Cached region"
    )
    ax2.axvline(x=common_prefix_length - 1, color="black", linestyle="--", alpha=0.5)
    ax2.set_title(f"V values - Head {head_idx}, Dimension {dim}")
    ax2.set_xlabel("Token Position")
    ax2.legend()

    plt.suptitle(
        "Input 2 KV Cache Verification\nYellow region shows cached tokens from Input 1"
    )
    plt.tight_layout()
    return fig


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
