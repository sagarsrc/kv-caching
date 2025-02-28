import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np


# Function to visualize attention patterns with GQA structure
def visualize_gqa_attention(
    attention_data, num_kv_heads=8, num_q_heads=32, title_prefix=""
):
    """
    Visualize attention patterns for a model using Grouped Query Attention (GQA).

    Args:
        attention_data: Dictionary containing attention patterns from compute_multihead_attention
        num_kv_heads: Number of key/value heads
        num_q_heads: Number of query heads
        title_prefix: Prefix for plot titles
    """
    # Create a figure with subplots for each KV head
    fig, axes = plt.subplots(2, 4, figsize=(20, 10), dpi=300)
    axes = axes.flatten()

    # Set a global title for the subplots
    fig.suptitle(
        f"{title_prefix} Average Attention Patterns Across KV Heads", fontsize=20
    )

    # For each KV head, plot the average attention pattern across all query heads that share it
    for kv_head_idx in range(num_kv_heads):
        # Find all query heads that share this KV head
        q_heads_for_this_kv = [
            q_head_idx
            for q_head_idx in range(num_q_heads)
            if q_head_idx // (num_q_heads // num_kv_heads) == kv_head_idx
        ]

        # Get the attention patterns for these query heads
        attn_patterns = [
            attention_data["heads"][f"q_head_{q_head_idx}_kv_head_{kv_head_idx}"][
                "attention_probs"
            ]
            for q_head_idx in q_heads_for_this_kv
        ]

        # Average the attention patterns
        avg_attn_pattern = torch.stack(attn_patterns).mean(dim=0)

        # Plot the average attention pattern
        ax = axes[kv_head_idx]
        sns.heatmap(
            avg_attn_pattern.cpu().detach().numpy(),
            cmap="hot",
            ax=ax,
            cbar=True,
        )
        ax.set_title(
            f"KV Head {kv_head_idx} (avg of {len(q_heads_for_this_kv)} Q heads)"
        )
        ax.set_xlabel("Key Position")
        ax.set_ylabel("Query Position")

    plt.tight_layout()
    plt.show()


def plot_single_matrix(
    matrix,
    layer_idx=None,
    matrix_type="",
    plot_title=None,
    cmap="viridis",
    xlabel="Hidden Dimension",
    ylabel="Token",
    tokens=None,
):
    """
    Visualize a single attention matrix as a heatmap.

    Args:
        matrix (tensor): The matrix to visualize
        layer_idx (int, optional): Layer index if applicable
        matrix_type (str): Type of matrix (e.g., "Q", "K", "QK")
        plot_title (str, optional): Custom title for the plot. If None, a default title is generated.
        cmap (str): Colormap to use for the heatmap
        xlabel (str): Label for the x-axis
        ylabel (str): Label for the y-axis
    """
    # Convert tensor to numpy array if it isn't already
    if hasattr(matrix, "cpu"):
        matrix = matrix.cpu().detach().numpy()

    # Reshape to 2D if needed
    if matrix.ndim > 2:
        orig_shape = matrix.shape
        matrix = matrix.reshape(-1, matrix.shape[-1])
        shape_info = f" (reshaped from {orig_shape})"
    else:
        shape_info = ""

    # Create default title if none provided
    if plot_title is None:
        title_parts = []
        if matrix_type:
            title_parts.append(f"{matrix_type} Matrix")
        if layer_idx is not None:
            title_parts.append(f"Layer {layer_idx}")
        plot_title = " - ".join(title_parts) if title_parts else "Matrix Visualization"

    # Plot heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(matrix, cmap=cmap, center=0 if cmap == "coolwarm" else None)
    plt.title(f"{plot_title}{shape_info}")

    if tokens is not None:
        # Create ytick labels combining index and token
        new_line_print_handle = lambda x: x.replace("\n", "\\n")

        ytick_labels = [
            f"{new_line_print_handle(token)} :: {i:02d}"
            for i, token in enumerate(tokens)
        ]

        plt.yticks(
            np.arange(len(tokens)) + 0.5,  # Center the labels
            ytick_labels,
            rotation=0,
            ha="right",
            va="center",
            fontsize=8,
        )

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.show()


def plot_matrix_comparison(
    matrix1,
    matrix2,
    plot_title="Matrix Comparison",
    cmap="coolwarm",
    xlabel="Hidden Dimension",
    ylabel="Token",
    plot_difference=True,
):
    """
    Plot comparison between two matrices (can be any matrices, not just Q).

    Args:
        matrix1 (tensor): First matrix
        matrix2 (tensor): Second matrix
        plot_title (str): Title for the plot
        cmap (str): Colormap to use for the
        xlabel (str): Label for the x-axis
        ylabel (str): Label for the y-axis
        plot_difference (bool): Whether to plot the difference between the two matrices
    """
    # Convert tensors to numpy arrays if they aren't already
    if hasattr(matrix1, "cpu"):
        matrix1 = matrix1.cpu().detach().numpy()
    if hasattr(matrix2, "cpu"):
        matrix2 = matrix2.cpu().detach().numpy()

    # Reshape to 2D if needed
    if matrix1.ndim > 2:
        matrix1 = matrix1.reshape(-1, matrix1.shape[-1])
    if matrix2.ndim > 2:
        matrix2 = matrix2.reshape(-1, matrix2.shape[-1])

    # Ensure matrices have compatible shapes for subtraction
    min_rows = min(matrix1.shape[0], matrix2.shape[0])
    min_cols = min(matrix1.shape[1], matrix2.shape[1])

    matrix1 = matrix1[:min_rows, :min_cols]
    matrix2 = matrix2[:min_rows, :min_cols]

    # Calculate difference
    difference_matrix = matrix1 - matrix2

    if plot_difference:
        # Plot heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(difference_matrix, cmap=cmap, center=0)
        plt.title(plot_title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.show()

    return difference_matrix


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
    ax1.legend(loc="lower right")

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
    ax2.legend(loc="lower right")

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
    ax1.set_xlim(0, max_seq_len)

    ax1.axvspan(
        0, common_prefix_length - 1, color="yellow", alpha=0.2, label="Cached region"
    )
    ax1.axvline(x=common_prefix_length - 1, color="black", linestyle="--", alpha=0.5)
    ax1.set_title(f"K values - Head {head_idx}, Dimension {dim}")
    ax1.set_xlabel("Token Position")
    ax1.set_ylabel("Value")
    ax1.legend(loc="lower right")

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
    ax2.set_xlim(0, max_seq_len)

    ax2.axvspan(
        0, common_prefix_length - 1, color="yellow", alpha=0.2, label="Cached region"
    )
    ax2.axvline(x=common_prefix_length - 1, color="black", linestyle="--", alpha=0.5)
    ax2.set_title(f"V values - Head {head_idx}, Dimension {dim}")
    ax2.set_xlabel("Token Position")
    ax2.legend(loc="lower right")

    plt.suptitle(
        "Input 2 KV Cache Verification\nYellow region shows cached tokens from Input 1"
    )
    plt.tight_layout()
    return fig
