import matplotlib.pyplot as plt
import seaborn as sns
import torch


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
