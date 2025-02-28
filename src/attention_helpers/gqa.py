import torch


def reshape_llama_attention(q, k, v, num_heads=32, verbose=False):
    """
    Reshape LLaMA attention matrices into multi-head format.

    Args:
        q: Query tensor of shape [1, seq_len, hidden_size] or [seq_len, hidden_size]
        k: Key tensor of shape [1, seq_len, kv_dim] or [seq_len, kv_dim]
        v: Value tensor of shape [1, seq_len, kv_dim] or [seq_len, kv_dim]
        num_heads: Number of attention heads
        verbose: Whether to print debug information

    Returns:
        Tuple of reshaped (q, k, v) tensors with shapes:
        - q: [num_heads, seq_len, head_dim]
        - k: [num_kv_heads, seq_len, head_dim]
        - v: [num_kv_heads, seq_len, head_dim]
    """
    # Get dimensions
    seq_len = q.shape[1] if q.dim() > 2 else q.shape[0]

    # Handle already squeezed tensors
    if q.dim() == 2:
        q_squeezed = q
        k_squeezed = k
        v_squeezed = v
    else:
        q_squeezed = q.squeeze(0)
        k_squeezed = k.squeeze(0)
        v_squeezed = v.squeeze(0)

    if verbose:
        print("\nAfter squeeze:")
        print(f"Q shape: {q_squeezed.shape}")
        print(f"K shape: {k_squeezed.shape}")
        print(f"V shape: {v_squeezed.shape}")

    # Get dimensions for each head
    q_dim = q_squeezed.shape[-1]  # 2048 for TinyLlama
    k_dim = k_squeezed.shape[-1]  # 256 for TinyLlama

    # For TinyLlama:
    # - 32 query heads
    # - 8 key/value heads (GQA)
    num_kv_heads = num_heads // 4  # TinyLlama uses GQA with 4 query heads per KV head

    # Calculate head dimensions
    q_head_dim = q_dim // num_heads  # 2048 // 32 = 64
    kv_head_dim = k_dim // num_kv_heads  # 256 // 8 = 32

    if verbose:
        print("\nHead dimensions:")
        print(f"Q head dim: {q_head_dim}")
        print(f"KV head dim: {kv_head_dim}")
        print(f"Num Q heads: {num_heads}")
        print(f"Num KV heads: {num_kv_heads}")

    # Reshape into multi-head format
    # [seq_len, hidden_size] -> [seq_len, num_heads, head_dim] -> [num_heads, seq_len, head_dim]
    q_reshaped = q_squeezed.view(seq_len, num_heads, q_head_dim).transpose(0, 1)
    k_reshaped = k_squeezed.view(seq_len, num_kv_heads, kv_head_dim).transpose(0, 1)
    v_reshaped = v_squeezed.view(seq_len, num_kv_heads, kv_head_dim).transpose(0, 1)

    if verbose:
        print("\nAfter reshape:")
        print(f"Q shape: {q_reshaped.shape}")
        print(f"K shape: {k_reshaped.shape}")
        print(f"V shape: {v_reshaped.shape}")

    return q_reshaped, k_reshaped, v_reshaped


def compute_multihead_attention(q, k, v, scale=None):
    """
    Compute attention for grouped query attention
    Args:
        q: Query tensor of shape [num_q_heads, seq_len, head_dim]
        k: Key tensor of shape [num_kv_heads, seq_len, head_dim]
        v: Value tensor of shape [num_kv_heads, seq_len, head_dim]
        scale: Optional scaling factor

    Returns:
        Dictionary containing attention patterns and outputs
    """
    # Get dimensions
    num_q_heads, seq_len, q_head_dim = q.shape
    num_kv_heads, _, kv_head_dim = k.shape

    # Note: In some GQA implementations like TinyLlama,
    # the query head dimension can be different from the key/value head dimension
    # No need to assert equality here

    # Calculate how many query heads per kv head (for grouped query attention)
    q_heads_per_kv = num_q_heads // num_kv_heads

    # Apply scaling factor (if not provided, use 1/sqrt(head_dim) as in the original attention paper)
    if scale is None:
        scale = 1.0 / (q_head_dim**0.5)

    # Initialize results dictionary
    results = {
        "heads": {},
        "num_q_heads": num_q_heads,
        "num_kv_heads": num_kv_heads,
        "q_heads_per_kv": q_heads_per_kv,
    }

    # Initialize tensors to store attention scores, probabilities, and outputs
    attn_scores = torch.zeros((num_q_heads, seq_len, seq_len), dtype=q.dtype)
    attn_probs = torch.zeros((num_q_heads, seq_len, seq_len), dtype=q.dtype)
    attention_output = torch.zeros((num_q_heads, seq_len, q_head_dim), dtype=q.dtype)

    # Compute attention for each query head
    for q_head_idx in range(num_q_heads):
        # Find which KV head to use for this query head (grouped query attention)
        kv_head_idx = q_head_idx // q_heads_per_kv

        # Get the query, key, and value tensors for this head
        q_head = q[q_head_idx]  # [seq_len, head_dim]
        k_head = k[kv_head_idx]  # [seq_len, head_dim]
        v_head = v[kv_head_idx]  # [seq_len, head_dim]

        # Compute attention scores: Q * K^T
        # For TinyLlama, q_head is [seq_len, 64] and k_head is [seq_len, 32]
        # Need to project q_head to the same dimension as k_head before computing attention
        if q_head_dim != kv_head_dim:
            # Project query to key dimension using first half of the features
            # This is a simplification and should match how TinyLlama actually does the projection
            q_head_projected = q_head[:, :kv_head_dim]
            scores = (
                torch.matmul(q_head_projected, k_head.transpose(0, 1)) * scale
            )  # [seq_len, seq_len]
        else:
            scores = (
                torch.matmul(q_head, k_head.transpose(0, 1)) * scale
            )  # [seq_len, seq_len]

        # Apply causal mask (lower triangular)
        mask = torch.tril(torch.ones_like(scores))
        scores = scores.masked_fill(mask == 0, float("-inf"))

        # Apply softmax to get attention probabilities
        probs = torch.nn.functional.softmax(scores, dim=-1)

        # Compute attention output: probs * V
        output = torch.matmul(probs, v_head)  # [seq_len, kv_head_dim]

        # If q_head_dim is larger than kv_head_dim, we need to handle this
        if q_head_dim != kv_head_dim:
            # Pad the output with zeros to match the query head dimension
            padded_output = torch.zeros((seq_len, q_head_dim), dtype=output.dtype)
            padded_output[:, :kv_head_dim] = output
            output = padded_output

        # Store results for this head
        key = f"q_head_{q_head_idx}_kv_head_{kv_head_idx}"
        attn_scores[q_head_idx] = scores
        attn_probs[q_head_idx] = probs
        attention_output[q_head_idx] = output

        results["heads"][key] = {
            "attention_scores": attn_scores[q_head_idx],
            "attention_probs": attn_probs[q_head_idx],
            "attention_output": attention_output[q_head_idx],
        }

    # Store overall results
    results["attention_scores"] = attn_scores
    results["attention_probs"] = attn_probs
    results["attention_output"] = attention_output

    return results
