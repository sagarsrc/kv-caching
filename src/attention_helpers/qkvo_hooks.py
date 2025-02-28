import torch
from transformers import LlamaForCausalLM
from typing import Dict, List, Tuple, Optional, Any


class LlamaAttentionExtractor:
    """
    A class to extract Q, K, V, and O matrices from a LlamaForCausalLM model.
    """

    def __init__(self, model: LlamaForCausalLM):
        self.model = model
        self.attention_outputs = {}
        self.hooks = []
        self._setup_hooks()

    def _setup_hooks(self):
        """Set up forward hooks on all attention layers"""
        for layer_idx, layer in enumerate(self.model.model.layers):
            # Hook for q_proj
            self.hooks.append(
                layer.self_attn.q_proj.register_forward_hook(
                    self._create_hook(layer_idx, "q_proj")
                )
            )

            # Hook for k_proj
            self.hooks.append(
                layer.self_attn.k_proj.register_forward_hook(
                    self._create_hook(layer_idx, "k_proj")
                )
            )

            # Hook for v_proj
            self.hooks.append(
                layer.self_attn.v_proj.register_forward_hook(
                    self._create_hook(layer_idx, "v_proj")
                )
            )

            # Hook for o_proj
            self.hooks.append(
                layer.self_attn.o_proj.register_forward_hook(
                    self._create_hook(layer_idx, "o_proj")
                )
            )

    def _create_hook(self, layer_idx: int, proj_name: str):
        """Create a hook function for a specific layer and projection"""

        def hook(module, input_tensor, output_tensor):
            if layer_idx not in self.attention_outputs:
                self.attention_outputs[layer_idx] = {}
            self.attention_outputs[layer_idx][
                proj_name
            ] = output_tensor.detach().clone()

        return hook

    def remove_hooks(self):
        """Remove all hooks to free memory"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def extract_attention_matrices(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        layer_indices: Optional[List[int]] = None,
    ) -> Dict[int, Dict[str, torch.Tensor]]:
        """
        Extract Q, K, V, O matrices for specified layers.

        Args:
            input_ids: Input token IDs
            attention_mask: Optional attention mask
            layer_indices: Optional list of layer indices to extract. If None, extracts from all layers.

        Returns:
            Dictionary mapping layer indices to dictionaries containing the Q, K, V, O matrices
        """
        # Clear previous outputs
        self.attention_outputs = {}

        # Forward pass
        with torch.no_grad():
            self.model(input_ids, attention_mask=attention_mask)

        # Filter by requested layers if specified
        if layer_indices is not None:
            filtered_outputs = {
                idx: self.attention_outputs[idx]
                for idx in layer_indices
                if idx in self.attention_outputs
            }
            return filtered_outputs

        return self.attention_outputs

    def __del__(self):
        """Clean up hooks when object is deleted"""
        self.remove_hooks()


def extract_qkvo_outputs(
    model: LlamaForCausalLM,
    input_text: str,
    tokenizer,
    padding: bool = False,
    truncation: bool = True,
    return_attention_mask: bool = True,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    layer_indices: Optional[List[int]] = None,
) -> Dict[int, Dict[str, torch.Tensor]]:
    """
    Extract Q, K, V, O outputs from a LlamaForCausalLM model for a given input text.

    Args:
        model: LlamaForCausalLM model
        input_text: Input text to process
        tokenizer: Tokenizer to convert text to tokens
        device: Device to run the model on
        layer_indices: Optional list of layer indices to extract from

    Returns:
        Dictionary mapping layer indices to dictionaries containing the Q, K, V, O matrices
    """
    # Tokenize input with proper padding
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        padding=padding,
        truncation=truncation,
        return_attention_mask=return_attention_mask,
    ).to(device)

    # Set up extractor
    extractor = LlamaAttentionExtractor(model)

    # Extract attention matrices
    attention_outputs = extractor.extract_attention_matrices(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        layer_indices=layer_indices,
    )

    # Clean up
    extractor.remove_hooks()

    return attention_outputs


def get_all_qkvo(
    model: LlamaForCausalLM,
    input_text: str,
    tokenizer,
    padding: bool = False,
    truncation: bool = True,
    return_attention_mask: bool = True,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict[str, List[torch.Tensor]]:
    """
    Extract all Q, K, V, O outputs across all layers and organize them by projection type.

    Args:
        model: LlamaForCausalLM model
        input_text: Input text to process
        tokenizer: Tokenizer to convert text to tokens
        device: Device to run the model on

    Returns:
        Dictionary with keys 'q', 'k', 'v', 'o', each containing a list of tensors
        where the index in the list corresponds to the layer index
    """
    # Get outputs organized by layer
    layer_outputs = extract_qkvo_outputs(
        model=model,
        input_text=input_text,
        tokenizer=tokenizer,
        padding=padding,
        truncation=truncation,
        return_attention_mask=return_attention_mask,
        device=device,
    )

    # Reorganize by projection type
    num_layers = len(model.model.layers)
    all_qkvo = {
        "q": [None] * num_layers,
        "k": [None] * num_layers,
        "v": [None] * num_layers,
        "o": [None] * num_layers,
    }

    # Fill in the outputs
    for layer_idx, projections in layer_outputs.items():
        all_qkvo["q"][layer_idx] = projections["q_proj"]
        all_qkvo["k"][layer_idx] = projections["k_proj"]
        all_qkvo["v"][layer_idx] = projections["v_proj"]
        all_qkvo["o"][layer_idx] = projections["o_proj"]

    return all_qkvo


def capture_model_attention_internals(
    messages,
    model,
    tokenizer,
    analysis_name="",
    padding=True,
    truncation=True,
    return_attention_mask=True,
    verbose=True,
) -> Dict[str, Any]:
    """
    Captures and returns the internal attention mechanism components (Q, K, V, O projections)
    for a given input message across all model layers.

    Args:
        messages (list): List of message dictionaries with 'role' and 'content'
        model: The language model to use
        tokenizer: The tokenizer to use
        padding (bool): Whether to pad sequences
        truncation (bool): Whether to truncate sequences
        return_attention_mask (bool): Whether to return attention mask

    Returns:
        dict: Dictionary containing all Q, K, V, O projections across layers and the input text
    """
    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        padding=padding,
        truncation=truncation,
    )
    input_tokens = tokenizer.encode(
        input_text,
        padding=padding,
        truncation=truncation,
        return_tensors="pt",
    )
    attention_matrices = get_all_qkvo(
        model=model,
        input_text=input_text,
        tokenizer=tokenizer,
        padding=padding,
        truncation=truncation,
    )

    # Print shapes for first and last layer as example
    first_layer = 0

    if verbose:
        print("Attention layer shapes:")
        print(f"Q projection: {attention_matrices['q'][first_layer].shape}")
        print(f"K projection: {attention_matrices['k'][first_layer].shape}")
        print(f"V projection: {attention_matrices['v'][first_layer].shape}")
        print(f"O projection: {attention_matrices['o'][first_layer].shape}")

    return {
        "input_text": input_text,
        "input_tokens": input_tokens,
        "decoded_input_tokens": [tokenizer.decode([t]) for t in input_tokens[0]],
        "attention_matrices": attention_matrices,
    }
