import torch
from transformers import LlamaForCausalLM
from typing import Dict, List, Tuple, Optional


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
    # Tokenize input
    inputs = tokenizer(input_text, return_tensors="pt").to(device)

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
    layer_outputs = extract_qkvo_outputs(model, input_text, tokenizer, device)

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


# Example usage:
"""
from transformers import LlamaForCausalLM, LlamaTokenizer

# Load model and tokenizer
model_name = "meta-llama/Llama-2-7b-hf"  # or your local model path
model = LlamaForCausalLM.from_pretrained(model_name).to("cuda")
tokenizer = LlamaTokenizer.from_pretrained(model_name)

# Extract attention matrices
input_text = "Hello, how are you today?"

# Option 1: Get outputs organized by layer
layer_outputs = extract_qkvo_outputs(model, input_text, tokenizer)

# Access the outputs for a specific layer
layer_idx = 0
q_proj_output = layer_outputs[layer_idx]['q_proj']
k_proj_output = layer_outputs[layer_idx]['k_proj']
v_proj_output = layer_outputs[layer_idx]['v_proj']
o_proj_output = layer_outputs[layer_idx]['o_proj']

print(f"Q projection shape for layer {layer_idx}: {q_proj_output.shape}")

# Option 2: Get all outputs organized by projection type
all_qkvo = get_all_qkvo(model, input_text, tokenizer)

# Access all Q projections across all layers
all_q_projections = all_qkvo['q']  # This is a list where index corresponds to layer

# Access Q projection for a specific layer
q_layer_5 = all_qkvo['q'][5]

# Example of analyzing all layers' outputs
for layer_idx, q_proj in enumerate(all_qkvo['q']):
    print(f"Layer {layer_idx} Q projection shape: {q_proj.shape}")
    print(f"Layer {layer_idx} Q mean value: {q_proj.mean().item()}")
"""
