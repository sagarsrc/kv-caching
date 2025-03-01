# KV-Caching Visualization and Analysis

This repository provides a stupidly simple demonstration and visualization of Key-Value (KV) caching in transformer models, specifically using TinyLlama with Grouped Query Attention (GQA). It helps understand how KV caching works and its impact on attention computation in transformer models.

## Repository Structure

```
.
├── src/
│   ├── get_attention_projections.py   # Extract Q,K,V,O projections from TinyLlama
│   ├── visualize_attention.py         # Visualize attention patterns
│   ├── visualize_kv_caching.py        # Demonstrate KV caching effects
│   ├── attention_helpers/             # Helper functions for attention computation
│   └── plot_helpers/                  # Utilities for visualization
├── notebooks/
│   ├── kv_cache_demo.ipynb           # Main demo notebook
│   └── kv_cache_demo.py              # Python version of the demo
```

## Key Components

### Source Code (src/)

The source code is organized into three main components:

1. **Attention Projections ([get_attention_projections.py](src/get_attention_projections.py))**
   - Loads TinyLlama model
   - Extracts Query (Q), Key (K), Value (V), and Output (O) projections
   - Captures internal attention states during inference

2. **Attention Visualization ([visualize_attention.py](src/visualize_attention.py))**
   - Visualizes raw attention patterns
   - Demonstrates Grouped Query Attention (GQA) mechanics
   - Shows attention distribution across heads

3. **KV Caching Visualization ([visualize_kv_caching.py](src/visualize_kv_caching.py))**
   - Demonstrates how KV caching works
   - Visualizes attention patterns with and without caching
   - Shows cache reuse across multiple inputs

### Demo Notebook (notebooks/)

The `kv_cache_demo.ipynb` (and its Python equivalent) serves as a one-stop shop for the entire demonstration. It:
- All the things that scripts do but in a jupyter notebook!

## Technical Details

### Model Architecture
- Uses **TinyLlama** (1.1B parameters) as the base model
- Implements **Grouped Query Attention (GQA)** for efficient attention computation
- Demonstrates practical KV caching implementation

### Approach

The repository demonstrates KV caching through the following approach:

1. **Cache Creation**
   - Computes and stores Key (K) and Value (V) vectors for input sequences
   - Shows how these cached vectors can be reused

2. **Cache Utilization**
   - Demonstrates how subsequent tokens can reuse cached KV pairs
   - Shows performance optimization by avoiding redundant computations

3. **Visualization**
   - Provides visual comparisons of attention patterns with and without caching
   - Shows how attention computation changes when using cached values
   - Illustrates cache reuse across different input sequences

## Getting Started

```bash
uv pip install -r requirements.txt
```

It's high time you use `uv` to install the dependencies!


# Author
Sagar Sarkale