# Local Normal Vector Computation on Epsilon Level-Sets

This document describes the methods for computing local normal vectors at points on epsilon level-sets of the KL divergence function.

## Overview

Given a point on an epsilon level-set (where `KL(model_output || target_distribution) ≈ ε`), we can compute the local normal vector using **central differencing**. The normal vector indicates the direction of steepest change perpendicular to the level-set.

## Mathematical Background

For a point **x** on the epsilon level-set, we want to find the normal vector **n** such that:
- **n** is perpendicular to the level-set at **x**
- **n** points in the direction of increasing KL divergence

### Central Differencing Method

For each dimension `i`:

1. **Positive perturbation**: Create `x⁺ᵢ = x + δeᵢ` where `eᵢ` is the i-th standard basis vector
2. **Optimize to level-set**: Find `x⁺ᵢ* = argmin |KL(f(x')) - ε|` starting from `x⁺ᵢ`
3. **Compute distance a**: `a = ||x⁺ᵢ - x⁺ᵢ*||`
4. **Negative perturbation**: Create `x⁻ᵢ = x - δeᵢ`
5. **Optimize to level-set**: Find `x⁻ᵢ* = argmin |KL(f(x')) - ε|` starting from `x⁻ᵢ`
6. **Compute distance b**: `b = ||x⁻ᵢ - x⁻ᵢ*||`
7. **Set normal component**: `nᵢ = b - a`

Finally, normalize: **n** ← **n** / ||**n**||

### Intuition

- If the level-set curves toward positive direction in dimension `i`, then `b > a` (negative perturbation has farther to travel back)
- If the level-set curves toward negative direction in dimension `i`, then `a > b`
- The difference `(b - a)` captures the local curvature in that dimension

## Implementation

### Functions Available

#### 1. `compute_local_normal_vector` (Sequential)

```python
def compute_local_normal_vector(
    point: np.ndarray,               # Point on epsilon level-set
    target_probability: np.ndarray,  # Target distribution
    epsilon: float,                   # Epsilon value
    delta: float = 1e-3,             # Perturbation size
    max_iters: int = 100,            # Optimization iterations
    learning_rate: float = 1e-3,     # Learning rate
    epsilon_tolerance_scale: float = 1e-2,
    verbose: bool = False
) -> np.ndarray:
```

**Pros:**
- Simple, straightforward implementation
- Easy to understand and debug

**Cons:**
- **VERY SLOW** for high-dimensional embeddings
- For 2048-dimensional Llama embeddings: 2048 dimensions × 2 perturbations = 4096 optimization runs!
- Not recommended for production use

#### 2. `compute_local_normal_vector_batched` (Batched - RECOMMENDED)

```python
def compute_local_normal_vector_batched(
    point: np.ndarray,
    target_probability: np.ndarray,
    epsilon: float,
    delta: float = 1e-3,
    max_iters: int = 100,
    learning_rate: float = 1e-3,
    epsilon_tolerance_scale: float = 1e-2,
    batch_size: int = 128,           # Dimensions per batch
    verbose: bool = True
) -> np.ndarray:
```

**Pros:**
- **Much faster**: Processes multiple dimensions simultaneously
- GPU-efficient: Leverages batched optimization
- For 2048 dims with batch_size=128: Only 16 optimization calls instead of 4096!

**Cons:**
- Slightly more memory usage (but manageable)

**Recommended settings:**
- `batch_size=64-128` for RTX 5090 with 32GB VRAM
- Adjust based on available GPU memory

## Usage Example

```python
import numpy as np
from compute_batch_llama_epsilon_level_sets import (
    BatchArgs,
    learn_embeddings,
    compute_local_normal_vector_batched
)
from llama_models import tokenizer, model

# 1. Define target distribution
vocab_size = model.config.vocab_size
target_prob = np.zeros(vocab_size, dtype=np.float32)
target_prob[tokenizer.encode("hello", add_special_tokens=False)] = 1.0
target_prob = target_prob / target_prob.sum()

# 2. Get a point on the epsilon level-set
epsilon = 1e-6
initial_point = sample_vocab_ellipse(n_samples=1)[0]

args = BatchArgs(
    inputs_embeds=initial_point.reshape(1, -1),
    target_probabilities=[target_prob],
    epsilon=epsilon,
    max_iters=200
)
result = learn_embeddings(args)
level_set_point = result.inputs_embeds[0]

# 3. Compute normal vector (BATCHED - recommended)
normal_vector = compute_local_normal_vector_batched(
    point=level_set_point,
    target_probability=target_prob,
    epsilon=epsilon,
    delta=1e-3,
    batch_size=128,
    verbose=True
)

print(f"Normal vector shape: {normal_vector.shape}")
print(f"Normal vector norm: {np.linalg.norm(normal_vector):.6f}")
```

## Parameters Guide

### `delta` (perturbation size)
- **Default**: `1e-3`
- **Smaller values**: More accurate but may be affected by numerical precision
- **Larger values**: Faster convergence but less accurate approximation
- **Recommended range**: `1e-4` to `1e-2`

### `batch_size`
- **Default**: `128`
- **Smaller values**: Less memory, slower computation
- **Larger values**: More memory, faster computation
- **For RTX 5090 (32GB)**: Can go up to `256-512` depending on embedding dimension

### `max_iters`
- **Default**: `100`
- Set based on how quickly your level-set optimizations converge
- Monitor convergence rates from initial `learn_embeddings` call

### `epsilon_tolerance_scale`
- **Default**: `1e-2`
- Tolerance is `epsilon * epsilon_tolerance_scale`
- For `epsilon=1e-6`, tolerance is `1e-8`
- Tighter tolerance = more accurate but slower

## Performance Benchmarks

Approximate times on RTX 5090 for Llama-3.2-1B (2048-dim embeddings):

| Method | Batch Size | Estimated Time |
|--------|------------|----------------|
| Sequential | N/A | ~6-8 hours |
| Batched | 64 | ~15-20 minutes |
| Batched | 128 | ~10-15 minutes |
| Batched | 256 | ~8-12 minutes |

*Note: Times depend on convergence speed and `max_iters`*

## Interpretation

The computed normal vector has several uses:

1. **Gradient information**: Shows direction of steepest ascent/descent on level-set
2. **Tangent space**: Vectors orthogonal to **n** lie in the tangent space
3. **Curvature analysis**: Large components indicate strong curvature in those dimensions
4. **Manifold exploration**: Can walk along tangent plane to explore the level-set

### Example Analysis

```python
# Find dimensions with largest normal components
top_k = 10
top_indices = np.argsort(np.abs(normal_vector))[-top_k:][::-1]
print("Dimensions with strongest curvature:")
for idx in top_indices:
    print(f"  Dimension {idx}: {normal_vector[idx]:+.4e}")

# Create tangent direction (orthogonal to normal)
random_dir = np.random.randn(normal_vector.shape[0])
random_dir = random_dir / np.linalg.norm(random_dir)
tangent_dir = random_dir - np.dot(random_dir, normal_vector) * normal_vector
tangent_dir = tangent_dir / np.linalg.norm(tangent_dir)

# Verify orthogonality
print(f"Dot product (should be ~0): {np.dot(tangent_dir, normal_vector):.2e}")
```

## Common Issues

### 1. Optimization doesn't converge
- Increase `max_iters`
- Adjust `learning_rate` (try `5e-4` or `2e-3`)
- Check if point is actually on the level-set initially

### 2. Normal vector has very small magnitude
- May indicate the level-set is nearly flat at this point
- Try a different point on the level-set
- Check if `delta` is too large

### 3. Out of memory
- Reduce `batch_size`
- Process in smaller chunks
- Clear GPU cache: `torch.cuda.empty_cache()` (or `torch.mps.empty_cache()` for Mac)

### 4. Takes too long
- Use the batched version, not sequential
- Increase `batch_size` if GPU memory allows
- Reduce `max_iters` if convergence is fast
- Loosen `epsilon_tolerance_scale`

## Related Files

- **`compute_batch_llama_epsilon_level_sets.py`**: Main implementation
- **`example_normal_vector_computation.py`**: Usage example
- **`find_llama_epsilon_level_sets.ipynb`**: Interactive notebook (recommended starting point)

## References

Central differencing for numerical derivatives:
- [Numerical Differentiation](https://en.wikipedia.org/wiki/Numerical_differentiation)
- [Finite Difference Methods](https://en.wikipedia.org/wiki/Finite_difference_method)

Level-set methods:
- Osher, S., & Fedkiw, R. (2003). Level Set Methods and Dynamic Implicit Surfaces.
