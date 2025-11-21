"""
Example: Computing local normal vectors on epsilon level-sets

This script demonstrates how to use the normal vector computation functions.
"""

import numpy as np
import torch
from compute_batch_llama_epsilon_level_sets import (
    BatchArgs,
    learn_embeddings,
    sample_vocab_ellipse,
    compute_local_normal_vector,
    compute_local_normal_vector_batched
)
from llama_models import tokenizer, model, device

# ============================================================================
# Setup: Create a target probability distribution and find a point on its
# epsilon level-set
# ============================================================================

print("="*80)
print("EXAMPLE: Computing Local Normal Vectors on Epsilon Level-Sets")
print("="*80)

# Define epsilon for the level-set
epsilon = 1e-6

# Create a simple target probability distribution
# For this example, we'll create a distribution that heavily favors a few tokens
vocab_size = model.config.vocab_size
target_probability = np.zeros(vocab_size, dtype=np.float32)

# Put most probability mass on a few tokens (e.g., tokens for common words)
target_tokens = tokenizer.encode("the cat sat", add_special_tokens=False)
for i, token_id in enumerate(target_tokens):
    target_probability[token_id] = 0.3 if i == 0 else 0.2

# Normalize
target_probability = target_probability / target_probability.sum()
target_probability = torch.tensor(target_probability, device=device)

print(f"\nTarget distribution:")
print(f"  Non-zero entries: {(target_probability > 0).sum().item()}")
print(f"  Entropy: {-(target_probability * torch.log(target_probability + 1e-10)).sum().item():.4f}")

# Sample an initial point from the vocabulary ellipse
print("\nSampling initial point from vocabulary ellipse...")
initial_points = sample_vocab_ellipse(n_samples=1, random_seed=42)
initial_point = initial_points[0]

print(f"Initial point shape: {initial_point.shape}")
print(f"Initial point norm: {np.linalg.norm(initial_point):.4f}")

# ============================================================================
# Step 1: Optimize initial point to the epsilon level-set
# ============================================================================

print("\n" + "="*80)
print("STEP 1: Finding a point on the epsilon level-set")
print("="*80)

args = BatchArgs(
    inputs_embeds=initial_point.reshape(1, -1),
    target_probabilities=[target_probability.cpu().numpy()],
    max_iters=200,
    learning_rate=1e-3,
    epsilon=epsilon,
    epsilon_tolerance_scale=1e-2
)

result = learn_embeddings(args)
level_set_point = result.inputs_embeds[0]

print(f"\nOptimization converged: {result.converged[0]}")
print(f"Point on level-set shape: {level_set_point.shape}")
print(f"Distance from initial: {np.linalg.norm(level_set_point - initial_point):.4e}")

# ============================================================================
# Step 2: Compute local normal vector using the sequential method
# ============================================================================

print("\n" + "="*80)
print("STEP 2: Computing normal vector (Sequential method)")
print("="*80)
print("\n⚠ Note: This will take a LONG time for high-dimensional embeddings!")
print("For demonstration, we'll only compute the first 10 dimensions.")
print("For production use, see the batched method below.\n")

# For demonstration, let's only compute on a subset of dimensions
# by truncating the point (in practice, you'd use the full dimension)
demo_point = level_set_point[:10]  # Only first 10 dimensions
demo_target = target_probability.cpu().numpy()

# Compute normal vector (this will be slow for full dimensions!)
# normal_vector = compute_local_normal_vector(
#     point=demo_point,
#     target_probability=demo_target,
#     epsilon=epsilon,
#     delta=1e-3,
#     max_iters=100,
#     learning_rate=1e-3,
#     verbose=True
# )

print("Skipping sequential method for this demo (would take too long).")
print("See the batched method below for efficient computation.\n")

# ============================================================================
# Step 3: Compute local normal vector using the BATCHED method (RECOMMENDED)
# ============================================================================

print("\n" + "="*80)
print("STEP 3: Computing normal vector (Batched method - RECOMMENDED)")
print("="*80)

# This is much faster! Process multiple dimensions simultaneously
normal_vector_batched = compute_local_normal_vector_batched(
    point=level_set_point,
    target_probability=target_probability.cpu().numpy(),
    epsilon=epsilon,
    delta=1e-3,
    max_iters=100,
    learning_rate=1e-3,
    batch_size=64,  # Process 64 dimensions at a time (128 points: 2 per dimension)
    verbose=True
)

# ============================================================================
# Step 4: Analyze the computed normal vector
# ============================================================================

print("\n" + "="*80)
print("STEP 4: Analyzing the normal vector")
print("="*80)

print(f"\nNormal vector statistics:")
print(f"  Shape: {normal_vector_batched.shape}")
print(f"  L2 norm: {np.linalg.norm(normal_vector_batched):.6f}")
print(f"  Min component: {normal_vector_batched.min():.4e}")
print(f"  Max component: {normal_vector_batched.max():.4e}")
print(f"  Mean |component|: {np.abs(normal_vector_batched).mean():.4e}")
print(f"  Std |component|: {np.abs(normal_vector_batched).std():.4e}")

# Find dimensions with largest contributions
top_k = 10
top_indices = np.argsort(np.abs(normal_vector_batched))[-top_k:][::-1]
print(f"\nTop {top_k} dimensions by magnitude:")
for i, idx in enumerate(top_indices):
    print(f"  {i+1}. Dimension {idx:4d}: {normal_vector_batched[idx]:+.4e}")

# ============================================================================
# Step 5: Verify the normal vector properties
# ============================================================================

print("\n" + "="*80)
print("STEP 5: Verification")
print("="*80)

# The normal vector should be perpendicular to the level-set at this point
# We can verify by checking that small movements along the tangent plane
# (orthogonal to the normal) stay approximately on the same level-set

# Create a random tangent direction (orthogonal to normal)
random_direction = np.random.randn(level_set_point.shape[0]).astype(np.float32)
random_direction = random_direction / np.linalg.norm(random_direction)

# Make it orthogonal to the normal via Gram-Schmidt
tangent_direction = random_direction - np.dot(random_direction, normal_vector_batched) * normal_vector_batched
tangent_direction = tangent_direction / np.linalg.norm(tangent_direction)

# Verify orthogonality
dot_product = np.dot(tangent_direction, normal_vector_batched)
print(f"\nOrthogonality check:")
print(f"  Dot product (normal · tangent): {dot_product:.4e}")
print(f"  Should be close to 0: {'✓ PASS' if abs(dot_product) < 1e-6 else '✗ FAIL'}")

print("\n" + "="*80)
print("Example complete!")
print("="*80)
