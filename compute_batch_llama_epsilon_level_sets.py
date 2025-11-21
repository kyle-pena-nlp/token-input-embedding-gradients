import os, sys, re, time
from dataclasses import dataclass
import torch.nn as nn
from typing import Optional, Any
import torch, numpy as np
from tqdm import tqdm
import time
from llama_models import (
    device,
    model,
    tokenizer,
    full_vocab_embedding
)
import torch.nn.functional as F
from scipy import linalg
from torch_utils import synchronize, empty_cache, autocast

MOCK_TOKEN = "dog"

@dataclass
class BatchArgs:


    # Token embeddings over a batch of tokens (Batch x Seq x Dim)
    inputs_embeds : torch.Tensor

    # What we are converging to
    target_probabilities : list[torch.Tensor]

    max_iters: int = 100

    learning_rate : float = 1e-3

    # What level-set we are gradient-descent converging to
    epsilon : float = 1e-6

    epsilon_tolerance_scale : float = 1e-2

@dataclass
class GradientDescentBatchArgs:
    max_iters: int
    learning_rate : float

    # What level-set we are gradient-descent converging to
    epsilon : float

    epsilon_tolerance_scale : float

    # Token embeddings over a batch of tokens (Batch x Seq x Dim)
    inputs_embeds : torch.Tensor

    # What we are converging to
    target_probabilities : list[torch.Tensor]

    # Where to zero out gradient (i.e.; if there are special tokens) - (Batch x Seq)
    gradient_masks : torch.Tensor

    # attention mask
    attention_mask : torch.Tensor

    # position_ids
    #position_ids : torch.Tensor



    def copy(self):
        return GradientDescentBatchArgs(**self.__dict__)

    def __repr__(self):
        repr_keys = ["max_iters", "learning_rate", "epsilon", "epsilon_tolerance_scale"]
        values = "\n,".join([f"{key} = {value}" for key, value in self.__dict__.items() if key in repr_keys])
        return f"GradientDescentBatchArgs(\n{values}\n)"

@dataclass
class LearnEmbeddingsResult:
    starting_inputs_embeds : np.ndarray
    inputs_embeds : np.ndarray
    converged : np.ndarray  # Boolean array indicating which examples converged (KL < epsilon)

def create_gradient_descent_batch_args(args: BatchArgs) -> GradientDescentBatchArgs:

    mock_sentences = [MOCK_TOKEN]*len(args.inputs_embeds)
    mock_tokenization = tokenizer(mock_sentences, return_tensors="pt", padding=True)

    gradient_masks = []
    for input_ids in mock_tokenization['input_ids']:
        gradient_mask = np.array([int(input_id not in tokenizer.all_special_ids) for input_id in input_ids.tolist()])
        gradient_masks.append(gradient_mask)
    gradient_masks = np.asarray(gradient_masks, dtype=float)

    inputs_embeds = model.model.embed_tokens(mock_tokenization['input_ids'].to(device))

    # The mock tokenization's embedding is going to have a special token (the start token) at the first position
    # We swap that out with the inputs_emebds we provided in the BatchArgs
    inputs_embeds[:,1,:] = torch.tensor(args.inputs_embeds, device = device)

    return GradientDescentBatchArgs(
        max_iters = args.max_iters,
        learning_rate = args.learning_rate,
        epsilon = args.epsilon,
        epsilon_tolerance_scale = args.epsilon_tolerance_scale,
        inputs_embeds = inputs_embeds,
        target_probabilities = args.target_probabilities,
        attention_mask = mock_tokenization['attention_mask'],
        #position_ids = mock_tokenization['position_ids'],
        gradient_masks = gradient_masks
    )

def learn_embeddings(args : BatchArgs) -> LearnEmbeddingsResult:

    # Prepare data
    args = create_gradient_descent_batch_args(args)

    # Prepare inputs for gradient computation
    inputs_embeds = nn.Parameter(torch.tensor(args.inputs_embeds, device = device), requires_grad=True)
    target_probabilities = torch.tensor(args.target_probabilities, device = device)
    gradient_masks = torch.tensor(args.gradient_masks, dtype=torch.float, device = device).unsqueeze(-1)
    attention_mask = torch.tensor(args.attention_mask, device = device)
    #position_ids = torch.tensor(args.position_ids, device = device)
    learning_rate = args.learning_rate or 1e-3

    model.eval()                              # inference mode
    model.requires_grad_(False)               # turn off grads for every param

    scaler = torch.amp.GradScaler()   # built-in utility
    opt = torch.optim.Adam([inputs_embeds], lr=learning_rate)
    ts = time.time()

    # Track which examples are still being optimized
    batch_size = inputs_embeds.shape[0]
    active_examples = torch.ones(batch_size, dtype=torch.bool, device=device)

    # Track previous embeddings for analysis
    previous_inputs_embeds = inputs_embeds.detach().clone()

    # Initial logging
    print("\n" + "="*80)
    print("GRADIENT DESCENT: Converging to epsilon-level set")
    print("="*80)
    print(f"Batch size:      {batch_size}")
    print(f"Target epsilon:  {args.epsilon:.2e}")
    print(f"Learning rate:   {learning_rate:.2e}")
    print(f"Max iterations:  {args.max_iters}")
    print("="*80 + "\n")

    for t in range(args.max_iters):
        opt.zero_grad(set_to_none=True)
        with autocast():
            logits = model(
                attention_mask = attention_mask,
                #position_ids = position_ids,
                inputs_embeds=inputs_embeds).logits[:, -1, :]

            # Compute per-example KL divergence
            kl_per_example = F.kl_div(F.log_softmax(logits, -1), target_probabilities, reduction='none').sum(dim=-1)

            # Loss: squared distance from epsilon (directly optimize for KL ≈ epsilon)
            # This eliminates the need for binary search refinement
            kl_diff_per_example = torch.abs(kl_per_example - args.epsilon)

        # Check convergence FIRST, before backward/step (in no_grad for efficiency)
        with torch.no_grad():
            # Converged = within tolerance of epsilon
            tolerance = args.epsilon * args.epsilon_tolerance_scale
            converged_mask = kl_diff_per_example < tolerance
            active_examples = active_examples & (~converged_mask)

        # Compute loss ONLY over active examples to avoid ADAM state being influenced
        # by converged examples with near-zero loss
        if active_examples.any():
            loss = (kl_diff_per_example[active_examples] ** 2).mean()
        else:
            # All converged - doesn't matter, but compute something for logging
            loss = (kl_diff_per_example ** 2).mean()

        scaler.scale(loss).backward()
        inputs_embeds.grad.mul_(gradient_masks)           # mask specials

        # Zero out gradients for inactive examples
        with torch.no_grad():
            # Create per-example mask (batch_size, 1, 1) to mask all tokens in frozen examples
            example_mask = active_examples.float().view(batch_size, 1, 1)
            inputs_embeds.grad.mul_(example_mask)

        # Save embeddings BEFORE step (for active examples that will take this step)
        # After the step, these might have converged, so this captures the "previous" state
        with torch.no_grad():
            if active_examples.any():
                previous_inputs_embeds[active_examples] = inputs_embeds.detach()[active_examples].clone()

        scaler.step(opt)
        scaler.update()

        # Detailed logging every 10 iters
        if t % 10 == 0:
            synchronize()
            num_active = active_examples.sum().item()
            num_converged = batch_size - num_active

            # Log KL and distance from epsilon
            if active_examples.any():
                kl_stats = kl_per_example[active_examples]
                kl_diff_stats = kl_diff_per_example[active_examples]
            else:
                kl_stats = kl_per_example
                kl_diff_stats = kl_diff_per_example

            kl_min = kl_stats.min().item() if kl_stats.numel() > 0 else 0.0
            kl_median = kl_stats.median().item() if kl_stats.numel() > 0 else 0.0
            kl_max = kl_stats.max().item() if kl_stats.numel() > 0 else 0.0

            kl_diff_mean = kl_diff_stats.mean().item() if kl_diff_stats.numel() > 0 else 0.0
            kl_diff_max = kl_diff_stats.max().item() if kl_diff_stats.numel() > 0 else 0.0

            time_per_step = (time.time() - ts) / 10

            print(f"Step {t:4d} │ Loss: {loss.item():.4f} │ "
                  f"KL [min/med/max]: [{kl_min:.2e}/{kl_median:.2e}/{kl_max:.2e}] │ "
                  f"|KL-ε| [mean/max]: [{kl_diff_mean:.2e}/{kl_diff_max:.2e}] │ "
                  f"Active: {num_active}/{batch_size} │ "
                  f"Converged: {num_converged} │ "
                  f"{time_per_step:.2f}s/step")
            ts = time.time()

        # occasional allocator clean-up
        if t % 100 == 0:
            empty_cache()

        # Early termination when all examples converged
        if not active_examples.any():
            print(f"\n✓ Early termination at step {t}: all examples converged (|KL - ε| < tolerance)")
            break

    # Final gradient descent summary
    print("\n" + "-"*80)
    print("OPTIMIZATION COMPLETE")
    converged = ~active_examples
    num_converged = converged.sum().item()
    print(f"Converged examples: {num_converged}/{batch_size} ({100*num_converged/batch_size:.1f}%)")
    if num_converged < batch_size:
        print(f"Non-converged:      {batch_size - num_converged}/{batch_size}")

    # Final statistics
    with torch.no_grad():
        with autocast():
            final_logits = model(
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds
            ).logits[:, -1, :]
            final_kl = F.kl_div(
                F.log_softmax(final_logits, -1),
                target_probabilities,
                reduction='none'
            ).sum(dim=-1)
            final_kl_diff = torch.abs(final_kl - args.epsilon)

    print(f"\nFinal KL statistics relative to epsilon ({args.epsilon:.2e}):")
    print(f"  Mean |KL - ε|:     {final_kl_diff.mean().item():.2e}")
    print(f"  Median |KL - ε|:   {final_kl_diff.median().item():.2e}")
    print(f"  Max |KL - ε|:      {final_kl_diff.max().item():.2e}")
    print(f"  Min |KL - ε|:      {final_kl_diff.min().item():.2e}")

    tolerance = args.epsilon * args.epsilon_tolerance_scale
    success_indicators = final_kl_diff < tolerance
    num_successful = success_indicators.sum().item()
    print(f"\nSuccess rate: {num_successful}/{batch_size} ({100*num_successful/batch_size:.1f}%) within tolerance")
    print("-"*80 + "\n")

    return LearnEmbeddingsResult(
        starting_inputs_embeds = args.inputs_embeds[:,1,:].detach().clone().cpu().numpy(),
        inputs_embeds = inputs_embeds[:,1,:].detach().clone().cpu().numpy(),
        converged = converged.cpu().numpy()
    )



def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__ if hasattr(func, '__name__') else func.__class__.__name__} took {end - start} seconds")
        return result
    return wrapper


def fit_gaussian_ellipse_and_sample(
    embeddings: np.ndarray,
    n_samples: int,
    confidence_level: float = 0.99,
    random_seed: Optional[int] = None
) -> np.ndarray:
    """Fit a Gaussian ellipse to embeddings and sample uniformly within it.

    Args:
        embeddings: Embedding matrix of shape (n_embeddings, embedding_dim)
        n_samples: Number of samples to generate
        confidence_level: Confidence level for the ellipse bounds (default: 0.99)
        random_seed: Random seed for reproducibility

    Returns:
        Sampled embeddings of shape (n_samples, embedding_dim) within the ellipse
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # Fit Gaussian: compute mean and covariance
    mean = np.mean(embeddings, axis=0)
    centered = embeddings - mean
    cov = np.cov(centered.T)

    # Eigendecomposition for the ellipse
    eigenvalues, eigenvectors = linalg.eigh(cov)

    # Chi-squared quantile for multivariate Gaussian at given confidence level
    from scipy.stats import chi2
    dim = embeddings.shape[1]
    chi2_val = chi2.ppf(confidence_level, df=dim)

    # Scale eigenvalues by chi-squared quantile
    scaled_radii = np.sqrt(eigenvalues * chi2_val)

    # Sample uniformly within unit sphere, then transform to ellipse
    samples = []
    while len(samples) < n_samples:
        # Sample from unit sphere using rejection sampling
        candidate = np.random.randn(dim)
        norm = np.linalg.norm(candidate)

        if norm > 0:
            # Uniform in unit ball: scale by u^(1/dim) where u ~ Uniform(0,1)
            u = np.random.uniform(0, 1)
            radius = u ** (1.0 / dim)
            unit_sample = (candidate / norm) * radius

            # Transform to ellipse: scale by eigenvalues and rotate by eigenvectors
            ellipse_sample = mean + eigenvectors @ (scaled_radii * unit_sample)
            samples.append(ellipse_sample)

    return np.array(samples).astype(np.float32)


def sample_vocab_ellipse(n_samples: int, confidence_level: float = 0.99, random_seed: Optional[int] = None) -> np.ndarray:
    """Sample embeddings uniformly within the Gaussian ellipse of the vocabulary.

    Args:
        n_samples: Number of samples to generate
        confidence_level: Confidence level for the ellipse bounds (default: 0.99)
        random_seed: Random seed for reproducibility

    Returns:
        Sampled embeddings of shape (n_samples, embedding_dim)
    """
    return fit_gaussian_ellipse_and_sample(
        embeddings=full_vocab_embedding,
        n_samples=n_samples,
        confidence_level=confidence_level,
        random_seed=random_seed
    )


def compute_local_normal_vector(
    point: np.ndarray,
    target_probability: np.ndarray,
    epsilon: float,
    delta: float = 1e-3,
    max_iters: int = 100,
    learning_rate: float = 1e-3,
    epsilon_tolerance_scale: float = 1e-2,
    verbose: bool = False
) -> np.ndarray:
    """Compute the local normal vector at a point on the epsilon level-set using central differencing.

    For each dimension, this function:
    1. Perturbs the point by adding a small positive offset
    2. Optimizes the perturbed point back to the epsilon level-set
    3. Computes distance 'a' from perturbed to converged point
    4. Perturbs the point by adding a small negative offset
    5. Optimizes this perturbed point back to the epsilon level-set
    6. Computes distance 'b' from perturbed to converged point
    7. Sets the normal vector component to (b - a)

    The resulting vector is then normalized.

    Args:
        point: Point on the epsilon level-set, shape (embedding_dim,)
        target_probability: Target probability distribution, shape (vocab_size,)
        epsilon: The epsilon value defining the level-set
        delta: Small offset for perturbation (default: 1e-3)
        max_iters: Maximum optimization iterations (default: 100)
        learning_rate: Learning rate for optimization (default: 1e-3)
        epsilon_tolerance_scale: Tolerance scale for convergence (default: 1e-2)
        verbose: Whether to print progress (default: False)

    Returns:
        Normal vector at the point, shape (embedding_dim,), normalized to unit length
    """
    embedding_dim = point.shape[0]
    normal_vector = np.zeros(embedding_dim, dtype=np.float32)

    if verbose:
        print("\n" + "="*80)
        print("COMPUTING LOCAL NORMAL VECTOR via Central Differencing")
        print("="*80)
        print(f"Embedding dimension: {embedding_dim}")
        print(f"Perturbation delta:  {delta:.2e}")
        print(f"Target epsilon:      {epsilon:.2e}")
        print("="*80 + "\n")

    # Process each dimension
    for dim_idx in tqdm(range(embedding_dim), desc="Computing normal components", disable=not verbose):

        # === Positive perturbation ===
        perturbed_positive = point.copy()
        perturbed_positive[dim_idx] += delta

        # Optimize positive perturbation to epsilon level-set
        args_positive = BatchArgs(
            inputs_embeds=perturbed_positive.reshape(1, -1),
            target_probabilities=[target_probability],
            max_iters=max_iters,
            learning_rate=learning_rate,
            epsilon=epsilon,
            epsilon_tolerance_scale=epsilon_tolerance_scale
        )

        # Suppress output during optimization
        import contextlib
        import io

        with contextlib.redirect_stdout(io.StringIO()):
            result_positive = learn_embeddings(args_positive)

        converged_positive = result_positive.inputs_embeds[0]
        distance_a = np.linalg.norm(perturbed_positive - converged_positive)

        # === Negative perturbation ===
        perturbed_negative = point.copy()
        perturbed_negative[dim_idx] -= delta

        # Optimize negative perturbation to epsilon level-set
        args_negative = BatchArgs(
            inputs_embeds=perturbed_negative.reshape(1, -1),
            target_probabilities=[target_probability],
            max_iters=max_iters,
            learning_rate=learning_rate,
            epsilon=epsilon,
            epsilon_tolerance_scale=epsilon_tolerance_scale
        )

        with contextlib.redirect_stdout(io.StringIO()):
            result_negative = learn_embeddings(args_negative)

        converged_negative = result_negative.inputs_embeds[0]
        distance_b = np.linalg.norm(perturbed_negative - converged_negative)

        # Set normal vector component using central difference
        normal_vector[dim_idx] = distance_b - distance_a

        if verbose and dim_idx % 100 == 0:
            print(f"Dimension {dim_idx:4d}: a={distance_a:.4e}, b={distance_b:.4e}, (b-a)={normal_vector[dim_idx]:.4e}")

    # Normalize the normal vector
    norm = np.linalg.norm(normal_vector)

    if norm > 0:
        normal_vector = normal_vector / norm
    else:
        if verbose:
            print("⚠ Warning: Normal vector has zero magnitude, returning zero vector")

    if verbose:
        print("\n" + "-"*80)
        print("NORMAL VECTOR COMPUTATION COMPLETE")
        print(f"Vector magnitude (before normalization): {norm:.4e}")
        print(f"Normalized normal vector L2 norm:        {np.linalg.norm(normal_vector):.6f}")
        print("-"*80 + "\n")

    return normal_vector


def compute_local_normal_vector_batched(
    point: np.ndarray,
    target_probability: np.ndarray,
    epsilon: float,
    delta: float = 1e-3,
    max_iters: int = 100,
    learning_rate: float = 1e-3,
    epsilon_tolerance_scale: float = 1e-2,
    batch_size: int = 128,
    verbose: bool = True
) -> np.ndarray:
    """Compute the local normal vector using central differencing with batched optimization.

    This is a more efficient version of compute_local_normal_vector that processes multiple
    dimensions simultaneously in batches, significantly reducing computation time.

    Args:
        point: Point on the epsilon level-set, shape (embedding_dim,)
        target_probability: Target probability distribution, shape (vocab_size,)
        epsilon: The epsilon value defining the level-set
        delta: Small offset for perturbation (default: 1e-3)
        max_iters: Maximum optimization iterations (default: 100)
        learning_rate: Learning rate for optimization (default: 1e-3)
        epsilon_tolerance_scale: Tolerance scale for convergence (default: 1e-2)
        batch_size: Number of dimensions to process simultaneously (default: 128)
        verbose: Whether to print progress (default: True)

    Returns:
        Normal vector at the point, shape (embedding_dim,), normalized to unit length
    """
    embedding_dim = point.shape[0]
    normal_vector = np.zeros(embedding_dim, dtype=np.float32)

    if verbose:
        print("\n" + "="*80)
        print("COMPUTING LOCAL NORMAL VECTOR via Batched Central Differencing")
        print("="*80)
        print(f"Embedding dimension: {embedding_dim}")
        print(f"Perturbation delta:  {delta:.2e}")
        print(f"Target epsilon:      {epsilon:.2e}")
        print(f"Batch size:          {batch_size}")
        print("="*80 + "\n")

    # Process dimensions in batches
    num_batches = (embedding_dim + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(num_batches), desc="Processing dimension batches", disable=not verbose):
        start_dim = batch_idx * batch_size
        end_dim = min((batch_idx + 1) * batch_size, embedding_dim)
        current_batch_size = end_dim - start_dim

        # Create perturbations for this batch
        # For each dimension in the batch, we need both positive and negative perturbations
        perturbed_points = []

        for dim_idx in range(start_dim, end_dim):
            # Positive perturbation
            perturbed_pos = point.copy()
            perturbed_pos[dim_idx] += delta
            perturbed_points.append(perturbed_pos)

            # Negative perturbation
            perturbed_neg = point.copy()
            perturbed_neg[dim_idx] -= delta
            perturbed_points.append(perturbed_neg)

        perturbed_points = np.array(perturbed_points)  # Shape: (2*current_batch_size, embedding_dim)

        # Prepare target probabilities for the batch
        target_probs_batch = [target_probability] * (2 * current_batch_size)

        # Optimize all perturbations in this batch simultaneously
        args_batch = BatchArgs(
            inputs_embeds=perturbed_points,
            target_probabilities=target_probs_batch,
            max_iters=max_iters,
            learning_rate=learning_rate,
            epsilon=epsilon,
            epsilon_tolerance_scale=epsilon_tolerance_scale
        )

        # Suppress output during optimization
        import contextlib
        import io

        if not verbose:
            with contextlib.redirect_stdout(io.StringIO()):
                result_batch = learn_embeddings(args_batch)
        else:
            print(f"\nBatch {batch_idx + 1}/{num_batches}: Optimizing dimensions {start_dim} to {end_dim-1}")
            result_batch = learn_embeddings(args_batch)

        converged_points = result_batch.inputs_embeds  # Shape: (2*current_batch_size, embedding_dim)

        # Compute normal vector components for this batch
        for i, dim_idx in enumerate(range(start_dim, end_dim)):
            pos_idx = 2 * i
            neg_idx = 2 * i + 1

            # Distance from positive perturbation to its converged point
            distance_a = np.linalg.norm(perturbed_points[pos_idx] - converged_points[pos_idx])

            # Distance from negative perturbation to its converged point
            distance_b = np.linalg.norm(perturbed_points[neg_idx] - converged_points[neg_idx])

            # Central difference
            normal_vector[dim_idx] = distance_b - distance_a

            if verbose and batch_idx == 0 and i < 5:
                print(f"  Dimension {dim_idx:4d}: a={distance_a:.4e}, b={distance_b:.4e}, (b-a)={normal_vector[dim_idx]:.4e}")

    # Normalize the normal vector
    norm = np.linalg.norm(normal_vector)

    if norm > 0:
        normal_vector = normal_vector / norm
    else:
        if verbose:
            print("⚠ Warning: Normal vector has zero magnitude, returning zero vector")

    if verbose:
        print("\n" + "-"*80)
        print("BATCHED NORMAL VECTOR COMPUTATION COMPLETE")
        print(f"Vector magnitude (before normalization): {norm:.4e}")
        print(f"Normalized normal vector L2 norm:        {np.linalg.norm(normal_vector):.6f}")
        print("-"*80 + "\n")

    return normal_vector
