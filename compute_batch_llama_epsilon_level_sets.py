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
    tokenizer
)
import torch.nn.functional as F

MOCK_TOKEN = "dog"

@dataclass
class BatchArgs:
    max_iters: int = 100
    learning_rate : float = 1e-3

    # What level-set we are gradient-descent converging to
    epsilon : float = 1e-6

    # Token embeddings over a batch of tokens (Batch x Seq x Dim)
    inputs_embeds : torch.Tensor

    # What we are converging to
    target_probabilities : list[torch.Tensor]

@dataclass
class GradientDescentBatchArgs:
    max_iters: int = 100
    learning_rate : float = 1e-3

    # What level-set we are gradient-descent converging to
    epsilon : float = 1e-2

    # Token embeddings over a batch of tokens (Batch x Seq x Dim)
    inputs_embeds : torch.Tensor

    # What we are converging to
    target_probabilities : list[torch.Tensor]

    # Where to zero out gradient (i.e.; if there are special tokens) - (Batch x Seq)
    gradient_masks : torch.Tensor

    # attention mask
    attention_mask : torch.Tensor

    # position_ids
    position_ids : torch.Tensor



    def copy(self):
        return GradientDescentBatchArgs(**self.__dict__)

    def __repr__(self):
        repr_keys = ["max_iters", "learning_rate", "epsilon"]
        values = "\n,".join([f"{key} = {value}" for key, value in self.__dict__.items() if key in repr_keys])
        return f"GradientDescentBatchArgs(\n{values}\n)"

@dataclass
class LearnEmbeddingsResult:
    inputs_embeds : np.ndarray
    previous_inputs_embeds : np.ndarray  # second-to-final embeddings


def create_gradient_descent_batch_args(args: BatchArgs) -> GradientDescentBatchArgs:

    mock_sentences = [MOCK_TOKEN]*len(args.inputs_embeds)
    mock_tokenization = tokenizer(mock_sentences, return_tensors="pt", padding=True)

    gradient_masks = []
    for input_ids in mock_tokenization['input_ids']:
        gradient_mask = np.array([int(input_id not in tokenizer.all_special_ids) for input_id in input_ids.tolist()])
        gradient_masks.append(gradient_mask)
    gradient_masks = np.asarray(gradient_masks, dtype=float)

    return GradientDescentBatchArgs(
        max_iters = args.max_iters,
        learning_rate = args.learning_rate,
        epsilon = args.epsilon,
        inputs_embeds = args.inputs_embeds,
        target_probabilities = args.target_probabilities,
        attention_mask = mock_tokenization['attention_mask'],
        position_ids = mock_tokenization['position_ids'],
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
    position_ids = torch.tensor(args.position_ids, device = device)
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

    for t in range(args.max_iters):
        opt.zero_grad(set_to_none=True)
        with torch.autocast("mps", dtype=torch.float16):
            logits = model(
                attention_mask = attention_mask,
                position_ids = position_ids,
                inputs_embeds=inputs_embeds).logits[:, -1, :]

            # Compute per-example KL divergence
            # To verify: is the last dimension prior to .sum(dim=-1) actual vocab dim?
            kl_per_example = F.kl_div(F.log_softmax(logits, -1), target_probabilities, reduction='none').sum(dim=-1)
            print(kl_per_example.shape)

            # Overall loss for backward pass (only for logging/backward)
            loss = kl_per_example.mean()

        scaler.scale(loss).backward()
        inputs_embeds.grad.mul_(gradient_masks)           # mask specials

        # Zero out gradients for examples below epsilon
        with torch.no_grad():
            below_epsilon = kl_per_example < args.epsilon
            active_examples = active_examples & (~below_epsilon)

            # Create per-example mask (batch_size, 1, 1) to mask all tokens in frozen examples
            example_mask = active_examples.float().view(batch_size, 1, 1)
            inputs_embeds.grad.mul_(example_mask)

        # Save previous embeddings before update (only for active examples)
        with torch.no_grad():
            if active_examples.any():
                previous_inputs_embeds[active_examples] = inputs_embeds.detach()[active_examples].clone()

        scaler.step(opt)
        scaler.update()

        # lightweight logging every 10 iters
        if t % 10 == 0:
            torch.mps.synchronize()
            num_active = active_examples.sum().item()
            print(f"step {t:4d}   loss {loss.item():.4f}   active: {num_active}/{batch_size}   ({(time.time() - ts)/10:.2f} seconds per step)")
            ts = time.time()

        # occasional allocator clean-up
        if t % 100 == 0:
            torch.mps.empty_cache()

        # Early termination when all examples are below epsilon
        if not active_examples.any():
            print(f"Early termination at step {t}: all examples below epsilon={args.epsilon}")
            break

    # Refinement pass to get closer to epsilon-level set
    print("Refining embeddings to epsilon-level set...")
    refined_embeds = refine_to_epsilon_level_set(
        model=model,
        previous_embeds=previous_inputs_embeds,
        final_embeds=inputs_embeds.detach(),
        target_probs=target_probabilities,
        epsilon=args.epsilon,
        attention_mask=attention_mask,
        position_ids=position_ids
    )

    return LearnEmbeddingsResult(
        inputs_embeds = refined_embeds.cpu().numpy(),
        previous_inputs_embeds = previous_inputs_embeds.detach().clone().cpu().numpy()
    )


def refine_to_epsilon_level_set(
    model,
    previous_embeds,
    final_embeds,
    target_probs,
    epsilon,
    attention_mask,
    position_ids,
    max_iters=20,
    tolerance=1e-7
):
    """Binary search between previous and final embeddings to find KL ≈ epsilon.

    Args:
        model: The language model
        previous_embeds: Embeddings before final step (likely KL >= epsilon)
        final_embeds: Embeddings after final step (likely KL < epsilon)
        target_probs: Target probability distributions
        epsilon: Target KL divergence threshold
        attention_mask: Attention mask for the model
        position_ids: Position IDs for the model
        max_iters: Maximum binary search iterations per example
        tolerance: Convergence tolerance for KL divergence

    Returns:
        Refined embeddings with KL ≈ epsilon
    """
    batch_size = final_embeds.shape[0]
    refined_embeds = final_embeds.clone()

    for batch_idx in range(batch_size):
        # Binary search for interpolation parameter α
        alpha_low, alpha_high = 0.0, 1.0  # 0=final, 1=previous
        best_alpha = 0.0
        best_kl_diff = float('inf')

        for iter_num in range(max_iters):
            alpha = (alpha_low + alpha_high) / 2.0

            # Interpolate: α=0 gives final_embeds, α=1 gives previous_embeds
            candidate = (1 - alpha) * final_embeds[batch_idx:batch_idx+1] + alpha * previous_embeds[batch_idx:batch_idx+1]

            with torch.no_grad():
                logits = model(
                    attention_mask=attention_mask[batch_idx:batch_idx+1],
                    position_ids=position_ids[batch_idx:batch_idx+1],
                    inputs_embeds=candidate
                ).logits[:, -1, :]

                kl = F.kl_div(
                    F.log_softmax(logits, -1),
                    target_probs[batch_idx:batch_idx+1],
                    reduction='none'
                ).sum(dim=-1).item()

            kl_diff = abs(kl - epsilon)

            # Track best candidate
            if kl_diff < best_kl_diff:
                best_kl_diff = kl_diff
                best_alpha = alpha

            # Check convergence
            if kl_diff < tolerance:
                if batch_idx == 0 or batch_idx == batch_size - 1:  # Log first and last
                    print(f"  Example {batch_idx}: converged at iter {iter_num}, KL={kl:.2e}, target={epsilon:.2e}")
                break

            # Binary search logic
            if kl < epsilon:
                # Need to move toward previous_embeds (increase α)
                alpha_low = alpha
            else:
                # Need to move toward final_embeds (decrease α)
                alpha_high = alpha
        else:
            # Max iterations reached, use best candidate
            alpha = best_alpha
            if batch_idx == 0 or batch_idx == batch_size - 1:
                print(f"  Example {batch_idx}: max iters reached, best KL diff={best_kl_diff:.2e}")

        # Apply best interpolation
        refined_embeds[batch_idx] = ((1 - alpha) * final_embeds[batch_idx] + alpha * previous_embeds[batch_idx])

    return refined_embeds


def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__ if hasattr(func, '__name__') else func.__class__.__name__} took {end - start} seconds")
        return result
    return wrapper
