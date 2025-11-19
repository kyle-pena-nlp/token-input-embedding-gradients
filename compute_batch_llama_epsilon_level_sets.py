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
    for t in range(args.max_iters):
        opt.zero_grad(set_to_none=True)
        with torch.autocast("mps", dtype=torch.float16):
            logits = model(
                attention_mask = attention_mask,
                position_ids = position_ids,
                inputs_embeds=inputs_embeds).logits[:, -1, :]
            loss = F.kl_div(F.log_softmax(logits, -1), target_probabilities, reduction='batchmean')
        scaler.scale(loss).backward()
        inputs_embeds.grad.mul_(gradient_masks)           # mask specials
        scaler.step(opt)
        scaler.update()

        # lightweight logging every 10 iters
        if t % 10 == 0:
            torch.mps.synchronize()
            print(f"step {t:4d}   loss {loss.item():.4f} ({(time.time() - ts)/10:.2f} seconds per step)")
            ts = time.time()

        # occasional allocator clean-up
        if t % 100 == 0:
            torch.mps.empty_cache()

    return LearnEmbeddingsResult(
        inputs_embeds = inputs_embeds.detach().clone().cpu().numpy()
    )


def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__ if hasattr(func, '__name__') else func.__class__.__name__} took {end - start} seconds")
        return result
    return wrapper
