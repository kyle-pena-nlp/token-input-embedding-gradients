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
    full_vocab_embedding,
    full_vocab_embedding_torch,
    vocab_embedding_no_special_tokens,
    vocab_embeddings_no_special_tokens_norm,
    full_vocab_embedding_torch_norm
)
import torch.nn.functional as F
import torch.distributions as dist



@dataclass
class BatchInputsEmbedsGradArgs:
    tokenized_no_input_ids : Any
    initial_inputs_embeds : torch.Tensor
    inputs_embeds : torch.Tensor
    target_probabilities : torch.Tensor
    N : torch.Tensor
    L1_embed_loss : float
    seq_plausibility_loss : float
    gradient_masks : torch.Tensor

@dataclass
class BatchArgs:
    steps: int = 100
    start_offset : int = 0
    scramble_target_probs : bool = False
    learning_rate : float = 1e-3
    L1_embed_loss : float = 0.0
    seq_plausibility_loss : float = 0.0
    early_stopping_threshold : float = 1e-2
    examples_filepath : str = "rope_move/eng_sentences.tsv"
    num_examples : int = 32
    example_stride : int = 50
    permutation_seed : int = 42
    masked_sentences_seed : int = 42
    min_num_words : Optional[int] = None
    randomize_input_embeds : bool = False
    trim_input_ids : bool = False
    # populated by learn_embeddings
    sentences : Optional[list[str]] = None
    tokenization : Optional[Any] = None
    probabilities : Optional[list[torch.Tensor]] = None
    sentence_permutation : Optional[np.ndarray] = None
    target_probabilities : Optional[list[torch.Tensor]] = None
    tokenized_no_input_ids : Optional[Any] = None
    inputs_embeds : Optional[torch.Tensor] = None
    gradient_masks : Optional[torch.Tensor] = None

    def copy(self):
        return BatchArgs(**self.__dict__)

    def __repr__(self):
        repr_keys = ["steps", "learning_rate", "examples_filepath", "num_examples", "example_stride", "permutation_seed", "masked_sentences_seed", "no_ADAM", "min_num_words", "randomize_input_embeds", "trim_input_ids", "L1_embed_loss", "seq_plausibility_loss"]
        values = "\n,".join([f"{key} = {value}" for key, value in self.__dict__.items() if key in repr_keys])
        return f"BatchArgs(\n{values}\n)"

@dataclass
class LearnEmbeddingsResult:
    inputs_embeds : np.ndarray


def assign_target_probabilities(args : BatchArgs):
    N = len(args.sentences)
    if args.target_probabilities is not None:
        permutation = np.arange(N)
        target_probs = args.target_probabilities
    else:
        tokenizations = args.tokenization
        np.random.seed(args.permutation_seed)
        N = len(args.sentences)
        permutation = np.random.permutation(N)
        results = model(**tokenizations)
        target_probs = results.logits[:, -1, :]
        target_probs = target_probs.softmax(dim=1)
        target_probs = target_probs[permutation]

    if args.scramble_target_probs:
        print("Scrambling target probabilities on vocab dimensions")
        rng = np.random.default_rng(args.permutation_seed + 1)
        rng.shuffle(target_probs, axis=1)

    if isinstance(target_probs, torch.Tensor):
        target_probs = target_probs.detach().clone().numpy()
    args.target_probabilities = target_probs
    args.sentence_permutation = permutation


def tokenize_sentences(args : BatchArgs):
    # Run tokenization
    print(f"Tokenizing {len(args.sentences)} sentences")
    args.tokenization = tokenizer(args.sentences, return_tensors="pt", padding=True)
    if args.trim_input_ids:
        args.tokenization['input_ids'] = args.tokenization['input_ids'][:,:-3] # Trim so there is something meaningful to predict.
    args.tokenized_no_input_ids = { key: value.to(device) for (key,value) in args.tokenization.items() if key != "input_ids"}



def read_sentences(args : BatchArgs):
    if isinstance(args.examples_filepath, list) and len(args.examples_filepath) > 0 and all([isinstance(x, str) for x in args.examples_filepath]):
        args.sentences = args.examples_filepath
        return
    sentences = []
    with open(args.examples_filepath, "r") as f:
        print("Reading sentences")
        i = 0
        offset = 0
        for line in tqdm(f):
            if line.startswith("#"):
                continue
            offset += 1
            if offset < args.start_offset:
                continue
            i += 1
            if i % args.example_stride != 0:
                continue
            sentence = line.split("\t")[2].strip()
            if args.min_num_words is not None and len(re.split(r"\s+", sentence)) < args.min_num_words:
                continue
            sentences.append(sentence)
            if len(sentences) >= args.num_examples:
                break
    args.sentences = sentences


def get_batch_inputs_embeds(args : BatchArgs):
    print("Creating input embeds")
    tokenization = args.tokenization
    inputs_embeds = model.model.embed_tokens(tokenization['input_ids'].to(device))
    if args.randomize_input_embeds:
        no_special_token_idxs = [ i for i in range(len(full_vocab_embedding_torch)) if i not in tokenizer.all_special_ids ]
        full_vocab_embedding_no_special_tokens = full_vocab_embedding_torch[no_special_token_idxs]
        full_vocab_embedding_means = torch.mean(full_vocab_embedding_no_special_tokens, dim=0, keepdim=True)
        full_vocab_embedding_stdevs = torch.std(full_vocab_embedding_no_special_tokens - full_vocab_embedding_means, dim=0, keepdim=True)
        is_special_token_mask = torch.zeros_like(tokenization['input_ids'], dtype=torch.float, device=device)
        for special_token_id in tokenizer.all_special_ids:
            is_special_token_mask[tokenization['input_ids'] == special_token_id] = 1
        torch.manual_seed(args.permutation_seed)
        random_input_embeds = torch.randn_like(inputs_embeds, device=device) * full_vocab_embedding_stdevs.unsqueeze(0) + full_vocab_embedding_means.unsqueeze(0)
        is_special_token_mask = is_special_token_mask.unsqueeze(-1)
        inputs_embeds = random_input_embeds * (1 - is_special_token_mask) + inputs_embeds * is_special_token_mask
    args.inputs_embeds = inputs_embeds.detach().clone().cpu().numpy()

def get_batch_gradient_masks(args : BatchArgs):
    print("Creating gradient masks")
    tokenization = args.tokenization
    gradient_masks = []
    for input_ids in tokenization['input_ids']:
        gradient_mask = np.array([int(input_id not in tokenizer.all_special_ids) for input_id in input_ids.tolist()])
        gradient_masks.append(gradient_mask)
    args.gradient_masks = np.asarray(gradient_masks, dtype=float)


def learn_embeddings(args : BatchArgs) -> LearnEmbeddingsResult:

    # Prepare data
    read_sentences(args)
    tokenize_sentences(args)
    assign_target_probabilities(args)
    get_batch_inputs_embeds(args)
    get_batch_gradient_masks(args)

    # Prepare inputs for gradient computation
    inputs_embeds = nn.Parameter(torch.tensor(args.inputs_embeds, device = device), requires_grad=True)
    target_probabilities = torch.tensor(args.target_probabilities, device = device)
    gradient_masks = torch.tensor(args.gradient_masks, dtype=torch.float, device = device).unsqueeze(-1)
    args.tokenized_no_input_ids = { key: value.to(device) for (key,value) in args.tokenized_no_input_ids.items() }

    learning_rate = args.learning_rate or 1e-3

    model.eval()                              # inference mode
    model.requires_grad_(False)               # turn off grads for every param

    scaler = torch.amp.GradScaler()   # built-in utility
    opt = torch.optim.Adam([inputs_embeds], lr=learning_rate)
    ts = time.time()
    for t in range(args.steps):
        opt.zero_grad(set_to_none=True)
        with torch.autocast("mps", dtype=torch.float16):
            logits = model(**args.tokenized_no_input_ids,
                        inputs_embeds=inputs_embeds).logits[:, -1, :]
            loss = F.kl_div(F.log_softmax(logits, -1), target_probabilities, reduction='batchmean')
            loss += args.L1_embed_loss * inputs_embeds.abs().mean()
        scaler.scale(loss).backward()
        inputs_embeds.grad.mul_(gradient_masks)           # mask specials
        scaler.step(opt)
        scaler.update()

        # lightweight logging every 10 steps
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
