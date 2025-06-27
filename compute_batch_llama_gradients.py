import os, sys, re, time
from dataclasses import dataclass
import torch.nn as nn
from typing import Optional, Any
import torch, numpy as np
from tqdm import tqdm
import time
from llama_models import (
    model,
    tokenizer,
    full_vocab_embedding,
    full_vocab_embedding_torch,
    vocab_embedding_no_special_tokens,
    vocab_embeddings_norm
)

@dataclass
class BatchInputsEmbedsGradArgs:
    tokenized_no_input_ids : Any
    initial_inputs_embeds : torch.Tensor
    inputs_embeds : torch.Tensor
    target_probabilities : torch.Tensor
    N : torch.Tensor

@dataclass
class BatchArgs:
    steps: int = 100
    scramble_target_probs : bool = False
    learning_rate : float = 1e-3
    examples_filepath : str = "rope_move/eng_sentences.tsv"
    num_examples : int = 32
    example_stride : int = 50
    permutation_seed : int = 42
    masked_sentences_seed : int = 42
    no_ADAM : bool = False
    min_num_words : Optional[int] = None
    randomize_input_embeds : bool = False

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
        repr_keys = ["steps", "learning_rate", "examples_filepath", "num_examples", "example_stride", "permutation_seed", "masked_sentences_seed", "no_ADAM", "min_num_words", "randomize_input_embeds"]
        values = "\n,".join([f"{key} = {value}" for key, value in self.__dict__.items() if key in repr_keys])
        return f"BatchArgs(\n{values}\n)"

@dataclass
class LearnEmbeddingsResult:
    losses_list : list[float]
    inputs_embeds_list : list[np.ndarray]
    last_gradient : np.ndarray


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
        target_probs = results.logits.softmax(dim=2)
        target_probs = target_probs[permutation]

    if args.scramble_target_probs:
        print("Scrambling target probabilities on vocab dimensions")
        rng = np.random.default_rng(args.permutation_seed + 1)
        rng.shuffle(target_probs, axis=1)

    target_probs = target_probs.detach().clone().numpy()
    args.target_probabilities = target_probs
    args.sentence_permutation = permutation


def tokenize_sentences(args : BatchArgs):
    # Run tokenization
    print(f"Tokenizing {len(args.sentences)} sentences")
    args.tokenization = tokenizer(args.sentences, return_tensors="pt", padding=True)
    args.tokenized_no_input_ids = { key: value for (key,value) in args.tokenization.items() if key != "input_ids"}



def read_sentences(args : BatchArgs):
    if isinstance(args.examples_filepath, list):
        return args.examples_filepath
    sentences = []
    with open(args.examples_filepath, "r") as f:
        print("Reading sentences")
        i = 0
        for line in tqdm(f):
            if line.startswith("#"):
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
    inputs_embeds = model.model.embeddings.tok_embeddings(tokenization['input_ids'])
    if args.randomize_input_embeds:
        no_special_token_idxs = [ i for i in range(len(full_vocab_embedding_torch)) if i not in tokenizer.all_special_tokens ]
        full_vocab_embedding_no_special_tokens = full_vocab_embedding_torch[no_special_token_idxs]
        full_vocab_embedding_means = torch.mean(full_vocab_embedding_no_special_tokens, dim=0, keepdim=True)
        full_vocab_embedding_stdevs = torch.std(full_vocab_embedding_no_special_tokens - full_vocab_embedding_means, dim=0, keepdim=True)
        is_special_token_mask = torch.zeros_like(tokenization['input_ids'], dtype=torch.float)
        for special_token_id in tokenizer.all_special_tokens:
            is_special_token_mask[tokenization['input_ids'] == special_token_id] = 1
        torch.manual_seed(args.permutation_seed)
        random_input_embeds = torch.randn_like(inputs_embeds) * full_vocab_embedding_stdevs.unsqueeze(0) + full_vocab_embedding_means.unsqueeze(0)
        is_special_token_mask = is_special_token_mask.unsqueeze(-1)
        inputs_embeds = random_input_embeds * (1 - is_special_token_mask) + inputs_embeds * is_special_token_mask
    args.inputs_embeds = inputs_embeds.detach().clone().numpy()

def get_batch_gradient_masks(args : BatchArgs):
    print("Creating gradient masks")
    tokenization = args.tokenization
    gradient_masks = []
    for input_ids in tokenization['input_ids']:
        gradient_mask = np.array([int(input_id not in tokenizer.all_special_tokens) for input_id in input_ids.tolist()])
        gradient_masks.append(gradient_mask)
    args.gradient_masks = np.asarray(gradient_masks, dtype=float)

def compute_batch_inputs_embeds_gradients(args : BatchInputsEmbedsGradArgs):

    # Use supplied input_embeds if availalbe, otherwise, compute from input ids
    tokenized_no_input_ids = args.tokenized_no_input_ids
    inputs_embeds = args.inputs_embeds
    target_probabilities = args.target_probabilities
    mask_positions = args.mask_positions
    N = args.N

    # Run a forward pass on the model
    model_result = model(**tokenized_no_input_ids, inputs_embeds=inputs_embeds)
    masked_token_logits = model_result.logits[N, mask_positions, :]

    # Calculate cross entropy loss WRT desired probability distribution
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(masked_token_logits, target_probabilities)

    inputs_embeds_grad = torch.autograd.grad(
        outputs=loss,
        inputs=inputs_embeds,
        create_graph=False,  # We don't need the graph after this
        retain_graph=False,  # Don't retain the graph to free memory
        allow_unused=False
    )

    if isinstance(inputs_embeds_grad, tuple):
        inputs_embeds_grad = inputs_embeds_grad[0]

    return loss, inputs_embeds_grad

def learn_embeddings(args : BatchArgs) -> LearnEmbeddingsResult:

    # Prepare data
    read_sentences(args)
    tokenize_sentences(args)
    assign_target_probabilities(args)
    get_batch_inputs_embeds(args)
    get_batch_gradient_masks(args)

    # Prepare inputs for gradient computation
    inputs_embeds = torch.Tensor(args.inputs_embeds).requires_grad_(True)
    initial_inputs_embeds = torch.Tensor(args.inputs_embeds).detach().clone()
    target_probabilities = torch.Tensor(args.target_probabilities)
    mask_positions = torch.tensor(args.mask_positions, dtype=torch.long)
    gradient_masks = torch.tensor(args.gradient_masks, dtype=torch.float).unsqueeze(-1)
    N = torch.arange(len(args.sentences))

    # Prepare ADAM optimizer
    if not args.no_ADAM:
        B1, B2, m, v = 0.9, 0.999, torch.zeros(inputs_embeds.shape, requires_grad = False, device="cpu"), torch.zeros(inputs_embeds.shape, requires_grad = False, device="cpu")
    learning_rate = args.learning_rate or 1e-3

    fn = timer(compute_batch_inputs_embeds_gradients)

    t = 0
    losses_list = []
    inputs_embeds_list = []
    while t < args.steps:
        t += 1
        loss, gradient = fn(BatchInputsEmbedsGradArgs(
            tokenized_no_input_ids = args.tokenized_no_input_ids,
            initial_inputs_embeds = initial_inputs_embeds,
            inputs_embeds = inputs_embeds,
            target_probabilities = target_probabilities,
            mask_positions = mask_positions,
            N = N,
            l1_lambda = args.l1_lambda,
            basin_loss_lambda = args.basin_loss_lambda,
            cosine_dist_lambda = args.cosine_dist_lambda
        ))
        with torch.no_grad():
            if args.no_ADAM:
                inputs_embeds -= args.learning_rate * gradient * gradient_masks
            else:
                m = B1 * m + (1 - B1) * gradient
                v = B2 * v + (1 - B2) * gradient**2
                m_hat = m / (1 - B1**t)
                v_hat = v / (1 - B2**t)
                inputs_embeds -= gradient_masks * learning_rate * m_hat / (torch.sqrt(v_hat) + 1e-8)
            inputs_embeds.grad = None
        print(t, loss.item())

        # Bookkeeping
        losses_list.append(loss.item())
        inputs_embeds_list.append(inputs_embeds.detach().clone().numpy())

    last_gradient = gradient.detach().clone().numpy()

    return LearnEmbeddingsResult(
        losses_list = losses_list,
        inputs_embeds_list = inputs_embeds_list,
        last_gradient = last_gradient
    )


def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__ if hasattr(func, '__name__') else func.__class__.__name__} took {end - start} seconds")
        return result
    return wrapper
