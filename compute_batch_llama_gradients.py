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
    no_ADAM : bool = False
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
    args.tokenization = { key: value.to(device) for (key,value) in args.tokenization.items() }
    if args.trim_input_ids:
        args.tokenization['input_ids'] = args.tokenization['input_ids'][:,:-3] # Trim so there is something meaningful to predict.
    args.tokenized_no_input_ids = { key: value for (key,value) in args.tokenization.items() if key != "input_ids"}



def read_sentences(args : BatchArgs):
    if isinstance(args.examples_filepath, list) and len(args.examples_filepath) > 0 and all([isinstance(x, str) for x in args.examples_filepath]):
        args.sentences = args.examples_filepath
        return
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
    inputs_embeds = model.model.embed_tokens(tokenization['input_ids'])
    if args.randomize_input_embeds:
        no_special_token_idxs = [ i for i in range(len(full_vocab_embedding_torch)) if i not in tokenizer.all_special_ids ]
        full_vocab_embedding_no_special_tokens = full_vocab_embedding_torch[no_special_token_idxs]
        full_vocab_embedding_means = torch.mean(full_vocab_embedding_no_special_tokens, dim=0, keepdim=True)
        full_vocab_embedding_stdevs = torch.std(full_vocab_embedding_no_special_tokens - full_vocab_embedding_means, dim=0, keepdim=True)
        is_special_token_mask = torch.zeros_like(tokenization['input_ids'], dtype=torch.float)
        for special_token_id in tokenizer.all_special_ids:
            is_special_token_mask[tokenization['input_ids'] == special_token_id] = 1
        torch.manual_seed(args.permutation_seed)
        random_input_embeds = torch.randn_like(inputs_embeds) * full_vocab_embedding_stdevs.unsqueeze(0) + full_vocab_embedding_means.unsqueeze(0)
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

def compute_batch_inputs_embeds_gradients(args : BatchInputsEmbedsGradArgs):

    # Use supplied input_embeds if availalbe, otherwise, compute from input ids
    tokenized_no_input_ids = args.tokenized_no_input_ids
    inputs_embeds = args.inputs_embeds
    target_probabilities = args.target_probabilities
    #gradient_masks = args.gradient_masks
    B  = args.N # B for "batch dimension"

    # Run a forward pass on the model
    model_result = model(**tokenized_no_input_ids, inputs_embeds=inputs_embeds)
    next_token_logits = model_result.logits[:,-1,:]

    # Calculate cross entropy loss WRT desired probability distribution
    #loss_fn = nn.CrossEntropyLoss()
    #raw_loss = loss_fn(next_token_logits, target_probabilities)

    # Calculate KL divergence loss between model output and target probabilities
    next_token_log_probs = F.log_softmax(next_token_logits, dim=-1)
    raw_loss = F.kl_div(next_token_log_probs, target_probabilities, reduction='batchmean')


    # Add embedding diff entropy regularization
    embed_L1_loss = args.L1_embed_loss * torch.abs(inputs_embeds).mean()

    # Add loss for sequence plausibility
    # This basically says that tokens that the input embedding is similar to should also have higher probabilities
    # We treat the cosine similarities like logits
    # input_embeds: [B,N,D]
    # vocab_embeds: [V,D]
    #input_embeds_normalized = torch.nn.functional.normalize(inputs_embeds, p=2, dim=-1)
    #input_embed_vocab_cosine_sims = torch.einsum("bnd,vd->bnv", input_embeds_normalized, full_vocab_embedding_torch_norm) # [B,N,V]
    #flattened_input_embed_vocab_cosine_sims = input_embed_vocab_cosine_sims.flatten(0,1)


    # Here follows several failed attempts at "sequence plausibility loss"

    #flattened_seq_probabilities = model_result.logits.log_softmax(dim=-1).flatten(0,1)
    #seq_plausibility_loss = F.kl_div(F.log_softmax(flattened_input_embed_vocab_cosine_sims, dim=-1), flattened_seq_probabilities, reduction='batchmean', log_target=True) # [B*N]

    #flattened_seq_probabilities = model_result.logits.log_softmax(dim=-1).flatten(0,1)
    #seq_plausibility_loss = torch.nn.functional.cross_entropy(flattened_input_embed_vocab_cosine_sims, torch.argmax(flattened_seq_probabilities, dim=-1), reduction='mean')

    #flattened_seq_probabilities = model_result.logits.softmax(dim=-1).flatten(0,1)
    #seq_plausibility_loss = torch.nn.functional.cross_entropy(flattened_input_embed_vocab_cosine_sims, flattened_seq_probabilities, reduction='mean')

    #flattened_seq_probabilities = model_result.logits.log_softmax(dim=-1).flatten(0,1)
    #seq_plausibility_loss = -torch.sum(flattened_input_embed_vocab_cosine_sims.softmax(dim=-1) * flattened_seq_probabilities * gradient_masks) / torch.sum(gradient_masks)

    #seq_plausibility_loss = (gradient_masks * dist.Categorical(logits = flattened_input_embed_vocab_cosine_sims).entropy()).sum() / torch.sum(gradient_masks)
    #seq_plausibility_loss = (dist.Categorical(logits = flattened_input_embed_vocab_cosine_sims).entropy()).mean()

    #seq_plausibility_loss = args.seq_plausibility_loss * seq_plausibility_loss

    reg_loss = embed_L1_loss #+ seq_plausibility_loss
    loss = raw_loss + reg_loss


    inputs_embeds_grad = torch.autograd.grad(
        outputs=loss,
        inputs=inputs_embeds,
        create_graph=False,  # We don't need the graph after this
        retain_graph=False,  # Don't retain the graph to free memory
        allow_unused=False
    )

    if isinstance(inputs_embeds_grad, tuple):
        inputs_embeds_grad = inputs_embeds_grad[0]

    print("loss", loss.item(), "raw_loss", raw_loss.item(), "embed_L1_loss", embed_L1_loss.item()) # , "seq_plausibility_loss", seq_plausibility_loss.item(

    return loss, inputs_embeds_grad


def learn_embeddings(args : BatchArgs) -> LearnEmbeddingsResult:

    # Prepare data
    read_sentences(args)
    tokenize_sentences(args)
    assign_target_probabilities(args)
    get_batch_inputs_embeds(args)
    get_batch_gradient_masks(args)

    # Prepare inputs for gradient computation
    inputs_embeds = torch.tensor(args.inputs_embeds, device=device).requires_grad_(True)
    initial_inputs_embeds = torch.tensor(args.inputs_embeds, device=device).detach().clone()
    target_probabilities = torch.tensor(args.target_probabilities, device=device)
    gradient_masks = torch.tensor(args.gradient_masks, dtype=torch.float, device=device).unsqueeze(-1)
    N = torch.arange(len(args.sentences), device=device)

    # Prepare ADAM optimizer
    if not args.no_ADAM:
        B1, B2, m, v = 0.9, 0.999, torch.zeros(inputs_embeds.shape, requires_grad = False, device=device), torch.zeros(inputs_embeds.shape, requires_grad = False, device=device)
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
            N = N,
            seq_plausibility_loss = args.seq_plausibility_loss,
            L1_embed_loss = args.L1_embed_loss,
            gradient_masks = gradient_masks
        ))

        if loss.item() < args.early_stopping_threshold:
            break
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

        # Bookkeeping
        losses_list.append(loss.item())
        inputs_embeds_list.append(inputs_embeds.detach().clone().cpu().numpy())

    last_gradient = gradient.detach().clone().cpu().numpy()

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
