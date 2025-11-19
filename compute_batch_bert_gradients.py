import os, sys, re, time
from dataclasses import dataclass
import torch.nn as nn
from typing import Optional, Any
import torch, numpy as np
from tqdm import tqdm
import time
from bert_models import (
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
    mask_positions : torch.Tensor
    N : torch.Tensor
    l1_lambda : float
    basin_loss_lambda : float
    cosine_dist_lambda : float

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
    l1_lambda : float = 0.0
    basin_loss_lambda : float = 0.0
    cosine_dist_lambda : float = 0.0
    min_num_words : Optional[int] = None
    randomize_input_embeds : bool = False

    # populated by learn_embeddings
    sentences : Optional[list[str]] = None
    tokenization : Optional[Any] = None
    mask_positions : Optional[list[int]] = None
    probabilities : Optional[list[torch.Tensor]] = None
    sentence_permutation : Optional[np.ndarray] = None
    target_probabilities : Optional[list[torch.Tensor]] = None
    tokenized_no_input_ids : Optional[Any] = None
    inputs_embeds : Optional[torch.Tensor] = None
    gradient_masks : Optional[torch.Tensor] = None

    def copy(self):
        return BatchArgs(**self.__dict__)

    def __repr__(self):
        repr_keys = ["steps", "scramble_target_probs", "learning_rate", "examples_filepath", "num_examples", "example_stride", "permutation_seed", "masked_sentences_seed", "no_ADAM", "l1_lambda", "min_num_words"]
        values = "\n,".join([f"{key} = {value}" for key, value in self.__dict__.items() if key in repr_keys])
        return f"BatchArgs(\n{values}\n)"

@dataclass
class LearnEmbeddingsResult:
    losses_list : list[float]
    reg_losses_list : list[float]
    inputs_embeds_list : list[torch.Tensor]
    last_gradient : np.ndarray


def assign_target_probabilities(args : BatchArgs):
    tokenizations = args.tokenization
    np.random.seed(args.permutation_seed)
    N = len(args.sentences)
    permutation = np.random.permutation(N)
    results = model(**tokenizations)
    probs = results.logits.softmax(dim=2)
    if args.target_probabilities is None:
        print("Shuffling token probability predictions with sentences")
        masked_token_probs = probs[torch.arange(N), torch.tensor(args.mask_positions, dtype=torch.long), :]
        masked_token_probs = masked_token_probs.detach().clone().numpy()
        target_probs = masked_token_probs[permutation,:]
    else:
        print("Canned target probabilities used instead of shuffling")
        assert args.target_probabilities.shape == (N, results.logits.shape[2])
        masked_token_probs = args.target_probabilities.detach().clone().numpy() if not isinstance(args.target_probabilities, np.ndarray) else args.target_probabilities
        permutation = np.arange(N)
        target_probs = np.copy(masked_token_probs)
    if args.scramble_target_probs:
        print("Scrambling target probabilities on vocab dimensions")
        rng = np.random.default_rng(args.permutation_seed + 1)
        rng.shuffle(target_probs, axis=1)
    args.probabilities = masked_token_probs
    args.target_probabilities = target_probs
    args.sentence_permutation = permutation


def tokenize_sentences(args : BatchArgs):

    # Run tokenization
    print(f"Tokenizing {len(args.sentences)} sentences")
    args.tokenization = tokenizer(args.sentences, return_tensors="pt", padding=True, return_attention_mask=True)

    # Pick random mask positions (which will be %'d by the number of non-special tokens)
    np.random.seed(args.masked_sentences_seed)
    random_mask_positions = np.random.randint(0, sys.maxsize, size = len(args.sentences))

    # For each sentence, pick a random token to mask if no mask token is already present
    mask_positions = []
    for i, sentence in tqdm(enumerate(args.sentences)):
        input_ids = args.tokenization['input_ids'][i,:]
        special_token_ids = set([tokenizer.mask_token_id, tokenizer.pad_token_id, tokenizer.cls_token_id, tokenizer.sep_token_id])
        non_special_token_idxs = [i for i in range(len(input_ids)) if int(input_ids[i]) not in special_token_ids]
        if tokenizer.mask_token_id in input_ids:
            mask_position = input_ids.tolist().index(tokenizer.mask_token_id)
            mask_positions.append(mask_position)
        else:
            mask_position = random_mask_positions[i] % len(non_special_token_idxs)
            real_mask_position = non_special_token_idxs[mask_position]
            mask_positions.append(real_mask_position)
            input_ids[real_mask_position] = tokenizer.mask_token_id
    args.mask_positions = mask_positions
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
        full_vocab_embedding_no_special_tokens = full_vocab_embedding_torch[2:50254]
        full_vocab_embedding_means = torch.mean(full_vocab_embedding_no_special_tokens, dim=0, keepdim=True)
        full_vocab_embedding_stdevs = torch.std(full_vocab_embedding_no_special_tokens - full_vocab_embedding_means, dim=0, keepdim=True)
        is_special_token_mask = torch.zeros_like(tokenization['input_ids'], dtype=torch.float)
        for special_token_id in set([tokenizer.mask_token_id, tokenizer.pad_token_id, tokenizer.cls_token_id, tokenizer.sep_token_id]):
            is_special_token_mask[tokenization['input_ids'] == special_token_id] = 1
        torch.manual_seed(args.permutation_seed)
        random_input_embeds = torch.randn_like(inputs_embeds) * full_vocab_embedding_stdevs.unsqueeze(0) + full_vocab_embedding_means.unsqueeze(0)
        is_special_token_mask = is_special_token_mask.unsqueeze(-1)
        inputs_embeds = random_input_embeds * (1 - is_special_token_mask) + inputs_embeds * is_special_token_mask
    args.inputs_embeds = inputs_embeds.detach().clone().numpy()

def get_batch_gradient_masks(args : BatchArgs):
    print("Creating gradient masks")
    tokenization = args.tokenization
    special_token_ids = set([tokenizer.mask_token_id, tokenizer.pad_token_id, tokenizer.cls_token_id, tokenizer.sep_token_id])
    gradient_masks = []
    for input_ids in tokenization['input_ids']:
        gradient_mask = np.array([int(input_id not in special_token_ids) for input_id in input_ids.tolist()])
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
    model_result = model(**tokenized_no_input_ids, inputs_embeds=inputs_embeds, output_attentions=True)
    masked_token_logits = model_result.logits[N, mask_positions, :]

    # Calculate cross entropy loss WRT desired probability distribution
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(masked_token_logits, target_probabilities)

    # Add L1 regularization
    l1_lambda = args.l1_lambda
    #reg_loss = l1_lambda * torch.norm(inputs_embeds - args.initial_inputs_embeds, dim = 2, p = 1).sum(axis=1).mean()
    attentions = model_result.attentions
    reg_loss = 0
    for i in [25,26,27]: #range(len(attentions)):
        attn = attentions[i].flatten()
        attn = attn[attn > 0]
        attn_entropy = -torch.sum(attn * torch.log(attn))
        reg_loss -= l1_lambda * attn_entropy
    loss = loss + reg_loss

    inputs_embeds_grad = torch.autograd.grad(
        outputs=loss,
        inputs=inputs_embeds,
        create_graph=False,  # We don't need the graph after this
        retain_graph=False,  # Don't retain the graph to free memory
        allow_unused=False
    )

    if isinstance(inputs_embeds_grad, tuple):
        inputs_embeds_grad = inputs_embeds_grad[0]

    return loss, inputs_embeds_grad, reg_loss

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
    reg_losses_list = []
    inputs_embeds_list = []

    model.eval()                              # inference mode
    model.requires_grad_(False)               # turn off grads for every param

    scaler = torch.amp.GradScaler()   # built-in utility
    opt = torch.optim.Adam([inputs_embeds], lr=learning_rate)
    ts = time.time()

    while t < args.steps:
        t += 1
        opt.zero_grad(set_to_none=True)

        with torch.autocast("mps", dtype=torch.float16):
            logits = model(**args.tokenized_no_input_ids,
                        inputs_embeds=inputs_embeds).logits[:, -1, :]
            loss = F.kl_div(F.log_softmax(logits, -1), target_probabilities, reduction='batchmean')
            reg_loss = args.L1_embed_loss * inputs_embeds.abs().mean()
            loss += reg_loss
        scaler.scale(loss).backward()
        inputs_embeds.grad.mul_(gradient_masks)           # mask specials
        scaler.step(opt)
        scaler.update()

        # Bookkeeping
        losses_list.append(loss.item())
        reg_losses_list.append(reg_loss.item())
        inputs_embeds_list.append(inputs_embeds.detach().clone().cpu().numpy())

    last_gradient = inputs_embeds.grad.detach().clone().cpu().numpy()

    return LearnEmbeddingsResult(
        losses_list = losses_list,
        reg_losses_list = reg_losses_list,
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
