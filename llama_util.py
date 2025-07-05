
import numpy as np
import sklearn.preprocessing
from compute_batch_bert_gradients import BatchArgs
from llama_models import (
    device,
    model,
    tokenizer,
    full_vocab_embedding_normalized
)
from tqdm import tqdm
import torch
import scipy.stats

def collect_probability_path(args : BatchArgs, input_embeds_list : list[np.ndarray]):
    # Assemble path of the predicted token probabilities
    target_probs = args.target_probabilities
    probs_path = np.zeros((len(input_embeds_list), target_probs.shape[0], target_probs.shape[-1]))
    for i in tqdm(range(len(input_embeds_list))):
        p = calculate_token_probabilities(args, torch.Tensor(input_embeds_list[i]))
        probs_path[i,:] = p
    return probs_path

def set_temperature(dist, temperature):
    # Convert probabilities back to logits (inverse of softmax)
    logits = np.log(dist)

    # Apply the new temperature
    logits_temp = logits / temperature
    # Subtract max for numerical stability
    logits_temp = logits_temp - np.max(logits_temp, axis=-1, keepdims=True)
    exp_logits = np.exp(logits_temp)
    new_probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

    return new_probs

def find_nearest_tokens_cosine_similarity(embeddings : np.ndarray, sentence_idx : int):
    sentence_embedding = embeddings[sentence_idx]
    normalized_gradient = sklearn.preprocessing.normalize(sentence_embedding,axis=1) # [N,D]
    cosine_sims = np.einsum('vd,nd->nv',full_vocab_embedding_normalized,normalized_gradient)
    closest_token_ids = np.argmax(cosine_sims,-1)



def clean_token(token):
    return token.replace("Ġ", " ")



def invert_embeddings(inputs_embeds):
    norm = np.linalg.norm(inputs_embeds, axis=2, keepdims=True)
    eps = 1e-15
    inputs_embeds_normalized = inputs_embeds / np.maximum(norm, eps)
    probs = np.einsum('bnd,vd->bnv', inputs_embeds_normalized, full_vocab_embedding_normalized)
    nearest_id = np.argmax(probs, axis=2)
    tokens_list = []
    for i in range(len(nearest_id)):
        tokens = tokenizer.convert_ids_to_tokens(nearest_id[i])
        tokens_list.append(tokens)
    return nearest_id, tokens_list

def calculate_token_probabilities(args : BatchArgs, inputs_embeds : torch.Tensor):
    if not isinstance(inputs_embeds, torch.Tensor):
        inputs_embeds = torch.Tensor(inputs_embeds)
    model_result = model(**args.tokenized_no_input_ids, inputs_embeds=inputs_embeds) # [B,N,V]
    probs = model_result.logits.softmax(dim=2)
    masked_token_probs = probs[torch.arange(len(probs)), -1, :]
    return masked_token_probs.detach().clone().numpy()

def get_token_distribution(sentences : list[str]):
    tokenized = tokenizer(sentences, return_tensors="pt")
    tokenized = { key: value.to(device) for (key,value) in tokenized.items() }
    model_result = model(**tokenized)
    probs = model_result.logits[:, -1, :].softmax(dim=1)
    return probs.detach().clone().cpu().numpy()

def compute_divergence(args: BatchArgs, input_embeds : np.ndarray, target_probs : np.ndarray):
    if not isinstance(input_embeds, torch.Tensor):
        input_embeds = torch.Tensor(input_embeds)
    model_result = model(**args.tokenized_no_input_ids, inputs_embeds=input_embeds) # [B,N,V]
    probs = model_result.logits[:, -1, :].softmax(dim=1).detach().numpy() # [B,V]
    return scipy.stats.entropy(target_probs, probs, axis=1)
