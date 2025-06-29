import os, sys
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import sklearn.preprocessing

# Switch to True after initial model DL to avoid 429 errors from HF
LOCAL_ONLY = False

# Turn on torch compile build caching, turn off parallelism for tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Model / Model metadata
MODEL = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True, local_files_only=LOCAL_ONLY)
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(MODEL, trust_remote_code=True, local_files_only=LOCAL_ONLY)

# Vocab embedding (torch)
full_vocab_embedding_torch = model.model.embed_tokens.weight

no_special_token_mask = torch.ones(full_vocab_embedding_torch.shape[0], dtype=torch.bool)
for special_token_id in tokenizer.all_special_ids:
    no_special_token_mask[special_token_id] = False

vocab_embedding_no_special_tokens = full_vocab_embedding_torch[no_special_token_mask]
vocab_embeddings_no_special_tokens_norm = torch.nn.functional.normalize(vocab_embedding_no_special_tokens, p=2, dim=1)

# Vocab embedding (numpy)
full_vocab_embedding = full_vocab_embedding_torch.detach().numpy()
full_vocab_embedding_normalized = sklearn.preprocessing.normalize(full_vocab_embedding, axis = 1, norm = 'l2')

np_vocab_embedding_no_special_tokens = full_vocab_embedding[no_special_token_mask.detach().numpy()]
np_vocab_embeddings_no_special_tokens_norm = sklearn.preprocessing.normalize(np_vocab_embedding_no_special_tokens, axis = 1, norm = 'l2')
