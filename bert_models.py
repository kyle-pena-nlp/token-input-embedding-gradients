import os, sys
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
import sklearn.preprocessing

# Switch to True after initial model DL to avoid 429 errors from HF
LOCAL_ONLY = False

# Turn on torch compile build caching, turn off parallelism for tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Model / Model metadata
MODEL = "answerdotai/ModernBERT-large"
tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True, local_files_only=LOCAL_ONLY)
model = AutoModelForMaskedLM.from_pretrained(MODEL, trust_remote_code=True, local_files_only=LOCAL_ONLY)

# Vocab embedding (torch)
full_vocab_embedding_torch = model.get_input_embeddings().weight
vocab_embedding_no_special_tokens = full_vocab_embedding_torch[2:50254]
vocab_embeddings_norm = torch.nn.functional.normalize(vocab_embedding_no_special_tokens, p=2, dim=1)

# Vocab embedding (numpy)
full_vocab_embedding = full_vocab_embedding_torch.detach().numpy()
full_vocab_embedding_normalized = sklearn.preprocessing.normalize(full_vocab_embedding, axis = 1, norm = 'l2')
