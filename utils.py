# utils.py

import torch
import numpy as np
import random
import re
import string

def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def clean_text(text):
    """Preprocess text by lowercasing, removing punctuation and numbers."""
    text = text.lower()
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.strip()
    return text

def pad_sequences_custom(sequences, maxlen):
    """Pad sequences to a maximum length."""
    padded = []
    for seq in sequences:
        if len(seq) < maxlen:
            padded_seq = seq + [0] * (maxlen - len(seq))
        else:
            padded_seq = seq[:maxlen]
        padded.append(padded_seq)
    return np.array(padded)
