# dataset.py

import torch
from torch.utils.data import Dataset
from collections import defaultdict

class Tokenizer:
    """Custom tokenizer similar to Keras' Tokenizer."""
    def __init__(self, num_words=None):
        self.num_words = num_words
        self.word_counts = defaultdict(int)
        self.word_index = {}
        self.index_word = {}
        self.oov_token = '<OOV>'

    def fit_on_texts(self, texts):
        """Build vocabulary from texts."""
        for text in texts:
            for word in text.split():
                self.word_counts[word] += 1
        # Sort by frequency
        sorted_words = sorted(self.word_counts.items(), key=lambda x: x[1], reverse=True)
        if self.num_words:
            sorted_words = sorted_words[:self.num_words - 2]  # Reserve spots for PAD and OOV
        self.word_index = {word: idx + 2 for idx, (word, _) in enumerate(sorted_words)}
        self.word_index[self.oov_token] = 1
        self.word_index['<PAD>'] = 0
        self.index_word = {idx: word for word, idx in self.word_index.items()}

    def texts_to_sequences(self, texts):
        """Convert texts to sequences of indices."""
        sequences = []
        for text in texts:
            seq = []
            for word in text.split():
                idx = self.word_index.get(word, self.word_index[self.oov_token])
                seq.append(idx)
            sequences.append(seq)
        return sequences

class SentimentDataset(Dataset):
    """Custom Dataset for sentiment analysis."""
    def __init__(self, reviews, targets):
        self.reviews = torch.LongTensor(reviews)
        self.targets = torch.LongTensor(targets)

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, idx):
        return self.reviews[idx], self.targets[idx]
