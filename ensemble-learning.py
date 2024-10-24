import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from collections import Counter, defaultdict
from sklearn.model_selection import StratifiedKFold
import re
import string
import random

def main():
    # Set random seeds for reproducibility
    def set_seed(seed=42):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if torch.backends.mps.is_available():
            torch.use_deterministic_algorithms(True)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

    # Check device
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Parameters
    MAX_NUM_WORDS = 20000  # Maximum number of words to keep, based on word frequency
    MAX_SEQUENCE_LENGTH = 100  # Maximum sequence length (in words)
    EMBEDDING_DIM = 100  # Dimension of the embedding vector
    BATCH_SIZE = 32
    EPOCHS = 15
    LEARNING_RATE = 1e-4

    # Load Data
    train_df = pd.read_csv('data/ueh-ecom/train.csv')
    test_df = pd.read_csv('data/ueh-ecom/test.csv')

    # Ensure 'review' and 'target' columns exist
    assert 'review' in train_df.columns and 'target' in train_df.columns, "train.csv must contain 'review' and 'target' columns."
    assert 'review' in test_df.columns and 'target' in test_df.columns, "test.csv must contain 'review' and 'target' columns."

    # Combine train and test data for splitting
    full_df = pd.concat([train_df, test_df]).reset_index(drop=True)

    # Preprocessing
    def clean_text(text):
        text = text.lower()
        text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
        text = re.sub(r'\d+', '', text)
        text = text.strip()
        return text

    # Tokenization and Vocabulary Building
    class Tokenizer:
        def __init__(self, num_words=None):
            self.num_words = num_words
            self.word_counts = defaultdict(int)
            self.word_index = {}
            self.index_word = {}
            self.oov_token = '<OOV>'

        def fit_on_texts(self, texts):
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
            sequences = []
            for text in texts:
                seq = []
                for word in text.split():
                    idx = self.word_index.get(word, self.word_index[self.oov_token])
                    seq.append(idx)
                sequences.append(seq)
            return sequences

    # Padding sequences
    def pad_sequences_custom(sequences, maxlen):
        padded = []
        for seq in sequences:
            if len(seq) < maxlen:
                padded_seq = seq + [0] * (maxlen - len(seq))
            else:
                padded_seq = seq[:maxlen]
            padded.append(padded_seq)
        return np.array(padded)

    # Create Dataset
    class SentimentDataset(Dataset):
        def __init__(self, reviews, targets):
            self.reviews = torch.LongTensor(reviews)
            self.targets = torch.LongTensor(targets)

        def __len__(self):
            return len(self.reviews)

        def __getitem__(self, idx):
            return self.reviews[idx], self.targets[idx]

    # Define Models

    # 1. RNN Model
    class RNNModel(nn.Module):
        def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, num_layers=2):
            super(RNNModel, self).__init__()
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            self.rnn = nn.RNN(embed_dim, hidden_dim, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_dim, output_dim)

        def forward(self, x):
            embedded = self.embedding(x)
            output, hidden = self.rnn(embedded)
            out = self.fc(hidden[-1])
            return out

    # 2. LSTM Model
    class LSTMModel(nn.Module):
        def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, num_layers=2):
            super(LSTMModel, self).__init__()
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_dim, output_dim)

        def forward(self, x):
            embedded = self.embedding(x)
            output, (hidden, cell) = self.lstm(embedded)
            out = self.fc(hidden[-1])
            return out

    # 3. BiLSTM Model
    class BiLSTMModel(nn.Module):
        def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, num_layers=2):
            super(BiLSTMModel, self).__init__()
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            self.bilstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
            self.fc = nn.Linear(hidden_dim * 2, output_dim)

        def forward(self, x):
            embedded = self.embedding(x)
            output, (hidden, cell) = self.bilstm(embedded)
            out = self.fc(torch.cat((hidden[-2], hidden[-1]), dim=1))
            return out

    # 4. TextCNN Model
    class TextCNN(nn.Module):
        def __init__(self, vocab_size, embed_dim, num_classes, kernel_sizes=[3,4,5], num_channels=100, dropout=0.5):
            super(TextCNN, self).__init__()
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            self.convs = nn.ModuleList([
                nn.Conv2d(1, num_channels, (k, embed_dim)) for k in kernel_sizes
            ])
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Linear(len(kernel_sizes) * num_channels, num_classes)

        def forward(self, x):
            embedded = self.embedding(x)  # [batch_size, seq_len, embed_dim]
            embedded = embedded.unsqueeze(1)  # [batch_size, 1, seq_len, embed_dim]
            conved = [torch.relu(conv(embedded)).squeeze(3) for conv in self.convs]  # list of [batch_size, num_channels, seq_len - k +1]
            pooled = [torch.max(c, dim=2)[0] for c in conved]  # list of [batch_size, num_channels]
            cat = torch.cat(pooled, dim=1)  # [batch_size, num_channels * len(kernel_sizes)]
            dropped = self.dropout(cat)
            out = self.fc(dropped)
            return out

    # 5. gMLP Model
    class gMLPBlock(nn.Module):
        def __init__(self, dim, seq_len):
            super(gMLPBlock, self).__init__()
            self.norm = nn.LayerNorm(dim)
            self.fc1 = nn.Linear(dim, dim * 4)
            self.gate = nn.Parameter(torch.randn(1, seq_len, dim * 4))
            self.act = nn.GELU()
            self.fc2 = nn.Linear(dim * 4, dim)

        def forward(self, x):
            y = self.norm(x)
            y = self.fc1(y)
            y = y * self.gate  # Broadcasting over batch dimension
            y = self.act(y)
            y = self.fc2(y)
            return x + y

    class gMLPModel(nn.Module):
        def __init__(self, vocab_size, embed_dim, num_classes, seq_len, num_blocks=3, dropout=0.5):
            super(gMLPModel, self).__init__()
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            self.dropout = nn.Dropout(dropout)
            self.blocks = nn.Sequential(
                *[gMLPBlock(embed_dim, seq_len) for _ in range(num_blocks)]
            )
            self.fc = nn.Linear(embed_dim, num_classes)

        def forward(self, x):
            x = self.embedding(x)  # [batch_size, seq_len, embed_dim]
            x = self.dropout(x)
            x = self.blocks(x)
            x = x.mean(dim=1)  # Global average pooling
            x = self.fc(x)
            return x

    # Function to count parameters
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Training Function
    def train_model(model, optimizer, train_loader, epochs, model_name):
        model.train()
        for epoch in range(epochs):
            total_loss = 0
            for reviews, targets in train_loader:
                reviews, targets = reviews.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(reviews)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_train_loss = total_loss / len(train_loader)
            print(f'{model_name} Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}')

    # Prediction Function
    def get_predictions(model, loader):
        model.eval()
        all_preds = []
        with torch.no_grad():
            for reviews, _ in loader:
                reviews = reviews.to(device)
                outputs = model(reviews)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
        return np.array(all_preds)

    # Ensemble with Majority Voting
    def majority_vote(preds_dict):
        preds = np.stack([preds for preds in preds_dict.values()], axis=1)
        ensemble_preds = []
        for instance in preds:
            most_common = Counter(instance).most_common(1)[0][0]
            ensemble_preds.append(most_common)
        return np.array(ensemble_preds)

    # Initialize Metrics Dictionary
    metrics = {'Ensemble': {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}}

    # Number of folds
    n_splits = 10

    # Initialize StratifiedKFold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    for fold, (train_index, test_index) in enumerate(skf.split(full_df['review'], full_df['target'])):
        print(f'\n--- Fold {fold+1}/{n_splits} ---')
        set_seed(fold)  # Use fold as seed for reproducibility

        # Get train and test splits
        train_df_split = full_df.iloc[train_index].reset_index(drop=True)
        test_df_split = full_df.iloc[test_index].reset_index(drop=True)

        # Preprocessing
        train_df_split['review'] = train_df_split['review'].apply(clean_text)
        test_df_split['review'] = test_df_split['review'].apply(clean_text)

        # Initialize and fit tokenizer
        tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
        tokenizer.fit_on_texts(train_df_split['review'])

        # Convert texts to sequences
        X_train = tokenizer.texts_to_sequences(train_df_split['review'])
        X_test = tokenizer.texts_to_sequences(test_df_split['review'])

        # Padding sequences
        X_train = pad_sequences_custom(X_train, MAX_SEQUENCE_LENGTH)
        X_test = pad_sequences_custom(X_test, MAX_SEQUENCE_LENGTH)

        # Encode targets
        label_encoder = LabelEncoder()
        y_train = label_encoder.fit_transform(train_df_split['target'])
        y_test = label_encoder.transform(test_df_split['target'])

        # Create Dataset
        train_dataset = SentimentDataset(X_train, y_train)
        test_dataset = SentimentDataset(X_test, y_test)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

        # Initialize Models
        vocab_size = min(MAX_NUM_WORDS, len(tokenizer.word_index)) + 1  # +1 for padding
        num_classes = len(label_encoder.classes_)

        models = {}

        rnn_model = RNNModel(vocab_size, EMBEDDING_DIM, hidden_dim=128, output_dim=num_classes).to(device)
        lstm_model = LSTMModel(vocab_size, EMBEDDING_DIM, hidden_dim=128, output_dim=num_classes).to(device)
        bilstm_model = BiLSTMModel(vocab_size, EMBEDDING_DIM, hidden_dim=128, output_dim=num_classes).to(device)
        textcnn_model = TextCNN(vocab_size, EMBEDDING_DIM, num_classes=num_classes).to(device)
        gmlp_model = gMLPModel(vocab_size, EMBEDDING_DIM, num_classes=num_classes, seq_len=MAX_SEQUENCE_LENGTH).to(device)

        models = {
            'RNN': rnn_model,
            'LSTM': lstm_model,
            'BiLSTM': bilstm_model,
            'TextCNN': textcnn_model,
            'gMLP': gmlp_model
        }

        # Initialize Metrics for this fold
        for name in models.keys():
            if name not in metrics:
                metrics[name] = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}

        # Define Loss and Optimizer
        criterion = nn.CrossEntropyLoss()

        optimizers = {
            name: optim.AdamW(model.parameters(), lr=LEARNING_RATE) for name, model in models.items()
        }

        # Train all models
        for name, model in models.items():
            print(f'\nTraining {name} model...')
            optimizer = optimizers[name]
            train_model(model, optimizer, train_loader, EPOCHS, name)

        # Get predictions from each model
        predictions = {}
        for name, model in models.items():
            print(f'Predicting with {name} model...')
            preds = get_predictions(model, test_loader)
            predictions[name] = preds

        # Ensemble Predictions
        ensemble_preds = majority_vote(predictions)

        # Evaluate Ensemble
        y_true = y_test

        ensemble_accuracy = accuracy_score(y_true, ensemble_preds)
        ensemble_precision = precision_score(y_true, ensemble_preds, zero_division=0)
        ensemble_recall = recall_score(y_true, ensemble_preds, zero_division=0)
        ensemble_f1 = f1_score(y_true, ensemble_preds, zero_division=0)

        metrics['Ensemble']['accuracy'].append(ensemble_accuracy)
        metrics['Ensemble']['precision'].append(ensemble_precision)
        metrics['Ensemble']['recall'].append(ensemble_recall)
        metrics['Ensemble']['f1'].append(ensemble_f1)

        # Evaluate Individual Models
        print("\nIndividual Models Performance on Test Set:")
        for name, preds in predictions.items():
            acc = accuracy_score(y_true, preds)
            prec = precision_score(y_true, preds, zero_division=0)
            rec = recall_score(y_true, preds, zero_division=0)
            f1_s = f1_score(y_true, preds, zero_division=0)
            metrics[name]['accuracy'].append(acc)
            metrics[name]['precision'].append(prec)
            metrics[name]['recall'].append(rec)
            metrics[name]['f1'].append(f1_s)
            print(f'{name} - Acc: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1_s:.4f}')

        # Ensemble Performance
        print("\nEnsemble Model Performance on Test Set:")
        print(f'Accuracy : {ensemble_accuracy:.4f}')
        print(f'Precision: {ensemble_precision:.4f}')
        print(f'Recall   : {ensemble_recall:.4f}')
        print(f'F1 Score : {ensemble_f1:.4f}')

    # Compute Mean and Variance of Metrics
    print("\n--- Overall Performance across all folds ---")
    for name in metrics.keys():
        print(f'\n{name} Model Performance over {n_splits} folds:')
        for metric_name in metrics[name]:
            values = metrics[name][metric_name]
            mean_val = np.mean(values)
            var_val = np.var(values)
            print(f'{metric_name.capitalize()}: Mean={mean_val:.4f}, Variance={var_val:.6f}')

if __name__ == '__main__':
    main()
