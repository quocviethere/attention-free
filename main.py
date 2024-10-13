# main.py

import argparse
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from utils import set_seed, clean_text, pad_sequences_custom
from dataset import Tokenizer, SentimentDataset
from models import RNNModel, LSTMModel, BiLSTMModel, TextCNN, gMLPModel, Mamba2ForSentimentAnalysis
from train import train_model
from evaluate import get_predictions, majority_vote, evaluate_model

def main(args):
    # Set random seed
    set_seed(args.seed)
    
    # Check device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load Data
    train_df = pd.read_csv(args.train_file)
    test_df = pd.read_csv(args.test_file)
    
    # Ensure 'review' and 'target' columns exist
    assert 'review' in train_df.columns and 'target' in train_df.columns, f"{args.train_file} must contain 'review' and 'target' columns."
    assert 'review' in test_df.columns and 'target' in test_df.columns, f"{args.test_file} must contain 'review' and 'target' columns."
    
    # Preprocessing
    train_df['review'] = train_df['review'].apply(clean_text)
    test_df['review'] = test_df['review'].apply(clean_text)
    
    # Initialize and fit tokenizer
    tokenizer = Tokenizer(num_words=args.max_num_words)
    tokenizer.fit_on_texts(train_df['review'])
    
    # Convert texts to sequences
    X_train = tokenizer.texts_to_sequences(train_df['review'])
    X_test = tokenizer.texts_to_sequences(test_df['review'])
    
    # Padding sequences
    X_train = pad_sequences_custom(X_train, args.max_sequence_length)
    X_test = pad_sequences_custom(X_test, args.max_sequence_length)
    
    # Encode targets
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_df['target'])
    y_test = label_encoder.transform(test_df['target'])
    
    # Create Dataset
    # Split training data into train and validation
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=args.seed, stratify=y_train
    )
    
    train_dataset = SentimentDataset(X_train_split, y_train_split)
    val_dataset = SentimentDataset(X_val, y_val)
    test_dataset = SentimentDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    # Initialize Models
    vocab_size = min(args.max_num_words, len(tokenizer.word_index)) + 1  # +1 for padding
    num_classes = len(label_encoder.classes_)
    
    models_dict = {}
    optimizers = {}
    
    if 'RNN' in args.models_to_train:
        rnn_model = RNNModel(vocab_size, args.embedding_dim, hidden_dim=128, output_dim=num_classes).to(device)
        optimizer_rnn = torch.optim.AdamW(rnn_model.parameters(), lr=args.learning_rate)
        models_dict['RNN'] = rnn_model
        optimizers['RNN'] = optimizer_rnn
        
    if 'LSTM' in args.models_to_train:
        lstm_model = LSTMModel(vocab_size, args.embedding_dim, hidden_dim=128, output_dim=num_classes).to(device)
        optimizer_lstm = torch.optim.AdamW(lstm_model.parameters(), lr=args.learning_rate)
        models_dict['LSTM'] = lstm_model
        optimizers['LSTM'] = optimizer_lstm
        
    if 'BiLSTM' in args.models_to_train:
        bilstm_model = BiLSTMModel(vocab_size, args.embedding_dim, hidden_dim=128, output_dim=num_classes).to(device)
        optimizer_bilstm = torch.optim.AdamW(bilstm_model.parameters(), lr=args.learning_rate)
        models_dict['BiLSTM'] = bilstm_model
        optimizers['BiLSTM'] = optimizer_bilstm
        
    if 'TextCNN' in args.models_to_train:
        textcnn_model = TextCNN(vocab_size, args.embedding_dim, num_classes=num_classes).to(device)
        optimizer_textcnn = torch.optim.AdamW(textcnn_model.parameters(), lr=args.learning_rate)
        models_dict['TextCNN'] = textcnn_model
        optimizers['TextCNN'] = optimizer_textcnn
        
    if 'gMLP' in args.models_to_train:
        gmlp_model = gMLPModel(vocab_size, args.embedding_dim, num_classes=num_classes, seq_len=args.max_sequence_length).to(device)
        optimizer_gmlp = torch.optim.AdamW(gmlp_model.parameters(), lr=args.learning_rate)
        models_dict['gMLP'] = gmlp_model
        optimizers['gMLP'] = optimizer_gmlp
        
    if 'Mamba' in args.models_to_train:
        mamba_model = Mamba2ForSentimentAnalysis(vocab_size, d_model=128, num_classes=num_classes).to(device)
        optimizer_mamba = torch.optim.AdamW(mamba_model.parameters(), lr=args.learning_rate)
        models_dict['Mamba'] = mamba_model
        optimizers['Mamba'] = optimizer_mamba
    
    # Train models
    for name, model in models_dict.items():
        print(f'\nTraining {name} model...')
        optimizer = optimizers[name]
        train_model(model, optimizer, train_loader, val_loader, args.epochs, name, device)
    
    # Load best models
    for name, model in models_dict.items():
        model.load_state_dict(torch.load(f'best_{name}.pt'))
        model.eval()
    
    # Get predictions from each model
    predictions = {}
    for name, model in models_dict.items():
        print(f'Predicting with {name} model...')
        preds = get_predictions(model, test_loader, device)
        predictions[name] = preds
    
    # Ensemble with Majority Voting
    ensemble_preds = majority_vote(predictions)
    
    # Evaluate
    y_true = y_test
    accuracy, precision, recall, f1 = evaluate_model(y_true, ensemble_preds)
    
    print("\nEnsemble Model Performance on Test Set:")
    print(f'Accuracy : {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall   : {recall:.4f}')
    print(f'F1 Score : {f1:.4f}')
    
    # Individual Model Performance
    print("\nIndividual Models Performance on Test Set:")
    for name, preds in predictions.items():
        acc, prec, rec, f1_s = evaluate_model(y_true, preds)
        print(f'{name} - Acc: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1_s:.4f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and evaluate sentiment analysis models.')
    parser.add_argument('--train_file', type=str, default='train.csv', help='Path to the training data CSV file.')
    parser.add_argument('--test_file', type=str, default='test.csv', help='Path to the test data CSV file.')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train.')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training.')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate.')
    parser.add_argument('--max_sequence_length', type=int, default=100, help='Maximum sequence length.')
    parser.add_argument('--max_num_words', type=int, default=20000, help='Maximum number of words in vocabulary.')
    parser.add_argument('--embedding_dim', type=int, default=100, help='Dimension of embedding vectors.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for training (cuda or cpu).')
    parser.add_argument('--models_to_train', nargs='+', default=['RNN', 'LSTM', 'BiLSTM', 'TextCNN', 'gMLP', 'Mamba'],
                        help='List of models to train. Options: RNN, LSTM, BiLSTM, TextCNN, gMLP, Mamba')

    args = parser.parse_args()
    main(args)
