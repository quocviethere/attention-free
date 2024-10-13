# evaluate.py

import torch
import numpy as np
from collections import Counter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def get_predictions(model, loader, device):
    """Get predictions from the model."""
    all_preds = []
    with torch.no_grad():
        for reviews, _ in loader:
            reviews = reviews.to(device)
            outputs = model(reviews)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
    return np.array(all_preds)

def majority_vote(preds_dict):
    """Perform majority voting ensemble."""
    preds = np.stack([preds for preds in preds_dict.values()], axis=1)
    ensemble_preds = []
    for instance in preds:
        most_common = Counter(instance).most_common(1)[0][0]
        ensemble_preds.append(most_common)
    return np.array(ensemble_preds)

def evaluate_model(y_true, y_pred):
    """Evaluate model performance."""
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    return accuracy, precision, recall, f1
