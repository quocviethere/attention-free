# train.py

import torch
import torch.nn as nn

criterion = nn.CrossEntropyLoss()

def train_model(model, optimizer, train_loader, val_loader, epochs, model_name, device):
    """Train the model and save the best checkpoint."""
    best_val_loss = float('inf')
    for epoch in range(epochs):
        model.train()
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

        # Validation
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for reviews, targets in val_loader:
                reviews, targets = reviews.to(device), targets.to(device)
                outputs = model(reviews)
                loss = criterion(outputs, targets)
                total_val_loss += loss.item()
        avg_val_loss = total_val_loss / len(val_loader)

        print(f'{model_name} Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}')

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f'best_{model_name}.pt')
