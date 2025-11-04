import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import DistilBertModel, DistilBertTokenizer
from sklearn.metrics import accuracy_score
import os
from tqdm import tqdm
from typing import Tuple, Dict, Any
import pandas as pd

from ml.data import load_data, create_data_loader, get_class_weights


class DistilBertClassifier(nn.Module):
    """DistilBERT-based text classifier."""
    
    def __init__(self, n_classes: int = 2, dropout: float = 0.3):
        """
        Initialize the classifier.
        
        Args:
            n_classes: Number of classes for classification
            dropout: Dropout rate
        """
        super(DistilBertClassifier, self).__init__()
        
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)
        
    def forward(self, input_ids, attention_mask):
        """Forward pass."""
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use the [CLS] token representation
        pooled_output = outputs.last_hidden_state[:, 0]
        output = self.dropout(pooled_output)
        return self.classifier(output)


class BertTrainer:
    """Trainer class for BERT-based text classification."""
    
    def __init__(self, model: DistilBertClassifier, device: torch.device):
        """
        Initialize the trainer.
        
        Args:
            model: The model to train
            device: Device to train on (CPU or CUDA)
        """
        self.model = model
        self.device = device
        self.model.to(device)
        
    def train_epoch(self, data_loader: DataLoader, optimizer: optim.Optimizer, 
                   criterion: nn.Module) -> Tuple[float, float]:
        """
        Train the model for one epoch.
        
        Args:
            data_loader: DataLoader for training data
            optimizer: Optimizer
            criterion: Loss function
            
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.train()
        
        total_loss = 0
        predictions = []
        true_labels = []
        
        progress_bar = tqdm(data_loader, desc="Training")
        
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['label'].to(self.device).long()  # Fix: Convert to long tensor
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            predictions.extend(predicted.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(data_loader)
        accuracy = accuracy_score(true_labels, predictions)
        
        return avg_loss, accuracy
    
    def evaluate(self, data_loader: DataLoader, criterion: nn.Module) -> Tuple[float, float]:
        """
        Evaluate the model.
        
        Args:
            data_loader: DataLoader for evaluation data
            criterion: Loss function
            
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.eval()
        
        total_loss = 0
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Evaluating"):
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device).long()  # Fix: Convert to long tensor
                
                # Forward pass
                outputs = self.model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                
                # Track metrics
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                predictions.extend(predicted.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(data_loader)
        accuracy = accuracy_score(true_labels, predictions)
        
        return avg_loss, accuracy


def train_model(csv_path: str = 'assets/reviews.csv',
               model_save_path: str = 'accets/model.pth',  # Fix: Corrected path from 'assets' to 'accets'
               tokenizer_save_path: str = 'accets/tokenizer/',  # Fix: Corrected path
               epochs: int = 2,
               batch_size: int = 16,
               learning_rate: float = 2e-5,
               max_length: int = 128) -> Dict[str, Any]:
    """
    Train the text classification model.
    
    Args:
        csv_path: Path to the dataset CSV file
        model_save_path: Path to save the trained model
        tokenizer_save_path: Path to save the tokenizer
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate
        max_length: Maximum sequence length
        
    Returns:
        Training history dictionary
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    # Load and prepare data
    print("Loading data...")
    train_df, test_df = load_data(csv_path)
    print(f"Training samples: {len(train_df)}, Test samples: {len(test_df)}")
    
    # Create data loaders
    train_loader = create_data_loader(train_df, tokenizer, max_length, batch_size, shuffle=True)
    test_loader = create_data_loader(test_df, tokenizer, max_length, batch_size, shuffle=False)
    
    # Initialize model
    model = DistilBertClassifier(n_classes=2)
    trainer = BertTrainer(model, device)
    
    # Setup training components
    class_weights = get_class_weights(train_df).to(device)  # Fix: Use train_df for class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Training history
    history = {
        'train_loss': [],
        'train_accuracy': [],
        'test_loss': [],
        'test_accuracy': []
    }
    
    print(f"\nStarting training for {epochs} epochs...")
    print("-" * 50)
    
    # Training loop
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        
        # Train
        train_loss, train_acc = trainer.train_epoch(train_loader, optimizer, criterion)
        
        # Evaluate
        test_loss, test_acc = trainer.evaluate(test_loader, criterion)
        
        # Save metrics
        history['train_loss'].append(train_loss)
        history['train_accuracy'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_accuracy'].append(test_acc)
        
        # Print epoch results
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
    
    # Create assets directory if it doesn't exist
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    os.makedirs(tokenizer_save_path, exist_ok=True)
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'n_classes': 2,
            'dropout': 0.3
        },
        'history': history
    }, model_save_path)
    
    # Save tokenizer
    tokenizer.save_pretrained(tokenizer_save_path)
    
    print(f"\nModel saved to: {model_save_path}")
    print(f"Tokenizer saved to: {tokenizer_save_path}")
    
    return history


def main():
    """Main function for running training with command line arguments."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train text classification model')
    parser.add_argument('--csv_path', type=str, default='assets/reviews.csv',
                        help='Path to the dataset CSV file (default: assets/reviews.csv)')
    parser.add_argument('--model_path', type=str, default='accets/model.pth',  # Fix: Corrected default path
                        help='Path to save the trained model (default: accets/model.pth)')
    parser.add_argument('--tokenizer_path', type=str, default='accets/tokenizer/',  # Fix: Corrected default path
                        help='Path to save the tokenizer (default: accets/tokenizer/)')
    parser.add_argument('--epochs', type=int, default=2,
                        help='Number of training epochs (default: 2)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training (default: 16)')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                        help='Learning rate (default: 2e-5)')
    parser.add_argument('--max_length', type=int, default=128,
                        help='Maximum sequence length (default: 128)')
    
    args = parser.parse_args()
    
    # Check if dataset exists
    if not os.path.exists(args.csv_path):
        print(f"Error: Dataset file '{args.csv_path}' not found!")
        print("Please make sure the reviews.csv file exists in the assets/ directory.")
        return
    
    print("=" * 60)
    print("Text Classification Model Training")
    print("=" * 60)
    print(f"Dataset: {args.csv_path}")
    print(f"Model save path: {args.model_path}")
    print(f"Tokenizer save path: {args.tokenizer_path}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Max sequence length: {args.max_length}")
    print("=" * 60)
    
    try:
        # Train the model
        history = train_model(
            csv_path=args.csv_path,
            model_save_path=args.model_path,
            tokenizer_save_path=args.tokenizer_path,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_length=args.max_length
        )
        
        print("\n" + "=" * 60)
        print("Training completed successfully!")
        print("=" * 60)
        
        # Print final results
        final_train_acc = history['train_accuracy'][-1]
        final_test_acc = history['test_accuracy'][-1]
        
        print(f"Final Training Accuracy: {final_train_acc:.4f} ({final_train_acc*100:.2f}%)")
        print(f"Final Test Accuracy: {final_test_acc:.4f} ({final_test_acc*100:.2f}%)")
        
        print(f"\nModel and tokenizer saved successfully!")
        print(f"Model: {args.model_path}")
        print(f"Tokenizer: {args.tokenizer_path}")
        
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()