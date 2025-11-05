import torch
import torch.nn.functional as F
from transformers import DistilBertTokenizer
import os
from typing import Tuple
import warnings

from ml.train import DistilBertClassifier


class ReviewClassifier:
    """
    A classifier for predicting sentiment of review text using a trained DistilBERT model.
    
    This class loads a pre-trained model and tokenizer from disk and provides
    an interface for making predictions on new text data.
    """
    
    def __init__(self, model_path: str = 'assets/model.pth', 
                 tokenizer_path: str = 'assets/tokenizer/', 
                 device: str = None):
        """
        Initialize the ReviewClassifier by loading the trained model and tokenizer.
        
        Args:
            model_path: Path to the saved model file
            tokenizer_path: Path to the saved tokenizer directory
            device: Device to run inference on ('cpu', 'cuda', or None for auto-detect)
        
        Raises:
            FileNotFoundError: If model or tokenizer files are not found
            RuntimeError: If model loading fails
        """
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Loading model on device: {self.device}")
        
        # Check if files exist
        if not os.path.exists(model_path):
            # Try the best model if main model doesn't exist
            best_model_path = model_path.replace('.pth', '_best.pth')
            if os.path.exists(best_model_path):
                model_path = best_model_path
                print(f"Using best model: {model_path}")
            else:
                raise FileNotFoundError(f"Model file not found: {model_path}")
        
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"Tokenizer directory not found: {tokenizer_path}")
        
        # Load tokenizer
        try:
            self.tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_path)
            print("Tokenizer loaded successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to load tokenizer: {str(e)}")
        
        # Load model
        try:
            # Load the checkpoint
            checkpoint = torch.load(model_path, map_location='cpu')
            
            # Get model configuration
            model_config = checkpoint.get('model_config', {'n_classes': 2, 'dropout': 0.3})
            
            # Initialize model with saved configuration
            self.model = DistilBertClassifier(
                n_classes=model_config['n_classes'],
                dropout=model_config['dropout']
            )
            
            # Load the state dict
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
            
            print("Model loaded successfully")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")
        
        # Define label mapping
        self.label_map = {0: 'negative', 1: 'positive'}
        
    def predict(self, text: str, max_length: int = 128) -> Tuple[str, float]:
        """
        Predict the sentiment of the given text.
        
        Args:
            text: Input text to classify
            max_length: Maximum sequence length for tokenization
            
        Returns:
            Tuple containing:
                - predicted_label: 'positive' or 'negative'
                - confidence: Confidence score (probability) for the predicted class
        
        Raises:
            ValueError: If input text is empty or None
        """
        if not text or not text.strip():
            raise ValueError("Input text cannot be empty or None")
        
        # Preprocess text
        text = str(text).strip()
        
        # Tokenize the input text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )
        
        # Move tensors to device
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # Make prediction (ADD THE MISSING CONTEXT MANAGER)
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
            
            # Apply softmax to get probabilities
            probabilities = F.softmax(outputs, dim=1)
            
            # Get the predicted class and confidence
            confidence, predicted_class = torch.max(probabilities, dim=1)
            
            predicted_label = self.label_map[predicted_class.item()]
            confidence_score = confidence.item()
        
        return predicted_label, confidence_score
    
    def predict_batch(self, texts: list, max_length: int = 64, batch_size: int = 8) -> list:  # Reduced defaults
        """
        Predict sentiments for a batch of texts.
        
        Args:
            texts: List of input texts to classify
            max_length: Maximum sequence length for tokenization
            batch_size: Batch size for processing
            
        Returns:
            List of tuples, each containing (predicted_label, confidence)
        """
        if not texts:
            return []
        
        results = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize the batch
            encodings = self.tokenizer(
                batch_texts,
                truncation=True,
                padding='max_length',
                max_length=max_length,
                return_tensors='pt'
            )
            
            # Move to device
            input_ids = encodings['input_ids'].to(self.device)
            attention_mask = encodings['attention_mask'].to(self.device)
            
            # Make predictions
            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask)
                probabilities = F.softmax(outputs, dim=1)
                
                confidences, predicted_classes = torch.max(probabilities, dim=1)
                
                # Convert to labels and add to results
                for pred_class, conf in zip(predicted_classes, confidences):
                    predicted_label = self.label_map[pred_class.item()]
                    confidence_score = conf.item()
                    results.append((predicted_label, confidence_score))
        
        return results
    
    def get_model_info(self) -> dict:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary containing model information
        """
        return {
            'device': str(self.device),
            'model_type': 'DistilBERT',
            'n_classes': len(self.label_map),
            'labels': list(self.label_map.values())
        }