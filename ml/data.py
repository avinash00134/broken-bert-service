import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer
from sklearn.model_selection import train_test_split
from typing import Tuple, List, Optional
import logging

logger = logging.getLogger(__name__)


class ReviewDataset(Dataset):
    """Dataset class for review text classification."""
    
    def __init__(self, reviews: List[str], labels: List[int], tokenizer: DistilBertTokenizer, max_length: int = 128):
        """
        Initialize the dataset.
        
        Args:
            reviews: List of review texts
            labels: List of labels (0 for negative, 1 for positive)
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
        """
        self.reviews = reviews
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.reviews)
    
    def __getitem__(self, idx):
        review = str(self.reviews[idx])
        label = self.labels[idx]
        
        # Tokenize the text
        encoding = self.tokenizer(
            review,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)  # FIX: Changed to long for CrossEntropyLoss
        }


def load_data(csv_path: str, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and split the dataset.
    
    Args:
        csv_path: Path to the CSV file
        test_size: Proportion of data for testing
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (train_df, test_df)
    """
    try:
        # Load the CSV file
        df = pd.read_csv(csv_path)
        
        logger.info(f"Loaded dataset with {len(df)} samples")
        
        # FIX: Properly handle the DataFrame copy and preprocessing
        df = df.copy()
        
        # Check required columns
        if 'review' not in df.columns or 'label' not in df.columns:
            raise ValueError("CSV must contain 'review' and 'label' columns")
        
        # Convert labels to numeric (0 for negative, 1 for positive)
        df['label_num'] = df['label'].map({'negative': 0, 'positive': 1})
        
        # Check for missing labels
        if df['label_num'].isna().any():
            invalid_labels = df[df['label_num'].isna()]['label'].unique()
            raise ValueError(f"Invalid labels found: {invalid_labels}. Expected 'positive' or 'negative'")
        
        # Clean and preprocess text
        df['review'] = df['review'].astype(str).str.lower().str.strip()
        
        # Remove empty reviews
        original_size = len(df)
        df = df[df['review'].str.len() > 0]
        if len(df) < original_size:
            logger.warning(f"Removed {original_size - len(df)} empty reviews")
        
        # Split the data
        train_df, test_df = train_test_split(
            df, 
            test_size=test_size, 
            random_state=random_state, 
            stratify=df['label_num']
        )
        
        logger.info(f"Training samples: {len(train_df)}, Test samples: {len(test_df)}")
        logger.info(f"Class distribution - Train: {train_df['label_num'].value_counts().to_dict()}")
        logger.info(f"Class distribution - Test: {test_df['label_num'].value_counts().to_dict()}")
        
        return train_df, test_df
        
    except Exception as e:
        logger.error(f"Error loading data from {csv_path}: {str(e)}")
        raise


def create_data_loader(df: pd.DataFrame, tokenizer: DistilBertTokenizer, max_length: int = 128, 
                      batch_size: int = 16, shuffle: bool = True) -> DataLoader:
    """
    Create a DataLoader from a DataFrame.
    
    Args:
        df: DataFrame containing reviews and labels
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length
        batch_size: Batch size for DataLoader
        shuffle: Whether to shuffle the data
        
    Returns:
        DataLoader object
    """
    # FIX: Validate input data
    if df is None or len(df) == 0:
        raise ValueError("DataFrame is empty or None")
    
    if 'review' not in df.columns or 'label_num' not in df.columns:
        raise ValueError("DataFrame must contain 'review' and 'label_num' columns")
    
    dataset = ReviewDataset(
        reviews=df['review'].tolist(),
        labels=df['label_num'].tolist(),
        tokenizer=tokenizer,
        max_length=max_length
    )
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)  # FIX: Added num_workers for stability


def get_class_weights(df: pd.DataFrame) -> torch.Tensor:
    """
    Calculate class weights for handling imbalanced data.
    
    Args:
        df: DataFrame containing the labels
        
    Returns:
        Tensor of class weights
    """
    if df is None or len(df) == 0:
        logger.warning("Empty DataFrame provided, returning default weights")
        return torch.tensor([1.0, 1.0], dtype=torch.float)
    
    # FIX: Use the entire dataset instead of sampling
    label_counts = df['label_num'].value_counts().sort_index()
    total_samples = len(df)
    
    # FIX: Handle case where some classes might be missing
    weights = []
    for i in range(2):  # We have 2 classes: 0 (negative) and 1 (positive)
        if i in label_counts.index:
            count = label_counts[i]
            # Calculate weight as: total_samples / (n_classes * count)
            weight = total_samples / (len(label_counts) * count)
            weights.append(weight)
        else:
            # If a class is missing, use weight 1.0
            logger.warning(f"Class {i} not found in dataset, using default weight 1.0")
            weights.append(1.0)
    
    logger.info(f"Class weights calculated: {weights}")
    return torch.tensor(weights, dtype=torch.float)