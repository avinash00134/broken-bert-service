import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer
from sklearn.model_selection import train_test_split
from typing import Tuple, List


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
            'label': torch.tensor(label, dtype=torch.long)  # Changed to long for CrossEntropyLoss
        }


def prepare_data(csv_path: str) -> Tuple[List[str], List[int]]:
    """
    Prepare data from CSV file.
    
    Args:
        csv_path: Path to CSV file
        
    Returns:
        Tuple of (texts, labels)
    """
    df = pd.read_csv(csv_path)
    
    # Encode labels as integers
    label_map = {'negative': 0, 'positive': 1}
    texts = df['review'].tolist()
    labels = df['label'].map(label_map).tolist()
    
    return texts, labels

def load_data(csv_path: str, test_size: float = 0.2, random_state: int = 42, sample_frac: float = 0.5) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and split the dataset.
    
    Args:
        csv_path: Path to the CSV file
        test_size: Proportion of data for testing
        random_state: Random seed for reproducibility
        sample_frac: Fraction of data to use (for memory constraints)
        
    Returns:
        Tuple of (train_df, test_df)
    """
    # Load the CSV file
    df = pd.read_csv(csv_path)
    
    # Sample a fraction of data if needed for memory constraints
    if sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=random_state)
        print(f"Using {sample_frac*100}% of data: {len(df)} samples")
    
    df = df.copy()
    # Convert labels to numeric (0 for negative, 1 for positive)
    df['label_num'] = df['label'].map({'negative': 0, 'positive': 1})
    df['review'] = df['review'].str.lower()
    
    # Split the data
    train_df, test_df = train_test_split(
        df, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=df['label_num']
    )
    
    return train_df, test_df


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
    dataset = ReviewDataset(
        reviews=df['review'].tolist(),
        labels=df['label_num'].tolist(),
        tokenizer=tokenizer,
        max_length=max_length
    )
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def get_class_weights(df: pd.DataFrame) -> torch.Tensor:
    """
    Calculate class weights for handling imbalanced data.
    
    Args:
        df: DataFrame containing the labels
        
    Returns:
        Tensor of class weights
    """
    label_counts = df['label_num'].value_counts().sort_index()
    total_samples = len(df)
    
    weights = []
    for count in label_counts:
        weight = total_samples / (len(label_counts) * count)
        weights.append(weight)
    
    return torch.tensor(weights, dtype=torch.float)