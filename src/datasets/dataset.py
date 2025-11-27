import torch
from torch.utils.data import Dataset
import pandas as pd
from transformers import AutoTokenizer
from typing import Optional, Dict, List


class SentimentDataset(Dataset):
    """
    PyTorch Dataset for sentiment classification tasks.
    Handles SST-2 and CSD Negation datasets.
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer: AutoTokenizer,
        max_length: int = 128,
        text_column: str = "sentence",
        label_column: str = "label",
    ):
        """
        Args:
            data_path: Path to parquet file containing the data
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length for tokenization
            text_column: Name of the column containing text
            label_column: Name of the column containing labels
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_column = text_column
        self.label_column = label_column
        
        # Load data
        self.data = pd.read_parquet(data_path)
        
        # Validate columns exist
        if text_column not in self.data.columns:
            raise ValueError(f"Column '{text_column}' not found in data. Available columns: {self.data.columns.tolist()}")
        if label_column not in self.data.columns:
            raise ValueError(f"Column '{label_column}' not found in data. Available columns: {self.data.columns.tolist()}")
        
        # Store texts and labels
        self.texts = self.data[text_column].tolist()
        self.labels = self.data[label_column].tolist()
        
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns a single sample with tokenized inputs and label.
        """
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        
        # Convert to single tensors (remove batch dimension)
        item = {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long),
        }
        
        return item


class NegationPairDataset(Dataset):
    """
    Dataset for handling negation pairs (e.g., from CSD Negation dataset).
    Useful for contrastive analysis and activation patching experiments.
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer: AutoTokenizer,
        max_length: int = 128,
        text_column: str = "sentence",
        label_column: str = "label",
        pair_id_column: Optional[str] = None,
    ):
        """
        Args:
            data_path: Path to parquet file containing the data
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length for tokenization
            text_column: Name of the column containing text
            label_column: Name of the column containing labels
            pair_id_column: Optional column for pairing negated/non-negated examples
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_column = text_column
        self.label_column = label_column
        self.pair_id_column = pair_id_column
        
        # Load data
        self.data = pd.read_parquet(data_path)
        
        # Validate columns
        if text_column not in self.data.columns:
            raise ValueError(f"Column '{text_column}' not found in data. Available columns: {self.data.columns.tolist()}")
        if label_column not in self.data.columns:
            raise ValueError(f"Column '{label_column}' not found in data. Available columns: {self.data.columns.tolist()}")
        
        self.texts = self.data[text_column].tolist()
        self.labels = self.data[label_column].tolist()
        
        # Store pair IDs if available
        if pair_id_column and pair_id_column in self.data.columns:
            self.pair_ids = self.data[pair_id_column].tolist()
        else:
            self.pair_ids = None
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns a single sample with tokenized inputs, label, and optional pair_id.
        """
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        
        item = {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long),
        }
        
        # Add pair_id if available
        if self.pair_ids is not None:
            item["pair_id"] = torch.tensor(self.pair_ids[idx], dtype=torch.long)
        
        return item

