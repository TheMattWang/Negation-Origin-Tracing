import torch
from torch.utils.data import Dataset
import pandas as pd
from transformers import AutoTokenizer
from typing import Optional, Dict, List
import os


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
        raw_labels = self.data[label_column].tolist()
        
        # Check for NaN or invalid values
        import numpy as np
        raw_labels_clean = [l for l in raw_labels if pd.notna(l)]
        if len(raw_labels_clean) != len(raw_labels):
            raise ValueError(
                f"Found {len(raw_labels) - len(raw_labels_clean)} NaN/invalid labels in dataset. "
                f"Total labels: {len(raw_labels)}"
            )
        
        # Convert to integers and check unique values
        try:
            raw_labels_int = [int(float(label)) for label in raw_labels]  # Handle float labels
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"Could not convert labels to integers. Sample labels: {raw_labels[:10]}. Error: {e}"
            )
        
        unique_labels = sorted(set(raw_labels_int))
        min_label, max_label = min(raw_labels_int), max(raw_labels_int)
        
        # Log label information for debugging
        print(f"[{os.path.basename(data_path)}] Labels - Unique: {unique_labels}, Range: [{min_label}, {max_label}], Count: {len(raw_labels_int)}")
        
        # Handle SST-2 test set which has -1 labels (no ground truth)
        if unique_labels == [-1] or (-1 in unique_labels and len(unique_labels) == 1):
            print(f"  → Test set with -1 labels (no ground truth) - will skip loss/metrics computation")
            self.labels = [-1] * len(raw_labels_int)
            self.has_labels = False
        else:
            self.has_labels = True
            # Validate and normalize labels
            if set(unique_labels) == {0, 1}:
                # Already correct
                self.labels = raw_labels_int
            elif len(unique_labels) == 2 and min_label >= 0:
                # Need to map to 0-1 (e.g., if labels are 1-2)
                label_map = {unique_labels[0]: 0, unique_labels[1]: 1}
                self.labels = [label_map[label] for label in raw_labels_int]
                print(f"  → Mapped labels {unique_labels} to [0, 1]")
            else:
                # Invalid labels - raise error with details
                raise ValueError(
                    f"Invalid labels for binary classification in {data_path}:\n"
                    f"  Unique labels: {unique_labels}\n"
                    f"  Range: [{min_label}, {max_label}]\n"
                    f"  Expected: [0, 1] or mappable to [0, 1], or [-1] for test sets\n"
                    f"  First 20 labels: {raw_labels_int[:20]}\n"
                    f"  Label distribution: {pd.Series(raw_labels_int).value_counts().to_dict()}"
                )
            
            # Final validation
            final_unique = set(self.labels)
            if final_unique != {0, 1}:
                raise ValueError(
                    f"Label normalization failed. Final labels: {final_unique}, expected {{0, 1}}"
                )
        
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
        }
        
        # Only include labels if they're valid (not -1)
        if label != -1:
            # Final validation
            if label not in [0, 1]:
                raise ValueError(
                    f"Invalid label {label} at index {idx}. Expected 0 or 1. "
                    f"This should not happen - check dataset initialization."
                )
            item["labels"] = torch.tensor(label, dtype=torch.long)
        # If label is -1, don't include it (test set without ground truth)
        
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
        raw_labels = self.data[label_column].tolist()
        
        # Check for NaN or invalid values
        import numpy as np
        raw_labels_clean = [l for l in raw_labels if pd.notna(l)]
        if len(raw_labels_clean) != len(raw_labels):
            raise ValueError(
                f"Found {len(raw_labels) - len(raw_labels_clean)} NaN/invalid labels in dataset. "
                f"Total labels: {len(raw_labels)}"
            )
        
        # Convert to integers and check unique values
        try:
            raw_labels_int = [int(float(label)) for label in raw_labels]  # Handle float labels
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"Could not convert labels to integers. Sample labels: {raw_labels[:10]}. Error: {e}"
            )
        
        unique_labels = sorted(set(raw_labels_int))
        min_label, max_label = min(raw_labels_int), max(raw_labels_int)
        
        # Log label information for debugging
        print(f"Dataset labels - Unique: {unique_labels}, Range: [{min_label}, {max_label}], Count: {len(raw_labels_int)}")
        
        # Handle test sets without labels (e.g., if external dataset omits labels)
        if unique_labels == [-1] or (-1 in unique_labels and len(unique_labels) == 1):
            print(f"  → Dataset has no ground truth labels (-1). Marking as unlabeled.")
            self.labels = [-1] * len(raw_labels_int)
            self.has_labels = False
        else:
            self.has_labels = True
            # Validate labels are in expected range [0, 1] for binary classification
            if min_label < 0 or max_label > 1:
                raise ValueError(
                    f"Labels out of range for binary classification: min={min_label}, max={max_label}. "
                    f"Expected range [0, 1]. Unique labels: {unique_labels}. "
                    f"First 10 labels: {raw_labels_int[:10]}"
                )
            
            # Ensure labels are exactly 0 and 1
            if set(unique_labels) != {0, 1}:
                if len(unique_labels) == 2 and min_label >= 0:
                    # Map to 0-1 if they're 1-2 or similar
                    label_map = {unique_labels[0]: 0, unique_labels[1]: 1}
                    self.labels = [label_map[label] for label in raw_labels_int]
                    print(f"  Mapped labels {unique_labels} to [0, 1]")
                else:
                    raise ValueError(
                        f"Unexpected label values: {unique_labels}. Expected exactly [0, 1] for binary classification, "
                        f"or [-1] for unlabeled test sets."
                    )
            else:
                self.labels = raw_labels_int
        
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
        
        # Final validation for labeled data
        if label not in [0, 1] and label != -1:
            raise ValueError(
                f"Invalid label {label} at index {idx}. Expected 0 or 1 (or -1 for unlabeled test data). "
                f"This should not happen - check dataset initialization."
            )
        
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
        }
        
        if label != -1:
            item["labels"] = torch.tensor(label, dtype=torch.long)
        
        # Add pair_id if available
        if self.pair_ids is not None:
            item["pair_id"] = torch.tensor(self.pair_ids[idx], dtype=torch.long)
        
        return item

