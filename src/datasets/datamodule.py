import os
from typing import Optional
from lightning import LightningDataModule
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from .dataset import SentimentDataset, NegationPairDataset


class NOTDataModule(LightningDataModule):
    """
    Lightning DataModule for Negation-Origin-Tracing project.
    Handles SST-2 and CSD Negation datasets.
    """
    
    def __init__(
        self,
        data_dir: str = "data/raw",
        model_name: str = "distilbert-base-uncased",
        batch_size: int = 32,
        num_workers: int = 4,
        max_length: int = 128,
        use_negation_dataset: bool = False,
        seed: int = 42,
    ):
        """
        Args:
            data_dir: Root directory containing train/validation/test subdirectories
            model_name: HuggingFace model name for tokenizer
            batch_size: Batch size for data loaders
            num_workers: Number of workers for data loading
            max_length: Maximum sequence length for tokenization
            use_negation_dataset: Whether to use CSD Negation dataset (if available)
            seed: Random seed for reproducibility
        """
        super().__init__()
        self.data_dir = data_dir
        self.model_name = model_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_length = max_length
        self.use_negation_dataset = use_negation_dataset
        self.seed = seed
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Dataset paths
        self.train_path = os.path.join(data_dir, "train", "sst.parquet")
        self.val_path = os.path.join(data_dir, "validation", "sst.parquet")
        self.test_path = os.path.join(data_dir, "test", "sst.parquet")
        
        # Optional negation dataset paths
        self.train_neg_path = os.path.join(data_dir, "train", "negation.parquet")
        self.val_neg_path = os.path.join(data_dir, "validation", "negation.parquet")
        self.test_neg_path = os.path.join(data_dir, "test", "negation.parquet")
        
        # Datasets (will be initialized in setup)
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    def setup(self, stage: Optional[str] = None):
        """
        Set up datasets for train/val/test.
        Called on every GPU/TPU in distributed training.
        """
        # Training dataset
        if os.path.exists(self.train_path):
            self.train_dataset = SentimentDataset(
                data_path=self.train_path,
                tokenizer=self.tokenizer,
                max_length=self.max_length,
            )
        else:
            raise FileNotFoundError(f"Training data not found at {self.train_path}")
        
        # Validation dataset
        if os.path.exists(self.val_path):
            self.val_dataset = SentimentDataset(
                data_path=self.val_path,
                tokenizer=self.tokenizer,
                max_length=self.max_length,
            )
        else:
            raise FileNotFoundError(f"Validation data not found at {self.val_path}")
        
        # Test dataset - use SST-2 by default, CSD Negation if available and requested
        if self.use_negation_dataset and os.path.exists(self.test_neg_path):
            # Use negation dataset for testing
            self.test_dataset = NegationPairDataset(
                data_path=self.test_neg_path,
                tokenizer=self.tokenizer,
                max_length=self.max_length,
            )
        elif os.path.exists(self.test_path):
            # Use standard SST-2 test set
            self.test_dataset = SentimentDataset(
                data_path=self.test_path,
                tokenizer=self.tokenizer,
                max_length=self.max_length,
            )
        else:
            raise FileNotFoundError(
                f"Test data not found. Tried: {self.test_path} and {self.test_neg_path}"
            )
    
    def train_dataloader(self) -> DataLoader:
        """Returns training data loader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )
    
    def val_dataloader(self) -> DataLoader:
        """Returns validation data loader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
    
    def test_dataloader(self) -> DataLoader:
        """Returns test data loader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
    
    def __len__(self) -> int:
        """Returns the size of the training dataset."""
        if self.train_dataset is None:
            return 0
        return len(self.train_dataset)
    
    @classmethod
    def from_args(cls, args):
        """
        Create DataModule from argparse.Namespace object.
        
        Args:
            args: argparse.Namespace with data_dir, model_name, batch_size, etc.
        
        Returns:
            NOTDataModule instance
        """
        return cls(
            data_dir=getattr(args, "data_dir", "data/raw"),
            model_name=getattr(args, "model_name", "distilbert-base-uncased"),
            batch_size=getattr(args, "batch_size", 32),
            num_workers=getattr(args, "num_workers", 4),
            max_length=getattr(args, "max_length", 128),
            use_negation_dataset=getattr(args, "use_negation_dataset", False),
            seed=getattr(args, "seed", 42),
        )