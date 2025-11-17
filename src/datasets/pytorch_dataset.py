import json
import os
from collections import defaultdict
from datetime import datetime, timedelta
from enum import StrEnum
from pathlib import Path

from omegaconf import DictConfig
from transformers import AutoTokenizer

class PytorchDataset(SeedableMixin, torch.utils.data.Dataset, TimeableMixin):
    """A PyTorch Dataset class for handling complex, multi-modal medical data.

    This dataset is designed to work with data from the MEDS (Medical Event Data Set) format, supporting
    various types of medical events, static patient information, and task-specific labels. It provides
    functionality for loading, processing, and collating data for use in PyTorch models.

    Key Features: - Handles different collation strategies (event stream, triplet, text-code, etc.) - Supports
    task-specific data handling for binary classification - Implements custom sampling strategies and sequence
    length constraints

    Args:     cfg (DictConfig): Configuration options for the dataset.     split (str): The data split to use
    (e.g., 'train', 'validation', 'test').

    Attributes:     config (DictConfig): The dataset configuration.     split (str): The current data split.
    code_metadata (pl.LazyFrame): Metadata for event codes.     static_dfs (dict): Dictionary of static
    DataFrames for each data shard.     subj_indices (dict): Mapping of subject IDs to their indices in the
    dataset.     subj_seq_bounds (dict): Sequence bounds (start, end) for each subject.     index (list): List
    of (subject_id, start, end) tuples for data access.     labels (dict): Task-specific labels for each data
    point.     tasks (list): List of task names.     task_types (dict): Mapping of task names to their types
    (classification, regression, etc.).     task_vocabs (dict): Vocabularies for classification tasks.
    tokenized_codes (dict): Tokenized representations of event codes (for text-code collation).

    Methods:     __len__(): Returns the number of items in the dataset.     __getitem__(idx): Retrieves a
    single data point.     collate(batch): Collates a batch of data points based on the specified collation
    strategy.
    """

    TYPE_CHECKERS = {
        "multi_class_classification": [
            (
                {pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64, pl.Int8, pl.Int16, pl.Int32, pl.Int64},
                None,
            ),
            ({pl.Categorical(ordering="physical"), pl.Categorical(ordering="lexical")}, to_int_index),
            ({pl.Utf8}, to_int_index),
        ],
        "binary_classification": [({pl.Boolean}, lambda Y: Y.cast(pl.Float32))],
        "regression": [({pl.Float32, pl.Float64}, None)],
    }
    """Type checker and conversion parameters for labeled datasets."""

    @classmethod
    def normalize_task(cls, col: pl.Expr, dtype: pl.DataType) -> tuple[str, pl.Expr]:
        """Normalize task labels to a common format based on their data type.

        This method determines the appropriate task type (e.g., multi-class classification, binary
        classification, regression) based on the data type of the label column and applies any necessary
        transformations to normalize the data.

        Args:     col (pl.Expr): The polars Expression containing the task labels.     dtype (pl.DataType):
        The polars data type of the task labels.

        Returns:     tuple: A tuple containing two elements:         - str: The determined task type (e.g.,
        'multi_class_classification', 'binary_classification', 'regression').         - pl.Expr: The
        normalized column expression.

        Raises:     TypeError: If the task labels are not of a supported type.
        """
        for task_type, checkers in cls.TYPE_CHECKERS.items():
            for valid_dtypes, normalize_fn in checkers:
                if dtype in valid_dtypes:
                    return task_type, (col if normalize_fn is None else normalize_fn(col))
        raise TypeError(f"Can't process label of {dtype} type!")

    def __init__(self, cfg: DictConfig, split: str):
        super().__init__()

        self.config = cfg
        self.split = split

        logger.info("Scanning code metadata")
        self.code_metadata = pl.scan_parquet(self.config.code_metadata_fp)

        logger.info("Reading splits & subject shards")
        self.read_shards()

        logger.info("Reading subject descriptors")
        self.read_subject_descriptors()

        if self.config.min_seq_len is not None and self.config.min_seq_len > 1:
            logger.info(f"Restricting to subjects with at least {self.config.min_seq_len} events")
            self.filter_to_min_seq_len()

        if self.config.train_subset_size not in (None, "FULL") and self.split == "train":
            logger.info(f"Filtering training subset size to {self.config.train_subset_size}")
            self.filter_to_subset()

        self.set_inter_event_time_stats()

        # Initialize tokenizer here
        self.init_tokenizer()

    def init_tokenizer(self):
        if self.config.collate_type == CollateType.text_code:
            if not hasattr(self, "tokenized_codes"):
                # Disable parallelism for tokenization as it will cause issues when num_workers > 0 in the
                # pytorch dataloader
                os.environ["TOKENIZERS_PARALLELISM"] = "false"
                tokenizer = AutoTokenizer.from_pretrained(
                    self.config.tokenizer, model_max_length=self.config.text_max_seq_len
                )
                self.tokenized_codes = self.tokenize_metadata(
                    tokenizer, self.code_metadata, special_tokens={self.config.EOS_TOKEN_ID: "[CLS]"}
                )

    def read_shards(self):
        """Reads the split-specific subject shards from the MEDS dataset.

        This method scans the specified MEDS cohort directory for Parquet files, organizes them by split, and
        creates mappings between subjects and their respective shards.
        """
        all_shards = generate_subject_split_dict(Path(self.config.meds_cohort_dir) / "data")
        self.shards = {sp: subjs for sp, subjs in all_shards.items() if sp.startswith(f"{self.split}")}
        self.subj_map = {subj: sp for sp, subjs in self.shards.items() for subj in subjs}
        if not self.shards:
            logger.warning(
                f"No shards found for split {self.split}. Check the directory structure and file names."
            )
