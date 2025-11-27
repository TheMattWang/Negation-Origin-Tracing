import argparse

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train sentiment / negation models with PyTorch Lightning"
    )

    # ----- Experiment -----
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="sst2_finetune",
        help="Name for logging/checkpoint folders.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["finetune", "probe"],
        default="finetune",
        help="Training mode: finetune full model or train a linear probe.",
    )
    
    # ----- Probe-specific arguments -----
    parser.add_argument(
        "--probe_layer",
        type=int,
        default=None,
        help="Layer index to probe (0-based). If None, probes all layers. Only used in probe mode.",
    )
    parser.add_argument(
        "--pooling_strategy",
        type=str,
        choices=["cls", "mean", "token"],
        default="cls",
        help="Pooling strategy for probe: 'cls' for [CLS] token, 'mean' for mean pooling, 'token' for token around 'not'.",
    )
    parser.add_argument(
        "--probe_lr",
        type=float,
        default=1e-3,
        help="Learning rate for linear probe (only used in probe mode).",
    )

    # ----- Data -----
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/raw",
        help="Root directory containing train/validation/test parquet files.",
    )

    # ----- Model -----
    parser.add_argument(
        "--model_name",
        type=str,
        default="distilbert-base-uncased",
        help="Hugging Face model name or path.",
    )
    parser.add_argument(
        "--num_labels",
        type=int,
        default=2,
        help="Number of classes for classification.",
    )

    # ----- Training -----
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size per device.",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=3,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-5,
        help="Learning rate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )

    # ----- Hardware -----
    parser.add_argument(
        "--devices",
        type=int,
        default=1,
        help="Number of devices (GPUs or CPUs) to use.",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=32,
        choices=[16, 32],
        help="Training precision.",
    )

    return parser


def parse_args():
    parser = get_parser()
    return parser.parse_args()
