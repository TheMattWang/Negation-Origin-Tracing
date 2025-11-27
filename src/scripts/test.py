import os
import argparse
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger

from src.models import BaseModule
from src.datasets import NOTDataModule
from src.utils.parser import get_parser


def main():
    """Test/evaluate a trained model."""
    parser = get_parser()
    
    # Add test-specific arguments
    parser.add_argument(
        "--ckpt_path",
        type=str,
        required=True,
        help="Path to model checkpoint to load.",
    )
    parser.add_argument(
        "--test_only",
        action="store_true",
        help="Only run test, skip validation.",
    )
    
    args = parser.parse_args()
    
    # Load model from checkpoint
    print(f"Loading model from checkpoint: {args.ckpt_path}")
    model = BaseModule.load_from_checkpoint(
        args.ckpt_path,
        strict=False,  # Allow partial loading
    )
    
    # Initialize data module
    print(f"\nInitializing data module...")
    datamodule = NOTDataModule.from_args(args)
    
    # Setup logger (optional, for logging test results)
    logger = TensorBoardLogger(
        save_dir="experiments",
        name="runs",
        version=f"{args.experiment_name}_test",
    )
    
    # Initialize trainer
    trainer = L.Trainer(
        devices=args.devices,
        precision=args.precision,
        logger=logger,
        enable_progress_bar=True,
        enable_model_summary=False,
    )
    
    # Run validation if requested
    if not args.test_only:
        print(f"\n{'='*60}")
        print(f"Running validation...")
        print(f"{'='*60}\n")
        
        datamodule.setup("validate")
        trainer.validate(
            model=model,
            datamodule=datamodule,
        )
    
    # Run test
    print(f"\n{'='*60}")
    print(f"Running test evaluation...")
    print(f"{'='*60}\n")
    
    datamodule.setup("test")
    trainer.test(
        model=model,
        datamodule=datamodule,
    )
    
    print(f"\n{'='*60}")
    print(f"Evaluation complete!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

