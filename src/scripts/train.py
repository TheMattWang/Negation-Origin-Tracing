import os
import sys
from pathlib import Path
import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.parser import parse_args
from src.models import BaseModule
from src.datasets import NOTDataModule


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Set deterministic algorithms (may impact performance)
    # torch.use_deterministic_algorithms(True)


def main():
    """Main training function."""
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Create output directories
    experiment_dir = os.path.join("experiments", "runs", args.experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Initialize model
    print(f"Initializing model: {args.model_name}")
    print(f"Mode: {args.mode}")
    if args.mode == "probe":
        print(f"  Probe layer: {args.probe_layer if args.probe_layer is not None else 'all'}")
        print(f"  Pooling strategy: {args.pooling_strategy}")
        print(f"  Probe learning rate: {args.probe_lr}")
    else:
        print(f"  Learning rate: {args.lr}")
    
    model = BaseModule.from_args(args)
    
    # Initialize data module
    print(f"\nInitializing data module...")
    print(f"  Data directory: {args.data_dir}")
    print(f"  Batch size: {args.batch_size}")
    
    datamodule = NOTDataModule.from_args(args)
    datamodule.setup("fit")  # Setup train/val splits
    
    print(f"  Training samples: {len(datamodule.train_dataset)}")
    print(f"  Validation samples: {len(datamodule.val_dataset)}")
    
    # Setup logging
    logger = TensorBoardLogger(
        save_dir="experiments",
        name="runs",
        version=args.experiment_name,
    )
    
    # Setup callbacks
    callbacks = []
    
    # Model checkpointing
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(experiment_dir, "checkpoints"),
        filename="{epoch:02d}-{val_loss:.3f}-{val_acc:.3f}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        save_last=True,
        verbose=True,
    )
    callbacks.append(checkpoint_callback)
    
    # Early stopping
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=5,
        verbose=True,
    )
    callbacks.append(early_stop_callback)
    
    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks.append(lr_monitor)
    
    # Initialize trainer
    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        devices=args.devices,
        precision=args.precision,
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=50,
        val_check_interval=0.5,  # Validate twice per epoch
        enable_progress_bar=True,
        enable_model_summary=True,
        deterministic=False,  # Set to True for full reproducibility (slower)
    )
    
    # Train
    print(f"\n{'='*60}")
    print(f"Starting training...")
    print(f"{'='*60}\n")
    
    trainer.fit(
        model=model,
        datamodule=datamodule,
    )
    
    # Test
    print(f"\n{'='*60}")
    print(f"Running evaluation on test set...")
    print(f"{'='*60}\n")
    
    datamodule.setup("test")  # Setup test split
    trainer.test(
        model=model,
        datamodule=datamodule,
        ckpt_path="best",  # Use best checkpoint
    )
    
    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Best model checkpoint: {checkpoint_callback.best_model_path}")
    print(f"Logs saved to: {logger.log_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
