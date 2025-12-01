"""
Test script to verify the setup works correctly.
Tests data loading, model initialization, training, and interventions.
"""

import os
import sys
import torch
import traceback
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def test_imports():
    """Test that all imports work."""
    print("=" * 60)
    print("Testing imports...")
    print("=" * 60)
    
    try:
        from src.datasets import NOTDataModule, SentimentDataset, NegationPairDataset
        print("✓ Datasets imported successfully")
        
        from src.models import BaseModule
        print("✓ Models imported successfully")
        
        from src.engine.causal_interventions import CausalInterventionRunner
        from src.engine.metrics import InterventionMetrics
        print("✓ Engine modules imported successfully")
        
        from src.utils.parser import parse_args
        print("✓ Utils imported successfully")
        
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        traceback.print_exc()
        return False


def test_data_loading():
    """Test data loading."""
    print("\n" + "=" * 60)
    print("Testing data loading...")
    print("=" * 60)
    
    try:
        from src.datasets import NOTDataModule
        
        # Check if data exists
        data_dir = "data/raw"
        train_path = os.path.join(data_dir, "train", "sst.parquet")
        
        if not os.path.exists(train_path):
            print(f"⚠ Warning: Training data not found at {train_path}")
            print("  Run: python src/data/download.py")
            return False
        
        # Initialize datamodule
        datamodule = NOTDataModule(
            data_dir=data_dir,
            model_name="distilbert-base-uncased",
            batch_size=4,
            num_workers=0,  # Use 0 for testing
            max_length=128,
        )
        
        # Setup
        datamodule.setup("fit")
        
        print(f"✓ DataModule initialized")
        print(f"  Training samples: {len(datamodule.train_dataset)}")
        print(f"  Validation samples: {len(datamodule.val_dataset)}")
        
        # Test data loader
        train_loader = datamodule.train_dataloader()
        batch = next(iter(train_loader))
        
        print(f"✓ DataLoader works")
        print(f"  Batch keys: {batch.keys()}")
        print(f"  Input IDs shape: {batch['input_ids'].shape}")
        print(f"  Labels shape: {batch['labels'].shape}")
        
        return True
    except Exception as e:
        print(f"✗ Data loading failed: {e}")
        traceback.print_exc()
        return False


def test_model_initialization():
    """Test model initialization."""
    print("\n" + "=" * 60)
    print("Testing model initialization...")
    print("=" * 60)
    
    try:
        from src.models import BaseModule
        
        # Test finetune mode
        print("\nTesting finetune mode...")
        model_finetune = BaseModule(
            model_name="distilbert-base-uncased",
            num_labels=2,
            mode="finetune",
            lr=2e-5,
        )
        print("✓ Finetune model initialized")
        
        # Test probe mode
        print("\nTesting probe mode...")
        model_probe = BaseModule(
            model_name="distilbert-base-uncased",
            num_labels=2,
            mode="probe",
            probe_layer=5,
            pooling_strategy="cls",
            probe_lr=1e-3,
        )
        print("✓ Probe model initialized")
        print(f"  Probe layers: {model_probe.probe_layers}")
        print(f"  Number of probes: {len(model_probe.probes)}")
        
        return True, model_finetune, model_probe
    except Exception as e:
        print(f"✗ Model initialization failed: {e}")
        traceback.print_exc()
        return False, None, None


def test_forward_pass(model_finetune, model_probe):
    """Test forward pass."""
    print("\n" + "=" * 60)
    print("Testing forward pass...")
    print("=" * 60)
    
    try:
        # Create dummy batch
        batch = {
            "input_ids": torch.randint(0, 1000, (2, 128)),
            "attention_mask": torch.ones(2, 128),
            "labels": torch.tensor([0, 1]),
        }
        
        # Test finetune forward
        print("\nTesting finetune forward pass...")
        with torch.no_grad():
            outputs = model_finetune(batch)
            print(f"✓ Finetune forward pass works")
            print(f"  Output keys: {outputs.keys()}")
            if "logits" in outputs:
                print(f"  Logits shape: {outputs['logits'].shape}")
        
        # Test probe forward
        print("\nTesting probe forward pass...")
        with torch.no_grad():
            outputs = model_probe(batch)
            print(f"✓ Probe forward pass works")
            print(f"  Output keys: {outputs.keys()}")
            if "logits" in outputs:
                print(f"  Logits shape: {outputs['logits'].shape}")
        
        return True
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        traceback.print_exc()
        return False


def test_training_step(model_finetune, model_probe):
    """Test training step."""
    print("\n" + "=" * 60)
    print("Testing training step...")
    print("=" * 60)
    
    try:
        # Create dummy batch
        batch = {
            "input_ids": torch.randint(0, 1000, (2, 128)),
            "attention_mask": torch.ones(2, 128),
            "labels": torch.tensor([0, 1]),
        }
        
        # Test finetune training step
        print("\nTesting finetune training step...")
        loss = model_finetune.training_step(batch, 0)
        print(f"✓ Finetune training step works")
        print(f"  Loss: {loss.item():.4f}")
        
        # Test probe training step
        print("\nTesting probe training step...")
        loss = model_probe.training_step(batch, 0)
        print(f"✓ Probe training step works")
        print(f"  Loss: {loss.item():.4f}")
        
        return True
    except Exception as e:
        print(f"✗ Training step failed: {e}")
        traceback.print_exc()
        return False


def test_optimizer(model_finetune, model_probe):
    """Test optimizer configuration."""
    print("\n" + "=" * 60)
    print("Testing optimizer configuration...")
    print("=" * 60)
    
    try:
        # Test finetune optimizer
        optimizer_finetune = model_finetune.configure_optimizers()
        print(f"✓ Finetune optimizer configured")
        print(f"  Optimizer type: {type(optimizer_finetune).__name__}")
        print(f"  Number of parameter groups: {len(optimizer_finetune.param_groups)}")
        
        # Test probe optimizer
        optimizer_probe = model_probe.configure_optimizers()
        print(f"✓ Probe optimizer configured")
        print(f"  Optimizer type: {type(optimizer_probe).__name__}")
        print(f"  Number of parameter groups: {len(optimizer_probe.param_groups)}")
        
        return True
    except Exception as e:
        print(f"✗ Optimizer configuration failed: {e}")
        traceback.print_exc()
        return False


def test_causal_interventions():
    """Test causal intervention setup."""
    print("\n" + "=" * 60)
    print("Testing causal interventions...")
    print("=" * 60)
    
    try:
        from transformers import AutoModel
        from src.engine.causal_interventions import CausalInterventionRunner
        
        # Load base model
        base_model = AutoModel.from_pretrained("distilbert-base-uncased")
        base_model.eval()
        
        # Initialize runner
        runner = CausalInterventionRunner(
            model=base_model,
            probe=None,
            device="cpu"
        )
        print("✓ CausalInterventionRunner initialized")
        
        # Create dummy batches
        batch1 = {
            "input_ids": torch.randint(0, 1000, (2, 128)),
            "attention_mask": torch.ones(2, 128),
            "labels": torch.tensor([0, 1]),
        }
        batch2 = {
            "input_ids": torch.randint(0, 1000, (2, 128)),
            "attention_mask": torch.ones(2, 128),
            "labels": torch.tensor([1, 0]),
        }
        
        # Test activation patcher (basic setup)
        print("\nTesting activation patcher setup...")
        patcher = runner.patcher
        print("✓ ActivationPatcher accessible")
        
        # Test ablator (basic setup)
        print("\nTesting targeted ablation setup...")
        ablator = runner.ablator
        print("✓ TargetedAblation accessible")
        
        return True
    except Exception as e:
        print(f"✗ Causal intervention setup failed: {e}")
        traceback.print_exc()
        return False


def test_metrics():
    """Test metrics computation."""
    print("\n" + "=" * 60)
    print("Testing metrics...")
    print("=" * 60)
    
    try:
        from src.engine.metrics import (
            compute_label_flip_rate,
            compute_logit_delta,
            InterventionMetrics,
        )
        
        # Test label flip rate
        original_preds = torch.tensor([0, 1, 0, 1])
        intervened_preds = torch.tensor([1, 1, 0, 0])
        flip_rate = compute_label_flip_rate(original_preds, intervened_preds)
        print(f"✓ Label flip rate computation works")
        print(f"  Flip rate: {flip_rate:.2f}")
        
        # Test logit delta
        original_logits = torch.randn(2, 2)
        intervened_logits = torch.randn(2, 2)
        delta = compute_logit_delta(original_logits, intervened_logits)
        print(f"✓ Logit delta computation works")
        print(f"  Delta: {delta:.4f}")
        
        # Test metrics tracker
        metrics = InterventionMetrics()
        metrics.add_result(
            experiment_type="activation_patching",
            layer_idx=5,
            label_flips=2,
            logit_delta=0.5,
            accuracy=0.8,
        )
        summary = metrics.get_layer_summary("activation_patching")
        print(f"✓ InterventionMetrics works")
        print(f"  Summary keys: {summary.keys()}")
        
        return True
    except Exception as e:
        print(f"✗ Metrics computation failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("SETUP TEST SUITE")
    print("=" * 60)
    
    results = {}
    
    # Test imports
    results["imports"] = test_imports()
    if not results["imports"]:
        print("\n✗ Critical: Imports failed. Please check dependencies.")
        return
    
    # Test data loading
    results["data"] = test_data_loading()
    
    # Test model initialization
    model_init_success, model_finetune, model_probe = test_model_initialization()
    results["model_init"] = model_init_success
    
    if not model_init_success:
        print("\n✗ Critical: Model initialization failed.")
        return
    
    # Test forward pass
    results["forward"] = test_forward_pass(model_finetune, model_probe)
    
    # Test training step
    results["training"] = test_training_step(model_finetune, model_probe)
    
    # Test optimizer
    results["optimizer"] = test_optimizer(model_finetune, model_probe)
    
    # Test causal interventions
    results["interventions"] = test_causal_interventions()
    
    # Test metrics
    results["metrics"] = test_metrics()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED!")
        print("=" * 60)
        print("\nYour setup is ready to use!")
    else:
        print("\n" + "=" * 60)
        print("✗ SOME TESTS FAILED")
        print("=" * 60)
        print("\nPlease fix the failing tests before proceeding.")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

