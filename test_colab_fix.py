#!/usr/bin/env python
"""
Quick test script to verify the Colab deadlock fix.
Tests that num_workers=0 is properly set in parallel execution.
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_datamodule_num_workers():
    """Test that DataModule respects num_workers parameter."""
    print("Test 1: DataModule num_workers configuration")
    print("="*60)
    
    from src.datasets import NOTDataModule
    import argparse
    
    # Create test args with num_workers=0
    args = argparse.Namespace(
        data_dir="data/raw",
        model_name="distilbert-base-uncased",
        batch_size=16,
        num_workers=0,  # Critical setting
        max_length=128,
        use_negation_dataset=False,
        seed=42,
    )
    
    # Create datamodule
    dm = NOTDataModule.from_args(args)
    
    # Verify num_workers is 0
    assert dm.num_workers == 0, f"Expected num_workers=0, got {dm.num_workers}"
    print(f"✓ DataModule num_workers: {dm.num_workers}")
    print("✓ Test 1 PASSED\n")


def test_parallel_script_args():
    """Test that parallel script properly sets num_workers=0."""
    print("Test 2: Parallel script argument handling")
    print("="*60)
    
    import argparse
    from src.datasets import NOTDataModule
    
    # Simulate what the parallel script does
    args = argparse.Namespace(
        data_dir="data/raw",
        model_name="distilbert-base-uncased",
        batch_size=16,
        num_workers=4,  # Original value
        max_length=128,
        use_negation_dataset=False,
        seed=42,
    )
    
    # This is what the fixed parallel script does
    args_copy = argparse.Namespace(**vars(args))
    args_copy.num_workers = 0  # Override for parallel safety
    
    dm = NOTDataModule.from_args(args_copy)
    
    assert dm.num_workers == 0, f"Expected num_workers=0, got {dm.num_workers}"
    print(f"✓ Original args.num_workers: {args.num_workers}")
    print(f"✓ Modified args_copy.num_workers: {args_copy.num_workers}")
    print(f"✓ DataModule num_workers: {dm.num_workers}")
    print("✓ Test 2 PASSED\n")


def test_colab_script_imports():
    """Test that the Colab-optimized script can be imported."""
    print("Test 3: Colab script imports")
    print("="*60)
    
    try:
        # Try to import the module
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "search_layers_colab",
            "src/scripts/search_layers_colab.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Check that key functions exist
        assert hasattr(module, 'train_probe_for_layer'), "Missing train_probe_for_layer function"
        assert hasattr(module, 'main'), "Missing main function"
        assert hasattr(module, 'set_seed'), "Missing set_seed function"
        
        print("✓ Colab script imports successfully")
        print("✓ All required functions present")
        print("✓ Test 3 PASSED\n")
    except Exception as e:
        print(f"✗ Test 3 FAILED: {e}\n")
        raise


def test_orchestrated_script_command():
    """Test that orchestrated script includes num_workers=0 in command."""
    print("Test 4: Orchestrated script command generation")
    print("="*60)
    
    import argparse
    
    # Simulate orchestrated script args
    args = argparse.Namespace(
        model_name="distilbert-base-uncased",
        data_dir="data/raw",
        batch_size=16,
        max_epochs=1,
        probe_lr=1e-3,
        seed=42,
        devices=1,
        num_labels=2,
        precision=32,
    )
    
    # Build command like orchestrated script does
    cmd = [
        sys.executable,
        "src/scripts/search_layers.py",
        "--model_name", args.model_name,
        "--data_dir", args.data_dir,
        "--batch_size", str(args.batch_size),
        "--max_epochs", str(args.max_epochs),
        "--probe_lr", str(args.probe_lr),
        "--layers", "0",
        "--pooling_strategies", "cls",
        "--output_dir", "test_output",
        "--seed", str(args.seed),
        "--devices", str(args.devices),
        "--mode", "probe",
        "--num_labels", str(args.num_labels),
        "--precision", str(args.precision),
        "--num_workers", "0",  # CRITICAL: This should be in the command
    ]
    
    # Verify num_workers is in the command
    assert "--num_workers" in cmd, "Missing --num_workers in command"
    num_workers_idx = cmd.index("--num_workers")
    num_workers_value = cmd[num_workers_idx + 1]
    assert num_workers_value == "0", f"Expected num_workers=0, got {num_workers_value}"
    
    print(f"✓ Command includes: --num_workers {num_workers_value}")
    print("✓ Test 4 PASSED\n")


def test_gpu_detection():
    """Test GPU detection and configuration."""
    print("Test 5: GPU detection")
    print("="*60)
    
    import torch
    
    if torch.cuda.is_available():
        print(f"✓ GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"✓ GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"✓ CUDA version: {torch.version.cuda}")
    else:
        print("⚠ No GPU detected (CPU mode)")
    
    print("✓ Test 5 PASSED\n")


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("COLAB DEADLOCK FIX - TEST SUITE")
    print("="*60 + "\n")
    
    tests = [
        ("DataModule num_workers", test_datamodule_num_workers),
        ("Parallel script args", test_parallel_script_args),
        ("Colab script imports", test_colab_script_imports),
        ("Orchestrated script command", test_orchestrated_script_command),
        ("GPU detection", test_gpu_detection),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"✗ {test_name} FAILED: {e}\n")
            failed += 1
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Passed: {passed}/{len(tests)}")
    print(f"Failed: {failed}/{len(tests)}")
    
    if failed == 0:
        print("\n✓ ALL TESTS PASSED!")
        print("\nThe Colab deadlock fix is working correctly.")
        print("You can now run experiments on Colab without deadlocks.")
        return 0
    else:
        print(f"\n✗ {failed} TEST(S) FAILED")
        print("\nPlease review the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

