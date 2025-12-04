# Test Scripts

This directory contains test scripts for verifying functionality.

## Tests

- **`test_colab_fix.py`** - Verify the Colab deadlock fix is working correctly
  ```bash
  python scripts/tests/test_colab_fix.py
  ```

- **`test_resume.py`** - Verify resume functionality works correctly
  ```bash
  python scripts/tests/test_resume.py
  ```

## When to Use

Run these tests to verify that:
- Colab deadlock fix is properly configured (`test_colab_fix.py`)
- Resume functionality is working (`test_resume.py`)

These are primarily for development/verification purposes.

