# Troubleshooting ckptplan

Use this guide to distinguish setup problems, expected limitations, and missing benchmark evidence. It summarizes the behavior documented in this checkout; it does not introduce new performance claims. Commands are usage examples, not a record of tests run while writing this guide.

## Supported Python environment

If installation rejects your Python version, use Python 3.10–3.12, the range documented by this checkout. Create a fresh virtual environment with a supported interpreter, activate it, and install from the repository root:

```sh
python -m pip install -e ".[dev]"
```

Do not remove the Python version constraint to force installation. See [contributor setup](../CONTRIBUTING.md#python-development).
