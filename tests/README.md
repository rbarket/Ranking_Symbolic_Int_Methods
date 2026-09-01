# Test Suite

Run this test suite after modifying model, data-loading, training, checkpoint, or inference code. All applicable tests should pass before changes are committed or pushed.

## Run the regular suite

From the project root, activate the Python 3.10 training environment and run:

```bash
python -m unittest discover -s tests -v
```

The command should exit with status `0` and finish with `OK`. The current suite discovers 49 tests. In an environment without DGL, the 11 TreeLSTM-specific tests are expected to be reported as skipped; failures and errors are not expected.

## Run the complete TreeLSTM suite

To execute the DGL tests as well, run the suite in the TreeLSTM environment:

```bash
conda run -n TreeLSTM_DGL python -m unittest discover -s tests -v
```

The current complete result is 49 tests passing with no failures, errors, or skips. Some PyTorch or DGL warnings may be printed and do not indicate failure when the final result is `OK`.

The tests create their own small temporary datasets and checkpoints, so the full downloaded training dataset is not required. When adding or changing behavior, update the relevant tests while preserving a successful final result.
