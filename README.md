# Tree-Based Deep learning for Ranking Symbolic Integration Algorithms
Pytorch implementation of Tree-Based Deep Learning to Rank Symbolic Integration Algorithms in Maple, available on [arxiv](https://www.arxiv.org/abs/2508.06383).

Please see the [example notebook](predict_method_example.ipynb) for an example of how this model will make a prediction of which method to use in Maple given an integrand!

## Requirements
The exact conda environment used for training and inference can be recreated by running the command `conda env create -f environment.yml` at the top level of this project directory. We used CUDA 12.1 and the GPU version of PyTorch to train the model. In general, if you wish to create your own environment, you need the following packages:
 - pytorch
 - pandas
 - pyarrow (format data is stored in)
 - pyyaml (for config.py) 

## Data
The preprocessed data is available on [Zenodo](https://zenodo.org/records/16754656) and is not stored in this repository. Download the Airy-inclusive `train_data.parquet` and `test_data.parquet` files into [data/processed/airy](data/processed/airy). Download `precomputed_positions.pt` into [data](data); it is required only by TreeTransformer. The superseded non-Airy splits may be retained in [data/processed/old](data/processed/old).

The committed vocabulary matches the Airy-inclusive dataset. To regenerate it deterministically after replacing the dataset, run `python -m scripts.create_vocab --config configs/train_tree_transformer_config.yaml`.

For the label column in the dataset, the value of each item in the list is the DAG sizes which correspond to the following method order: "default", "derivativedivides", "parts", "risch", "norman", "trager", "parallelrisch", "meijerg", "elliptic", "pseudoelliptic", "lookup", "gosper", "orering". To see these sub-methods that `int` calls in Maple, see the [help page](https://www.maplesoft.com/support/help/maple/view.aspx?path=int%2fmethods). The DAG sizes were acquired by taking the integrand, running the integrand through Maple's `int` command with each available method, and then recording the DAG size of the output.

## Training
All models use the shared [training script](scripts/train.py). The architecture is selected by `model.type` in the YAML configuration. Supported values are `tree_transformer`, `transformer`, `lstm`, and `tree_lstm`. If `model.type` or `--config` is omitted, TreeTransformer is used by default.

The provided configurations are:

- `configs/train_tree_transformer_config.yaml`
- `configs/train_transformer_config.yaml`
- `configs/train_lstm_config.yaml`
- `configs/train_tree_lstm_config.yaml` (requires the DGL environment)

Run training from the project root, for example:

```bash
# Default: TreeTransformer
python -m scripts.train

# Select another model through its configuration
python -m scripts.train --config configs/train_transformer_config.yaml
```

Common overrides include `--epochs`, `--n`, `--eval_n`, `--device`, and `--save_dir`. TreeTransformer depth changes beyond the precomputed range require regenerating positions with `python -m scripts.precompute_positions`.

Training saves a latest checkpoint as `<experiment_name>.pth` and the best validation checkpoint as `<experiment_name>_best.pth`. Resume from the latest checkpoint with `python -m scripts.train --resume_from path/to/model.pth --epochs 10`. Here, `--epochs` is the total target: an epoch-3 checkpoint resumes at epoch 4 and trains through epoch 10. New checkpoints contain their configuration and vocabulary, so no YAML is required when resuming. Legacy checkpoints still require their matching `--config`.

## Inference
Run inference on any of the four models with:

```bash
python -m scripts.inference \
  --checkpoint_path path/to/model_best.pth \
  --split test
```

For new checkpoints, the model type, hyperparameters, and vocabulary are loaded directly from the checkpoint. Use `--config` only for a legacy checkpoint. Optional arguments include `--sample_n`, `--device`, `--batch_size`, `--num_workers`, and `--output_dir`. Inference saves aligned predictions and overall, source, and complexity metrics beside the checkpoint by default.

TreeLSTM training and inference must be run in the `TreeLSTM_DGL` environment. TreeLSTM does not support the current `--data_parallel` path.
