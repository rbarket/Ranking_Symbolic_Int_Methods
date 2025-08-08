# Tree-Based Deep learning for Ranking Symbolic Integration Algorithms
Pytorch implementation of Tree Transformers to Rank Symbolic Integration Algorithms in Maple

## Data
The preprocessed data is available on Zenodo, ready to use right away for machine learning training. Download the data here: [Zenodo](https://zenodo.org/records/16752399). It is split between train and test, and furthermore between elementary and non-elementary expressions. We train on a mix of both, but the post-training analysis can be done split if you wish to see the results between elementary and non-elementary expressions. Note that for the tree tranformer, we precomputed the positional encodings to significantly save time, and is saved in Zenodo as `precomputed_positions.pt`. Download this file and the train and test data into the `data` folder in the root directory.

## Training
The main script to produce a trained model is in `scripts/train.py`. From the root of the project, this can be run with `python -m scripts.train`. All main configurations for any model hyperparameters can be found in `configs/train_configs.yaml`. The parameters in this file were the ones used for the best saved model. You may change these parameters as you please. However, if changing the depth parameter to anything higher than what is currently listed in the config file (20 right now), then you need to precompute a new set of tree positional encodings. This can be done with running `python -m scripts.precompute_positions` after changing the config file.

## Testing
Once a model is trained, we can evaluate the model with `scripts/inference.py` by running the command `python -m scripts.inference --checkpoint_path <PATH TO MODEL>.pth`. In `<PATH TO MODEL>`, you pass any model you wish to evaluate. This repository includes a trained model that produces the results in the paper, with path `models/ranking/ranking_best.pth`. This will produce the results for the exact matches in Figure 10(a) of the paper. The predictions are also saved in pytorch pt format for further analysis.  
