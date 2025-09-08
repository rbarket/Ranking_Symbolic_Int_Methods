# Tree-Based Deep learning for Ranking Symbolic Integration Algorithms
Pytorch implementation of Tree-Based Deep Learning to Rank Symbolic Integration Algorithms in Maple, available on [arxiv](https://www.arxiv.org/abs/2508.06383)

Please see the [example notebook](predict_method_example.ipynb) for an example of how this model will make a prediction of which method to use in Maple given an integrand!

## Requirements
The exact conda environment used for training and inference can be recreated by running the command `conda env create -f environment.yml` at the top level of this project directory. We used CUDA 12.1 and the GPU version of PyTorch to train the model. In general, if you wish to create your own environment, you need the following packages:
 - pytorch
 - pandas
 - pyarrow (format data is stored in)
 - pyyaml (for config.py) 

## Data
The preprocessed data is available on Zenodo, ready to use right away for machine learning training. Download the data here: [Zenodo](https://zenodo.org/records/16754656). It is split between train and test, and contains both elementary and nonelementary functions. We train on a mix of both, but the post-training inference will show results for individual and combined types/ Note that for the tree tranformer, we precomputed the positional encodings to significantly save time, and is saved in Zenodo as `precomputed_positions.pt`. Download this file and the train and test data into the [data/processed](data/processed) folder in the root directory.

For the label column in the dataset, the value of each item in the list is the DAG sizes which correspond to the following method order: "default", "derivativedivides", "parts", "risch", "norman", "trager", "parallelrisch", "meijerg", "elliptic", "pseudoelliptic", "lookup", "gosper", "orering". To see these sub-methods that `int` calls in Maple, see the [help page](https://www.maplesoft.com/support/help/maple/view.aspx?path=int%2fmethods). The DAG sizes were acquired by taking the integrand, running the integrand through Maple's `int` command with each available method, and then recording the DAG size of the output.

## Training
The main script to produce a trained model is in [scripts/train.py](scripts/train.py). From the root of the project, this can be run with `python -m scripts.train`. All main configurations for any model hyperparameters can be found in [configs/train_configs.yaml](configs/train_configs.yaml). The parameters in this file were the ones used for the best saved model. You may change these parameters as you please. However, if changing the depth parameter to anything higher than what is currently listed in the config file (20 right now), then you need to precompute a new set of tree positional encodings. This can be done with running `python -m scripts.precompute_positions` after changing the config file.

During training, the model will checkpoint after each epoch, as well as what the current best model is. If you wish to resume training, you must provide the path to the checkpointed model, otherwise the script will train from scratch. To resume training, either add the command `--resume_from path/to/model.pth` or provide the path in the parameter `resume_from` in the config file. Note that because we use use the OneCycleLR learning rate scheduler, the final number of epochs must not have changed from when you first started training the model (this can be fixed but is not part of this implementation).

## Testing
Once a model is trained, we can evaluate the model with [scripts/inference.py](scripts/inference.py) by running the command `python -m scripts.inference --checkpoint_path <PATH TO MODEL>.pth`. In `<PATH TO MODEL>`, you pass any model you wish to evaluate. This repository includes a trained model that produces the results in the paper, with path `models/ranking/ranking_best.pth`. This will produce the results for the exact matches in Figure 10(a) of the paper. The predictions are also saved in pytorch pt format for further analysis.  
