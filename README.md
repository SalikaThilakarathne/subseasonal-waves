# Wave-Emulator

Python workflows for atmospheric predictor preprocessing, tensor generation, Transformer-based wave modelling, and evaluation.

Author: Salika Thilakarathne
Affiliation: Helmholtz-Zentrum Hereon, Germany
Website: https://salikathilakarathne.github.io/

## Structure

- `codes/`    core source code (preprocessing, tensor generation, training, prediction
- `config/`   configuration files and hyperparameter settings
- `models/`   trained model checkpoints
- `results/`  generated outputs (predictions, plots, tables, and analysis)

## Usage

Typical workflow:
1. preprocess atmospheric and wave datasets
2. generate atmosphere–wave tensors
3. train Transformer models
4. generate predictions
5. evaluate model performance

## Notes
- `data/` and `scripts/` are ignored by version control and must be available locally.
- embed_dim (d_model) must be divisible by nhead. Invalid combinations will cause model initialisation errors. See config/transformer_grid.txt.
- all experiments were conducted on the hlre-4 levante system at dkrz using nvidia a100-sxm4 gpus and amd epyc 7763 cpus.
- transformer experiments were performed using 6 h, 12 h, and 24 h atmospheric history windows.
- SEE is the primary study location used for model development and window-sensitivity experiments.
- ELB is used as an independent transferability check