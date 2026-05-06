# Wave Forecasts

The Python source code, Jupyter notebooks, and shell scripts used for data processing, parallel execution, and GPU-accelerated deep-learning experiments.

## Structure

- `codes/`    Core source code (Python modules and functions)
- `config/`   Configuration files (parameters, settings)
- `models/`   Model definitions or trained models
- `results/`  Generated outputs (plots, tables, analysis results)
- `scripts/`  Executable scripts (run, preprocess, train, evaluate)

## Usage
Describe how to run your code here.

## Notes
- `data/` is ignored and are provided locally.
- embed_dim must be divisible by num_heads: so your d_model should be divisable by nhead. otherwise grid search would result in an error. check the /config/transformer_grid.txt
``