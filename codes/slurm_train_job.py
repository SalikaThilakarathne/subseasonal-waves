# ---------------------------------------------------------------------------
# run transformer training using hyperparameters from transformer_hyperparams.txt
# Salika Thilakarathne
# Helmholtz-Zentrum Hereon
# March, 2026
# ---------------------------------------------------------------------------

import torch
from pathlib import Path

from train_wave_transformer import train_and_save_model

def load_tensors(tensor_ckpt_path: Path):
    """
    load pre-built tensors (X_mslp, X_windU, X_windV, y, y_dates).
    """
    if not tensor_ckpt_path.is_file():
        raise FileNotFoundError(f"tensor file not found: {tensor_ckpt_path}")

    print(f"loading tensors from: {tensor_ckpt_path}")
    ckpt = torch.load(tensor_ckpt_path, map_location="cpu")

    X_mslp  = ckpt["X_mslp"]
    X_windU = ckpt["X_windU"]
    X_windV = ckpt["X_windV"]
    y       = ckpt["y"]
    y_dates = ckpt["y_dates"]

    assert len(X_mslp) == len(X_windU) == len(X_windV) == len(y) == len(y_dates), \
        "input tensors have inconsistent lengths."

    print("tensor shapes:")
    print("  X_mslp :", X_mslp.shape)
    print("  X_windU:", X_windU.shape)
    print("  X_windV:", X_windV.shape)
    print("  y      :", y.shape)

    return X_mslp, X_windU, X_windV, y, y_dates


def main():
    base = Path(__file__).resolve().parent

    poi = 'SEE'
    tensor_file = base / f"../data/prepared_tensors/wave_mslp_wind_tensors_sl{poi}.pt"
    model_dir   = base / "../models"

    # load tensors
    X_mslp, X_windU, X_windV, y, y_dates = load_tensors(tensor_file)

    # training automatically loads hyperparameters from:
    # ../config/transformer_hyperparams.txt
    train_and_save_model(
        X_mslp=X_mslp,
        X_windU=X_windU,
        X_windV=X_windV,
        y=y,
        y_dates=y_dates,
        models_dir=str(model_dir),
        poi=poi,
    )


if __name__ == "__main__":
    main()