# ---------------------------------------------------------------------------
# Load trained mslp + wind transformer and make predictions
# Salika Thilakarathne
# Helmholtz-Zentrum Hereon
# March, 2026
# ---------------------------------------------------------------------------

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path

from train_wave_transformer import AtmosWaveTransformer


def load_model_checkpoint(ckpt_path: str, device: str = None):
    """
    load the trained model and normalisation statistics from a checkpoint file.

    returns:
        model  : pytorch module in eval mode
        y_mean : tensor of shape (1, output_dim)
        y_std  : tensor of shape (1, output_dim)
        h      : dict of hyperparameters
    """
    if device is None:  #in helper functions...
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # safe here because we only load checkpoints we generated ourselves
    checkpoint = torch.load(ckpt_path, map_location=device)

    h = checkpoint["hyperparams"]
    model = AtmosWaveTransformer(
        d_model=h["d_model"],
        nhead=h["nhead"],
        num_layers=h["num_layers"],
        d_ff=h["d_ff"],
        dropout=h["dropout"],
        spatial_dim=h["spatial_dim"],   # new-cnn output dimension
        base_channels=h["base_channels"],
        output_dim=h["output_dim"],
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    y_mean = checkpoint["normalisation"]["y_mean"].to(device)
    y_std = checkpoint["normalisation"]["y_std"].to(device)

    return model, y_mean, y_std, h


def predict_wave(
    model: nn.Module,
    y_mean: torch.Tensor,
    y_std: torch.Tensor,
    X_mslp_in: torch.Tensor,
    X_windU_in: torch.Tensor,
    X_windV_in: torch.Tensor,
    device: str = None,
) -> torch.Tensor:
    """
    run the model on input tensors and return unnormalised predictions.

    inputs:
        X_mslp_in  : (N, T, lat, lon) torch.float32
        X_windU_in : (N, T, lat, lon) torch.float32
        X_windV_in : (N, T, lat, lon) torch.float32

    output:
        preds_ph : (N, 4) tensor in physical units [hs, t0, thq_sin, thq_cos]
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(device)

    X_mslp_in = X_mslp_in.to(device)
    X_windU_in = X_windU_in.to(device)
    X_windV_in = X_windV_in.to(device)

    model.eval()
    with torch.no_grad():
        preds_norm = model(X_mslp_in, X_windU_in, X_windV_in)
        preds_ph = preds_norm * y_std + y_mean

    return preds_ph


if __name__ == "__main__":
    """
    direct execution:
    - loads the latest CNN + transformer checkpoint
    - loads pre-built tensors from disk
    - selects the test dataset (last split)
    - runs predictions on the test set
    """

    import pandas as pd
    import numpy as np

    # directory paths
    models_dir = Path("../models")
    tensor_ckpt_path = Path("../data/prepared_tensors/wave_mslp_wind_tensors_4d.pt")

    # find latest model checkpoint
    ckpts = sorted(models_dir.glob("*_mslp_wind_transformer_4d.pt"))
    if not ckpts:
        raise FileNotFoundError("no transformer checkpoints found in ../models.")
    ckpt_path = ckpts[-1]  # to get the latest model run. change this if you prefer to hand pick the model.
    print(f"loading checkpoint: {ckpt_path}")

    # load model and normalisation stats
    model, y_mean, y_std, h = load_model_checkpoint(str(ckpt_path))

    # load tensors (prepared earlier in the codes / preprocessing script)
    if not tensor_ckpt_path.is_file():
        raise FileNotFoundError(
            f"could not find prepared tensor file: {tensor_ckpt_path}\n"
            "make sure you have run the preprocessing script that saves X_mslp, X_windU, X_windV, y."
        )

    print(f"loading tensors from: {tensor_ckpt_path}")
    tckpt = torch.load(tensor_ckpt_path, map_location="cpu")

    X_mslp = tckpt["X_mslp"]
    X_windU = tckpt["X_windU"]
    X_windV = tckpt["X_windV"]
    y = tckpt["y"]
    y_dates = tckpt["y_dates"]  # list of timestamp strings

    # basic alignment check
    assert (
        X_mslp.shape[0]
        == X_windU.shape[0]
        == X_windV.shape[0]
        == y.shape[0]
        == len(y_dates)
    ), "mismatch in number of samples between inputs, target, and dates."

    N = X_mslp.shape[0]
    ratio = 0.7894620698765736
    n_train = int(0.8166789314257354 * ratio * N)
    n_val = int(ratio * N)

    # select test dataset (last 20 percent) exactly as in training
    X_mslp_test = X_mslp[n_val:]
    X_windU_test = X_windU[n_val:]
    X_windV_test = X_windV[n_val:]
    y_test = y[n_val:]
    test_dates = y_dates[n_val:]

    print(f"number of test samples: {X_mslp_test.shape[0]}")
    print(f"test period begins at: {test_dates[0]}")
    print(f"test period ends   at: {test_dates[-1]}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    batch_size = 512

    dl_test = DataLoader(
        TensorDataset(X_mslp_test, X_windU_test, X_windV_test),
        batch_size=batch_size,
        shuffle=False,
    )

    model.eval()
    preds = []
    
    with torch.no_grad():
        for xb_mslp, xb_u, xb_v in dl_test:
            xb_mslp = xb_mslp.to(device)
            xb_u = xb_u.to(device)
            xb_v = xb_v.to(device)
    
            preds_norm = model(xb_mslp, xb_u, xb_v)
            preds_ph_batch = preds_norm * y_std + y_mean
    
            preds.append(preds_ph_batch.cpu())
    
    # concatenate all batches
    preds_ph = torch.cat(preds, dim=0)

    # convert predictions and dates to a dataframe for later analysis
    pred_np = preds_ph.cpu().numpy()
    test_index = pd.to_datetime(test_dates)

    df_pred = pd.DataFrame(
        {
            "hs_pred": pred_np[:, 0],
            "t0_pred": pred_np[:, 1],
            "thq_sin_pred": pred_np[:, 2],
            "thq_cos_pred": pred_np[:, 3],
        },
        index=test_index,
    )

    # derive predicted direction in radians and degrees
    df_pred["thq_rad_pred"] = np.arctan2(
        df_pred["thq_sin_pred"], df_pred["thq_cos_pred"]
    )
    df_pred["thq_rad_pred"] = np.mod(df_pred["thq_rad_pred"], 2 * np.pi)
    df_pred["thq_deg_pred"] = np.degrees(df_pred["thq_rad_pred"])

    print("example of prediction dataframe:")
    print(df_pred.head())