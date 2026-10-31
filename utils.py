"""Plotting utilities for the synthetic datasets and model forecasts.

Figures are saved as PNG files, defaulting to the FIGURES_DIR ("figures/")
folder when no explicit output path is given.
"""

import csv
import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

import nn
from data import TimeSeriesDataset, is_noisy_name

FIGURES_DIR = "figures"
SPLIT_COLORS = {"train": "tab:blue", "val": "tab:orange", "test": "tab:green"}


def make_windows(series: np.ndarray, context: int, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Args:
        series: 1D array, shape (T,).
        context: number of past steps used as input, C.
        horizon: number of future steps to predict, p.

    Returns:
        X: input windows, shape (num_windows, context).
        Y: target windows, shape (num_windows, horizon).
    """
    window_len = context + horizon
    if len(series) < window_len:
        raise ValueError(f"series of length {len(series)} is shorter than context + horizon = {window_len}")
    windows = np.lib.stride_tricks.sliding_window_view(series, window_len)
    return windows[:, :context].copy(), windows[:, context:].copy()


def standardize_stats(train_series: np.ndarray) -> tuple[float, float]:
    """
    Args:
        train_series: 1D reference array used to compute the statistics, shape (T,).

    Returns:
        mean: scalar mean of `train_series`.
        std: scalar standard deviation of `train_series`, floored away from 0.
    """
    return float(train_series.mean()), float(train_series.std()) + 1e-8


def _resolve_path(path: Optional[str], default_name: str) -> str:
    path = path or os.path.join(FIGURES_DIR, default_name)
    dirname = os.path.dirname(path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    return path


def plot_dataset(ds: TimeSeriesDataset, path: Optional[str] = None, show_clean: bool = True) -> str:
    """Plots a dataset's train/val/test splits, each in a different color.

    Args:
        ds: dataset to plot.
        path: output PNG path. Defaults to "figures/<name>.png".
        show_clean: for noisy datasets, also overlay the noiseless
            ground truth as a thin dashed black line.

    Returns:
        the path the figure was saved to.
    """
    path = _resolve_path(path, f"{ds.name}.png")

    n_train, n_val = len(ds.train), len(ds.val)
    x_train = np.arange(n_train)
    x_val = np.arange(n_train, n_train + n_val)
    x_test = np.arange(n_train + n_val, n_train + n_val + len(ds.test))

    plt.figure(figsize=(10, 4))
    plt.plot(x_train, ds.train, color=SPLIT_COLORS["train"], label="train")
    plt.plot(x_val, ds.val, color=SPLIT_COLORS["val"], label="val")
    plt.plot(x_test, ds.test, color=SPLIT_COLORS["test"], label="test")
    if show_clean and is_noisy_name(ds.name):
        plt.plot(
            np.arange(len(ds.clean_values)), ds.clean_values,
            color="black", linestyle="--", linewidth=1, alpha=0.6, label="clean signal",
        )

    plt.title(ds.name)
    plt.xlabel("time step")
    plt.ylabel("value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path


def save_prediction_csv(
    prediction: np.ndarray,
    target: np.ndarray,
    path: str,
    target_clean: Optional[np.ndarray] = None,
) -> None:
    """Saves one forecast window (prediction vs. ground truth) to a CSV file.

    Args:
        prediction: model forecast, shape (horizon,).
        target: ground-truth target, shape (horizon,), same shape as `prediction`.
        path: output CSV path.
        target_clean: noiseless ground truth, shape (horizon,), optional.
    """
    dirname = os.path.dirname(path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

    fields = ["step", "prediction", "target"] + (["target_clean"] if target_clean is not None else [])
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for i in range(len(prediction)):
            row = {"step": i, "prediction": prediction[i], "target": target[i]}
            if target_clean is not None:
                row["target_clean"] = target_clean[i]
            writer.writerow(row)


def load_prediction_csv(path: str) -> dict[str, np.ndarray]:
    """
    Args:
        path: CSV file written by `save_prediction_csv`.

    Returns:
        dict with keys "prediction" and "target" (each shape (horizon,)),
        plus "target_clean" (shape (horizon,)) if present in the file.
    """
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    out = {
        "prediction": np.array([float(r["prediction"]) for r in rows]),
        "target": np.array([float(r["target"]) for r in rows]),
    }
    if rows and "target_clean" in rows[0]:
        out["target_clean"] = np.array([float(r["target_clean"]) for r in rows])
    return out


def predict_window(
    model: nn.Module,
    ds: TimeSeriesDataset,
    context: int,
    horizon: int,
    split: str = "test",
    window_index: int = 0,
) -> dict[str, np.ndarray]:
    """Runs `model` on one forecast window of `ds`, in the original data scale.

    Standardizes the chosen split using the train split's mean/std (matching
    the procedure used in `train.py`), predicts, then inverse-transforms
    the prediction and target back to the dataset's original scale.

    Args:
        model: a trained module mapping (batch, context) -> (batch, horizon).
        ds: dataset to draw the window from.
        context: context length C the model was trained with.
        horizon: prediction horizon p the model was trained with.
        split: which split to draw the window from ("train", "val", or "test").
        window_index: index of the sliding window within that split.

    Returns:
        dict with "prediction" and "target" (each shape (horizon,)), plus
        "target_clean" (shape (horizon,)) if `ds` is a noisy dataset.
    """
    mean, std = standardize_stats(ds.train)
    series = {"train": ds.train, "val": ds.val, "test": ds.test}[split]
    X, Y = make_windows((series - mean) / std, context, horizon)

    pred = model(X[window_index : window_index + 1])[0] * std + mean
    target = Y[window_index] * std + mean
    out = {"prediction": pred, "target": target}

    if is_noisy_name(ds.name):
        clean_series = {"train": ds.clean_train, "val": ds.clean_val, "test": ds.clean_test}[split]
        _, Y_clean = make_windows((clean_series - mean) / std, context, horizon)
        out["target_clean"] = Y_clean[window_index] * std + mean

    return out


def plot_prediction_arrays(
    prediction: np.ndarray,
    target: np.ndarray,
    target_clean: Optional[np.ndarray] = None,
    title: Optional[str] = None,
    path: Optional[str] = None,
) -> str:
    """Plots one forecast window: prediction vs. target (and clean target, if given).

    Args:
        prediction: shape (horizon,).
        target: shape (horizon,), same shape as `prediction`.
        target_clean: noiseless ground truth, shape (horizon,), optional.
        title: figure title.
        path: output PNG path. Defaults to "figures/predictions.png".

    Returns:
        the path the figure was saved to.
    """
    path = _resolve_path(path, "predictions.png")

    plt.figure(figsize=(8, 4))
    if target_clean is not None:
        plt.plot(target_clean, label="clean ground truth", linewidth=2)
        plt.plot(target, label="noisy target", alpha=0.5)
    else:
        plt.plot(target, label="ground truth", linewidth=2)
    plt.plot(prediction, label="prediction", linestyle="--")
    if title:
        plt.title(title)
    plt.xlabel("forecast step")
    plt.ylabel("value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path


def plot_predictions(
    model: Optional[nn.Module] = None,
    ds: Optional[TimeSeriesDataset] = None,
    context: Optional[int] = None,
    horizon: Optional[int] = None,
    csv_path: Optional[str] = None,
    split: str = "test",
    window_index: int = 0,
    title: Optional[str] = None,
    path: Optional[str] = None,
) -> str:
    """Plots a forecast window's prediction against its ground truth.

    Exactly one of `model` or `csv_path` must be given:
      - `model`: run inference live on `ds` (with `context`/`horizon`/`split`/
        `window_index` selecting the window) and plot the result.
      - `csv_path`: load a previously saved prediction (via
        `save_prediction_csv`) and plot it directly.

    Args:
        model: trained module mapping (batch, context) -> (batch, horizon).
        ds: dataset to draw the window from (required with `model`).
        context: context length C (required with `model`).
        horizon: prediction horizon p (required with `model`).
        csv_path: path to a CSV saved by `save_prediction_csv`.
        split: split to draw the window from, when using `model`.
        window_index: window index within that split, when using `model`.
        title: figure title. Defaults to a description of the source.
        path: output PNG path. Defaults to "figures/predictions_<...>.png".

    Returns:
        the path the figure was saved to.
    """
    if (model is None) == (csv_path is None):
        raise ValueError("pass exactly one of `model` or `csv_path`")

    if model is not None:
        if ds is None or context is None or horizon is None:
            raise ValueError("`ds`, `context`, and `horizon` are required when passing `model`")
        data = predict_window(model, ds, context, horizon, split=split, window_index=window_index)
        default_name = f"predictions_{ds.name}_ctx{context}_p{horizon}.png"
        title = title or f"{ds.name} ({split}) -- ctx={context}, p={horizon}"
    else:
        data = load_prediction_csv(csv_path)
        default_name = f"predictions_{os.path.splitext(os.path.basename(csv_path))[0]}.png"
        title = title or os.path.splitext(os.path.basename(csv_path))[0]

    path = _resolve_path(path, default_name)
    return plot_prediction_arrays(
        data["prediction"], data["target"], data.get("target_clean"), title=title, path=path
    )
