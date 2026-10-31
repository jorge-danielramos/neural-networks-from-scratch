"""Part 2: train the Part-1 MLPs on synthetic time-series forecasting.

For every (dataset, model, task) combination, this script:
  1. builds sliding-window (context -> horizon) supervised pairs,
  2. trains the model with mini-batch SGD,
  3. reports test_mse / test_mae, and, for noisy datasets, gt_test_mse /
     gt_test_mae evaluated against the noiseless ground truth,
  4. saves the results to a CSV file and plots the best/worst denoiser
     (across models and task settings) for each noisy dataset.
"""

import argparse
import csv
import os

import numpy as np
from tqdm import tqdm

import nn
from data import ALL_DATASET_NAMES, TimeSeriesDataset, is_noisy_name
from model import MLPClassifier_1, MLPClassifier_2
from optim import SGD
from utils import make_windows, plot_dataset, plot_prediction_arrays, save_prediction_csv, standardize_stats

TASK_SETTINGS = [(96, 336), (336, 336), (336, 720)]  # (context C, horizon p)

MODEL_SPECS = {
    "MLP_1": {"kind": "mlp1", "activation": None, "loss": nn.MSE},
    "MLP_2_ReLU": {"kind": "mlp2", "activation": nn.ReLU, "loss": nn.MSE},
    "MLP_2_Sigmoid_MSE": {"kind": "mlp2", "activation": nn.Sigmoid, "loss": nn.MSE},
    "MLP_2_Sigmoid_MAE": {"kind": "mlp2", "activation": nn.Sigmoid, "loss": nn.MAE},
    "MLP_2_Tanh": {"kind": "mlp2", "activation": nn.Tanh, "loss": nn.MSE},
}

HIDDEN_FEATURES = 64


def build_model(name: str, context: int, horizon: int) -> nn.Module:
    """
    Args:
        name: key into MODEL_SPECS.
        context: model input size, C.
        horizon: model output size, p.

    Returns:
        model: untrained module mapping (batch, context) -> (batch, horizon).
    """
    spec = MODEL_SPECS[name]
    if spec["kind"] == "mlp1":
        return MLPClassifier_1(in_features=context, out_features=horizon)
    return MLPClassifier_2(
        in_features=context,
        hidden_features=HIDDEN_FEATURES,
        out_features=horizon,
        activation=spec["activation"],
    )


def train_model(
    model: nn.Module,
    loss_fn: nn.Module,
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    desc: str = "",
) -> dict[str, list[float]]:
    """Trains `model` in place with mini-batch SGD.

    Args:
        model: model to train, mapping (batch, context) -> (batch, horizon).
        loss_fn: training loss module (MSE or MAE) used to compute gradients.
        X_train: shape (n_train, context).
        Y_train: shape (n_train, horizon).
        X_val: shape (n_val, context).
        Y_val: shape (n_val, horizon).
        epochs: number of passes over the training data.
        batch_size: mini-batch size.
        lr: SGD learning rate.
        seed: seed for the mini-batch shuffling order.
        desc: label shown on the per-epoch tqdm progress bar.

    Returns:
        history: dict with keys "train_loss" and "val_loss", each a list of
            `epochs` scalars (the loss_fn value at the end of that epoch).
    """
    rng = np.random.default_rng(seed)
    optimizer = SGD(model, lr=lr)
    n_train = len(X_train)
    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

    epoch_bar = tqdm(range(epochs), desc=desc, leave=False, unit="epoch")
    for _ in epoch_bar:
        order = rng.permutation(n_train)
        epoch_loss = 0.0
        for start in range(0, n_train, batch_size):
            idx = order[start : start + batch_size]
            xb, yb = X_train[idx], Y_train[idx]

            ##CODE
            # TODO: implement one mini-batch SGD training step:
            #   1. zero the optimizer's gradients
            optimizer.zero_grad()
            #   2. forward pass: pred = model(xb)
            pred = model(xb) #Y^
            #   3. compute the loss: loss = loss_fn(pred, yb)
            loss = loss_fn(pred, yb)
            #   4. backward pass: model.backward(loss_fn.backward())
            dPred = loss_fn.backward()
            model.backward(dPred)
            #   5. take an optimizer step
            optimizer.step()
            
            ##END

            epoch_loss += float(loss) * len(idx)

        train_loss = epoch_loss / n_train
        val_loss = float(loss_fn(model(X_val), Y_val))
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        epoch_bar.set_postfix(train_loss=f"{train_loss:.4g}", val_loss=f"{val_loss:.4g}")

    return history


def mse(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Args:
        pred: shape (n, horizon).
        target: shape (n, horizon), same shape as `pred`.

    Returns:
        scalar mean squared error.
    """
    return float(np.mean((pred - target) ** 2))


def mae(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Args:
        pred: shape (n, horizon).
        target: shape (n, horizon), same shape as `pred`.

    Returns:
        scalar mean absolute error.
    """
    return float(np.mean(np.abs(pred - target)))


def run_experiment(
    dataset_name: str,
    ds: TimeSeriesDataset,
    model_name: str,
    context: int,
    horizon: int,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
) -> dict:
    """Trains one model on one (dataset, context, horizon) combination and evaluates it.

    Args:
        dataset_name: name of the dataset (key into `MODEL_SPECS`-style registry).
        ds: `TimeSeriesDataset` instance providing train/val/test (and, for
            noisy datasets, clean_train/clean_val/clean_test) splits.
        model_name: key into MODEL_SPECS.
        context: context length C.
        horizon: prediction horizon p.
        epochs, batch_size, lr: training hyperparameters, see `train_model`.
        seed: seed for weight init / minibatch shuffling.

    Returns:
        result: dict with keys "dataset", "model", "context", "horizon",
            "test_mse", "test_mae", and, for noisy datasets,
            "gt_test_mse" / "gt_test_mae" (evaluated against the noiseless
            ground truth) plus the raw-scale arrays needed for plotting:
            "test_pred", "test_target", "test_target_clean" (the last only
            for noisy datasets).
    """
    np.random.seed(seed)  # controls Linear's Glorot weight initialization

    mean, std = standardize_stats(ds.train)
    train_n = (ds.train - mean) / std
    val_n = (ds.val - mean) / std
    test_n = (ds.test - mean) / std

    X_train, Y_train = make_windows(train_n, context, horizon)
    X_val, Y_val = make_windows(val_n, context, horizon)
    X_test, Y_test = make_windows(test_n, context, horizon)

    spec = MODEL_SPECS[model_name]
    model = build_model(model_name, context, horizon)
    loss_fn = spec["loss"]()
    desc = f"{dataset_name} | {model_name} | ctx={context}, p={horizon}"
    train_model(model, loss_fn, X_train, Y_train, X_val, Y_val, epochs, batch_size, lr, seed, desc=desc)

    pred = model(X_test) * std + mean
    target = Y_test * std + mean

    result = {
        "dataset": dataset_name,
        "model": model_name,
        "context": context,
        "horizon": horizon,
        "test_mse": mse(pred, target),
        "test_mae": mae(pred, target),
        "test_pred": pred,
        "test_target": target,
    }

    if is_noisy_name(dataset_name):
        clean_test_n = (ds.clean_test - mean) / std
        _, Y_test_clean = make_windows(clean_test_n, context, horizon)
        target_clean = Y_test_clean * std + mean
        result["gt_test_mse"] = mse(pred, target_clean)
        result["gt_test_mae"] = mae(pred, target_clean)
        result["test_target_clean"] = target_clean

    return result


def save_results_csv(results: list[dict], path: str) -> None:
    """
    Args:
        results: list of result dicts from `run_experiment`.
        path: output CSV file path.
    """
    fields = ["dataset", "model", "context", "horizon", "test_mse", "test_mae", "gt_test_mse", "gt_test_mae"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in results:
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=ALL_DATASET_NAMES, choices=ALL_DATASET_NAMES)
    parser.add_argument("--models", nargs="+", default=list(MODEL_SPECS), choices=list(MODEL_SPECS))
    parser.add_argument("--n-points", type=int, default=11000)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument(
        "--plot-datasets", action="store_true",
        help="Also save a raw train/val/test plot for every dataset (see utils.plot_dataset), into figures/.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    datasets = {
        name: TimeSeriesDataset(name=name, n_points=args.n_points, seed=args.seed + i)
        for i, name in enumerate(args.datasets)
    }

    if args.plot_datasets:
        for ds in datasets.values():
            plot_dataset(ds)

    runs = [
        (dataset_name, context, horizon, model_name)
        for dataset_name in datasets
        for context, horizon in TASK_SETTINGS
        for model_name in args.models
    ]

    results = []
    run_bar = tqdm(runs, desc="experiments", unit="run")
    for dataset_name, context, horizon, model_name in run_bar:
        run_bar.set_description(f"{dataset_name} | {model_name} | ctx={context}, p={horizon}")
        result = run_experiment(
            dataset_name, datasets[dataset_name], model_name, context, horizon,
            epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, seed=args.seed,
        )
        tqdm.write(
            f"[{dataset_name} | {model_name} | ctx={context}, p={horizon}] "
            f"test_mse={result['test_mse']:.4g} test_mae={result['test_mae']:.4g}"
        )
        results.append(result)

    save_results_csv(results, os.path.join(args.output_dir, "results.csv"))

    noisy_results = [r for r in results if "gt_test_mse" in r]
    if noisy_results:
        denoiser_dir = os.path.join(args.output_dir, "denoisers")

        by_dataset: dict[str, list[dict]] = {}
        for r in noisy_results:
            by_dataset.setdefault(r["dataset"], []).append(r)

        for dataset_name, group in by_dataset.items():
            best = min(group, key=lambda r: r["gt_test_mse"])
            worst = max(group, key=lambda r: r["gt_test_mse"])

            for tag, r in [("best", best), ("worst", worst)]:
                save_prediction_csv(
                    r["test_pred"][0], r["test_target"][0],
                    os.path.join(denoiser_dir, f"{dataset_name}_{tag}.csv"),
                    target_clean=r["test_target_clean"][0],
                )
                plot_prediction_arrays(
                    r["test_pred"][0], r["test_target"][0], r["test_target_clean"][0],
                    title=f"{tag.capitalize()} denoiser on {dataset_name}: {r['model']} "
                    f"(ctx={r['context']}, p={r['horizon']}, gt_test_mse={r['gt_test_mse']:.4g})",
                    path=os.path.join(denoiser_dir, f"{dataset_name}_{tag}.png"),
                )

            print(
                f"{dataset_name}: best={best['model']} (ctx={best['context']}, p={best['horizon']}) | "
                f"worst={worst['model']} (ctx={worst['context']}, p={worst['horizon']})"
            )

    print(f"Saved results to {args.output_dir}/")


if __name__ == "__main__":
    main()
