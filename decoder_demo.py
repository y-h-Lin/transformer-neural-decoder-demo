"""Runnable causal Transformer demo for neural-to-kinematic decoding.

The script works without research data by generating synthetic, Poisson-like
neural activity. Real data can be supplied as an NPZ file containing:

    neural:     [trials, time_bins, channels]
    kinematics: [trials, time_bins, outputs]

The outputs can be, for example, [x, y, vx, vy] or
[x, y, z, vx, vy, vz].
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class Config:
    seed: int = 7
    trials: int = 120
    sequence_length: int = 48
    channels: int = 32
    output_dim: int = 4
    pool_size: int = 1
    batch_size: int = 16
    epochs: int = 10
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    d_model: int = 48
    nhead: int = 4
    layers: int = 2
    feedforward_dim: int = 96
    dropout: float = 0.15


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def generate_synthetic_data(
    trials: int,
    sequence_length: int,
    channels: int,
    output_dim: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate motor trajectories and tuned Poisson neural activity.

    This is only a pipeline smoke-test dataset. It intentionally creates a
    learnable relationship between recent neural activity and kinematics; it
    is not intended to reproduce biological recordings.
    """
    if output_dim not in (4, 6):
        raise ValueError("Synthetic mode supports output_dim=4 or output_dim=6")

    rng = np.random.default_rng(seed)
    dimensions = output_dim // 2
    t = np.linspace(0.0, 1.0, sequence_length, dtype=np.float32)
    kinematics = np.empty(
        (trials, sequence_length, output_dim), dtype=np.float32
    )

    for trial in range(trials):
        positions = []
        for axis in range(dimensions):
            frequency = rng.uniform(0.6, 1.8)
            phase = rng.uniform(0.0, 2.0 * np.pi)
            amplitude = rng.uniform(0.6, 1.4)
            harmonic = 0.25 * np.sin(
                2.0 * np.pi * (frequency * 1.7) * t + phase / 2.0
            )
            position = amplitude * np.sin(
                2.0 * np.pi * frequency * t + phase
            ) + harmonic
            positions.append(position)
        position_array = np.stack(positions, axis=-1)
        velocity_array = np.gradient(position_array, axis=0)
        kinematics[trial] = np.concatenate(
            [position_array, velocity_array], axis=-1
        )

    flat_targets = kinematics.reshape(-1, output_dim)
    target_scale = flat_targets.std(axis=0, keepdims=True) + 1e-6
    standardized_targets = flat_targets / target_scale

    tuning = rng.normal(0.0, 0.45, size=(output_dim, channels))
    baseline = rng.uniform(0.5, 1.5, size=(1, channels))
    log_rate = standardized_targets @ tuning + np.log(baseline)
    rate = np.clip(np.exp(log_rate), 0.05, 12.0)
    neural = rng.poisson(rate).astype(np.float32)
    neural = neural.reshape(trials, sequence_length, channels)

    # A short causal smoothing kernel approximates binned population activity.
    smoothed = neural.copy()
    smoothed[:, 1:] += 0.35 * neural[:, :-1]
    smoothed[:, 2:] += 0.15 * neural[:, :-2]
    return smoothed.astype(np.float32), kinematics


def load_npz(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    if "neural" not in data or "kinematics" not in data:
        raise KeyError("NPZ must contain 'neural' and 'kinematics' arrays")
    neural = np.asarray(data["neural"], dtype=np.float32)
    kinematics = np.asarray(data["kinematics"], dtype=np.float32)
    if neural.ndim != 3 or kinematics.ndim != 3:
        raise ValueError("Both arrays must have shape [trials, time, features]")
    if neural.shape[:2] != kinematics.shape[:2]:
        raise ValueError("Neural and kinematic trial/time dimensions must match")
    return neural, kinematics


def pool_channels(neural: np.ndarray, pool_size: int) -> np.ndarray:
    """Sum fixed groups of feature channels as a simple pooling proxy."""
    if pool_size <= 1:
        return neural
    channels = neural.shape[-1]
    pooled_channels = math.ceil(channels / pool_size)
    padded_channels = pooled_channels * pool_size
    if padded_channels != channels:
        neural = np.pad(
            neural,
            ((0, 0), (0, 0), (0, padded_channels - channels)),
            mode="constant",
        )
    return neural.reshape(*neural.shape[:-1], pooled_channels, pool_size).sum(-1)


class SinusoidalPositionEncoding(nn.Module):
    def __init__(self, d_model: int, max_length: int = 4096) -> None:
        super().__init__()
        position = torch.arange(max_length).unsqueeze(1)
        divisor = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        encoding = torch.zeros(max_length, d_model)
        encoding[:, 0::2] = torch.sin(position * divisor)
        encoding[:, 1::2] = torch.cos(position * divisor[: encoding[:, 1::2].shape[1]])
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[1] > self.encoding.shape[1]:
            raise ValueError("Sequence exceeds positional encoding max_length")
        return x + self.encoding[:, : x.shape[1]].to(dtype=x.dtype)


class CausalTransformerKinematicDecoder(nn.Module):
    """Online-compatible neural decoder based on causal self-attention."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        d_model: int,
        nhead: int,
        layers: int,
        feedforward_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
        )
        self.position = SinusoidalPositionEncoding(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_model = nn.TransformerEncoder(
            layer,
            num_layers=layers,
            norm=nn.LayerNorm(d_model),
            enable_nested_tensor=False,
        )
        self.output_head = nn.Linear(d_model, output_dim)

    def forward(self, neural: Tensor) -> Tensor:
        hidden = self.position(self.input_projection(neural))
        length = neural.shape[1]
        causal_mask = torch.triu(
            torch.ones(length, length, dtype=torch.bool, device=neural.device),
            diagonal=1,
        )
        hidden = self.temporal_model(hidden, mask=causal_mask)
        return self.output_head(hidden)


class GRUKinematicDecoder(nn.Module):
    """RNN baseline with a parameter scale similar to the small Transformer."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, layers: int) -> None:
        super().__init__()
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.temporal_model = nn.GRU(
            hidden_dim,
            hidden_dim,
            num_layers=layers,
            batch_first=True,
            dropout=0.1 if layers > 1 else 0.0,
        )
        self.output_head = nn.Linear(hidden_dim, output_dim)

    def forward(self, neural: Tensor) -> Tensor:
        hidden, _ = self.temporal_model(self.input_projection(neural))
        return self.output_head(hidden)


def split_and_standardize(
    neural: np.ndarray, kinematics: np.ndarray, seed: int
) -> Tuple[Dict[str, TensorDataset], Dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(neural))
    train_end = max(1, int(0.70 * len(indices)))
    validation_end = max(train_end + 1, int(0.85 * len(indices)))
    validation_end = min(validation_end, len(indices) - 1)
    split_indices = {
        "train": indices[:train_end],
        "validation": indices[train_end:validation_end],
        "test": indices[validation_end:],
    }

    train_neural = neural[split_indices["train"]]
    train_kinematics = kinematics[split_indices["train"]]
    x_mean = train_neural.mean(axis=(0, 1), keepdims=True)
    x_std = train_neural.std(axis=(0, 1), keepdims=True) + 1e-6
    y_mean = train_kinematics.mean(axis=(0, 1), keepdims=True)
    y_std = train_kinematics.std(axis=(0, 1), keepdims=True) + 1e-6

    datasets: Dict[str, TensorDataset] = {}
    for name, subset in split_indices.items():
        x = (neural[subset] - x_mean) / x_std
        y = (kinematics[subset] - y_mean) / y_std
        datasets[name] = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))

    stats = {
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
    }
    return datasets, stats


def make_loaders(
    datasets: Dict[str, TensorDataset], batch_size: int
) -> Dict[str, DataLoader]:
    return {
        "train": DataLoader(datasets["train"], batch_size=batch_size, shuffle=True),
        "validation": DataLoader(
            datasets["validation"], batch_size=batch_size, shuffle=False
        ),
        "test": DataLoader(datasets["test"], batch_size=batch_size, shuffle=False),
    }


@torch.no_grad()
def standardized_mse(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    squared_error = 0.0
    values = 0
    for neural, target in loader:
        neural, target = neural.to(device), target.to(device)
        prediction = model(neural)
        squared_error += nn.functional.mse_loss(
            prediction, target, reduction="sum"
        ).item()
        values += target.numel()
    return squared_error / max(values, 1)


def train_model(
    model: nn.Module,
    loaders: Dict[str, DataLoader],
    config: Config,
    device: torch.device,
) -> nn.Module:
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    best_state = None
    best_validation = float("inf")

    for epoch in range(1, config.epochs + 1):
        model.train()
        for neural, target in loaders["train"]:
            neural, target = neural.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.mse_loss(model(neural), target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        validation_loss = standardized_mse(model, loaders["validation"], device)
        print(f"  epoch {epoch:02d}/{config.epochs}: validation MSE={validation_loss:.4f}")
        if validation_loss < best_validation:
            best_validation = validation_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    stats: Dict[str, np.ndarray],
    device: torch.device,
) -> Dict[str, object]:
    model.eval()
    predictions = []
    targets = []
    for neural, target in loader:
        predictions.append(model(neural.to(device)).cpu().numpy())
        targets.append(target.numpy())

    predicted_standardized = np.concatenate(predictions, axis=0)
    target_standardized = np.concatenate(targets, axis=0)
    y_mean = stats["y_mean"]
    y_std = stats["y_std"]
    predicted = predicted_standardized * y_std + y_mean
    target = target_standardized * y_std + y_mean

    errors = predicted - target
    rmse = np.sqrt(np.mean(errors**2, axis=(0, 1)))
    target_mean = np.mean(target, axis=(0, 1), keepdims=True)
    residual_sum = np.sum(errors**2, axis=(0, 1))
    total_sum = np.sum((target - target_mean) ** 2, axis=(0, 1))
    r2 = 1.0 - residual_sum / np.maximum(total_sum, 1e-12)
    return {
        "rmse_per_output": rmse.tolist(),
        "r2_per_output": r2.tolist(),
        "mean_rmse": float(np.mean(rmse)),
        "mean_r2": float(np.mean(r2)),
    }


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def build_models(
    model_choice: str,
    input_dim: int,
    output_dim: int,
    config: Config,
) -> Iterable[Tuple[str, nn.Module]]:
    if model_choice in ("transformer", "both"):
        yield "transformer", CausalTransformerKinematicDecoder(
            input_dim=input_dim,
            output_dim=output_dim,
            d_model=config.d_model,
            nhead=config.nhead,
            layers=config.layers,
            feedforward_dim=config.feedforward_dim,
            dropout=config.dropout,
        )
    if model_choice in ("gru", "both"):
        yield "gru", GRUKinematicDecoder(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=config.d_model,
            layers=config.layers,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, help="NPZ containing neural and kinematics")
    parser.add_argument("--model", choices=("transformer", "gru", "both"), default="both")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--pool-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--quick", action="store_true", help="Small CPU smoke test")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = Config(seed=args.seed, epochs=args.epochs, pool_size=args.pool_size)
    if args.quick:
        config.trials = 48
        config.sequence_length = 32
        config.channels = 24
        config.batch_size = 12
        config.epochs = min(config.epochs, 2)
        config.d_model = 32
        config.feedforward_dim = 64
        config.layers = 1

    seed_everything(config.seed)
    if args.data:
        neural, kinematics = load_npz(args.data)
        data_source = str(args.data)
    else:
        neural, kinematics = generate_synthetic_data(
            trials=config.trials,
            sequence_length=config.sequence_length,
            channels=config.channels,
            output_dim=config.output_dim,
            seed=config.seed,
        )
        data_source = "synthetic"

    neural = pool_channels(neural, config.pool_size)
    datasets, stats = split_and_standardize(neural, kinematics, config.seed)
    loaders = make_loaders(datasets, config.batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Data source: {data_source}")
    print(f"Neural shape: {tuple(neural.shape)}")
    print(f"Kinematic shape: {tuple(kinematics.shape)}")
    print(f"Device: {device}")

    results: Dict[str, object] = {
        "data_source": data_source,
        "neural_shape": list(neural.shape),
        "kinematic_shape": list(kinematics.shape),
        "config": asdict(config),
        "models": {},
    }

    for model_name, model in build_models(
        args.model, neural.shape[-1], kinematics.shape[-1], config
    ):
        print(f"\nTraining {model_name} ({parameter_count(model):,} parameters)")
        trained = train_model(model, loaders, config, device)
        metrics = evaluate_model(trained, loaders["test"], stats, device)
        metrics["trainable_parameters"] = parameter_count(trained)
        results["models"][model_name] = metrics
        torch.save(
            {
                "model_state_dict": trained.state_dict(),
                "config": asdict(config),
                "input_dim": neural.shape[-1],
                "output_dim": kinematics.shape[-1],
                "normalization": stats,
            },
            args.output_dir / f"{model_name}_decoder.pt",
        )
        print(
            f"  test mean RMSE={metrics['mean_rmse']:.4f}, "
            f"mean R2={metrics['mean_r2']:.4f}"
        )

    with (args.output_dir / "results.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    print(f"\nSaved results to {args.output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
