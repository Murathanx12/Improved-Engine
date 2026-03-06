"""
Temporal Sequence Models — LSTM + TCN Ensemble
===============================================

WHY THIS EXISTS:
    LightGBM and XGBoost treat every prediction date independently. They
    cannot learn temporal patterns: VIX rising for 6 consecutive weeks,
    yield curve staying inverted for 8 months, credit spreads widening
    over a deterioration cycle. These patterns precede nearly every
    major crash in history.

ARCHITECTURE:
    Two complementary temporal models trained on 60-day feature windows:

    1. LSTM (Long Short-Term Memory): Learns variable-length dependencies.
       The forget gate naturally captures "how long has stress been building?"
       Two-layer architecture with dropout for regularization.

    2. TCN (Temporal Convolutional Network): Dilated causal convolutions
       capture multi-scale patterns (5-day, 10-day, 20-day, 40-day) in
       parallel. Faster training, stable gradients, no vanishing gradient
       problem. Often matches or exceeds LSTM on financial time series.

    Both share the same input (60-day normalized feature windows) and
    output (multi-horizon crash probabilities). TemporalEnsemble averages
    their predictions for the final temporal signal.

DATA LEAKAGE PREVENTION:
    - Feature windows use ONLY backward-looking data (features.iloc[idx-60:idx])
    - Normalization statistics computed on training set only
    - Purged train/val split with gap covering forward-looking horizon
    - No future targets leak into input construction
"""

import numpy as np
import pandas as pd
from typing import Optional

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


# ═══════════════════════════════════════════════════════════════════════
# SEQUENCE DATASET
# ═══════════════════════════════════════════════════════════════════════

if _HAS_TORCH:

    class CrashSequenceDataset(Dataset):
        """Converts a feature matrix into sliding-window sequences for temporal models.

        Each sample is a (window_size, n_features) tensor paired with a
        multi-horizon binary target vector.

        Args:
            features: Normalized feature matrix (n_days, n_features).
            targets: Dict of {horizon: binary_series} aligned with features index.
            window_size: Number of trading days per sequence (default 60 = ~3 months).
            sample_weights: Optional per-sample weights for temporal emphasis.
        """

        def __init__(
            self,
            features: np.ndarray,
            targets: dict,
            window_size: int = 60,
            sample_weights: np.ndarray = None,
        ):
            self.window_size = window_size
            self.horizons = sorted(targets.keys())

            # Build aligned arrays
            # Each sequence at index i uses features[i : i + window_size]
            # and predicts targets at index i + window_size - 1 (end of window)
            n_samples = len(features) - window_size
            if n_samples <= 0:
                raise ValueError(f"Not enough data: {len(features)} rows < window_size {window_size}")

            self.X = np.array(features, dtype=np.float32)
            self.y = {}
            for h in self.horizons:
                t = np.array(targets[h], dtype=np.float32)
                self.y[h] = t[window_size:]  # align with end of each window

            self.n_samples = min(n_samples, len(list(self.y.values())[0]))
            self.weights = sample_weights[window_size:self.n_samples + window_size] \
                if sample_weights is not None else np.ones(self.n_samples, dtype=np.float32)

        def __len__(self):
            return self.n_samples

        def __getitem__(self, idx):
            x = torch.from_numpy(self.X[idx: idx + self.window_size])
            targets = torch.tensor(
                [self.y[h][idx] for h in self.horizons], dtype=torch.float32
            )
            weight = torch.tensor(self.weights[idx], dtype=torch.float32)
            return x, targets, weight


    # ═══════════════════════════════════════════════════════════════════
    # LSTM MODEL
    # ═══════════════════════════════════════════════════════════════════

    class LSTMCrashModel(nn.Module):
        """2-layer LSTM for temporal crash pattern recognition.

        Architecture:
            Input (batch, seq_len=60, n_features)
            → LayerNorm
            → LSTM (2 layers, hidden=128, dropout=0.3)
            → Take last hidden state
            → Dropout → Linear(128, 64) → ReLU → Dropout
            → Linear(64, n_horizons) → Sigmoid

        The model learns sequential patterns like:
        - Volatility rising for consecutive weeks
        - Credit spread widening acceleration
        - VIX double-top formations
        - Duration of yield curve inversion
        """

        def __init__(
            self,
            input_dim: int,
            hidden_dim: int = 128,
            num_layers: int = 2,
            n_horizons: int = 3,
            dropout: float = 0.3,
        ):
            super().__init__()
            self.input_norm = nn.LayerNorm(input_dim)
            self.lstm = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 64),
                nn.ReLU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(64, n_horizons),
                nn.Sigmoid(),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass.

            Args:
                x: (batch, seq_len, n_features)

            Returns:
                (batch, n_horizons) crash probabilities in [0, 1]
            """
            x = self.input_norm(x)
            lstm_out, (h_n, _) = self.lstm(x)
            last_hidden = h_n[-1]  # Last layer's hidden state: (batch, hidden_dim)
            return self.head(last_hidden)


    # ═══════════════════════════════════════════════════════════════════
    # TCN MODEL
    # ═══════════════════════════════════════════════════════════════════

    class _CausalConv1d(nn.Module):
        """Causal convolution: output at time t depends only on inputs at time <= t."""

        def __init__(self, in_channels, out_channels, kernel_size, dilation):
            super().__init__()
            self.padding = (kernel_size - 1) * dilation
            self.conv = nn.Conv1d(
                in_channels, out_channels, kernel_size,
                padding=self.padding, dilation=dilation,
            )

        def forward(self, x):
            out = self.conv(x)
            if self.padding > 0:
                out = out[:, :, :-self.padding]  # Remove future-looking padding
            return out

    class _TCNBlock(nn.Module):
        """Residual TCN block with dilated causal convolution."""

        def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
            super().__init__()
            self.net = nn.Sequential(
                _CausalConv1d(in_channels, out_channels, kernel_size, dilation),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Dropout(dropout),
                _CausalConv1d(out_channels, out_channels, kernel_size, dilation),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.residual = nn.Conv1d(in_channels, out_channels, 1) \
                if in_channels != out_channels else nn.Identity()

        def forward(self, x):
            return self.net(x) + self.residual(x)

    class TCNCrashModel(nn.Module):
        """Temporal Convolutional Network for multi-scale pattern recognition.

        Architecture:
            Input (batch, seq_len=60, n_features)
            → Transpose to (batch, n_features, seq_len)
            → 4 TCN blocks with exponentially increasing dilation [1, 2, 4, 8]
            → Global average pooling over time
            → Linear(64, n_horizons) → Sigmoid

        The dilated convolutions capture patterns at multiple time scales:
        - Dilation 1: 3-day patterns (immediate shock)
        - Dilation 2: 5-day patterns (weekly)
        - Dilation 4: 9-day patterns (bi-weekly)
        - Dilation 8: 17-day patterns (monthly)
        Combined receptive field covers the entire 60-day window.
        """

        def __init__(
            self,
            input_dim: int,
            num_channels: list = None,
            kernel_size: int = 3,
            n_horizons: int = 3,
            dropout: float = 0.2,
        ):
            super().__init__()
            if num_channels is None:
                num_channels = [64, 64, 64, 64]

            layers = []
            in_ch = input_dim
            for i, out_ch in enumerate(num_channels):
                dilation = 2 ** i
                layers.append(_TCNBlock(in_ch, out_ch, kernel_size, dilation, dropout))
                in_ch = out_ch

            self.tcn = nn.Sequential(*layers)
            self.head = nn.Sequential(
                nn.Linear(num_channels[-1], n_horizons),
                nn.Sigmoid(),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass.

            Args:
                x: (batch, seq_len, n_features)

            Returns:
                (batch, n_horizons) crash probabilities in [0, 1]
            """
            # TCN expects (batch, channels, seq_len)
            x = x.transpose(1, 2)
            tcn_out = self.tcn(x)
            # Global average pooling over the temporal dimension
            pooled = tcn_out.mean(dim=2)
            return self.head(pooled)


    # ═══════════════════════════════════════════════════════════════════
    # TEMPORAL ENSEMBLE
    # ═══════════════════════════════════════════════════════════════════

    class TemporalEnsemble:
        """Trains and manages LSTM + TCN temporal models.

        Provides a single predict_proba interface that averages predictions
        from both models for robust temporal pattern recognition.
        """

        WINDOW_SIZE = 60  # 3 months of trading days
        HORIZONS = ["3m", "6m", "12m"]

        def __init__(self, hidden_dim: int = 128, random_state: int = 42):
            self.hidden_dim = hidden_dim
            self.random_state = random_state
            self.lstm_model = None
            self.tcn_model = None
            self.is_trained = False
            self.feature_names = None
            self._train_mean = None
            self._train_std = None
            self._device = torch.device("cpu")

        def train(
            self,
            features: pd.DataFrame,
            targets: dict,
            train_end_idx: Optional[int] = None,
            min_sequences: int = 400,
            n_epochs: int = 100,
            batch_size: int = 64,
            lr: float = 1e-3,
            patience: int = 15,
        ) -> dict:
            """Train LSTM and TCN on temporal sequences.

            Args:
                features: Feature matrix (daily, backward-looking).
                targets: Dict of {horizon: binary_series}.
                train_end_idx: Temporal cutoff for training data.
                min_sequences: Minimum sequences needed (default 400).
                n_epochs: Maximum training epochs.
                batch_size: Mini-batch size.
                lr: Initial learning rate.
                patience: Early stopping patience.
            """
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)

            if train_end_idx is not None:
                X = features.iloc[:train_end_idx]
                tgt = {h: t.iloc[:train_end_idx] for h, t in targets.items()}
            else:
                X = features
                tgt = targets

            self.feature_names = list(X.columns)
            n_features = len(self.feature_names)

            # Filter to available horizons
            avail_horizons = [h for h in self.HORIZONS if h in tgt]
            if not avail_horizons:
                return {"success": False, "reason": "No horizon targets available"}

            # Check minimum data
            n_available = len(X) - self.WINDOW_SIZE
            if n_available < min_sequences:
                return {"success": False, "reason": f"Only {n_available} sequences < {min_sequences}"}

            # Normalize features using training data statistics
            self._train_mean = X.mean().values.astype(np.float32)
            self._train_std = X.std().values.astype(np.float32)
            self._train_std = np.clip(self._train_std, 1e-8, None)
            X_norm = (X.values - self._train_mean) / self._train_std

            # Replace any remaining NaN/inf with 0
            X_norm = np.nan_to_num(X_norm, nan=0.0, posinf=0.0, neginf=0.0)

            # Build target dict with aligned arrays
            target_arrays = {}
            for h in avail_horizons:
                t = tgt[h].values.astype(np.float32)
                # Fill NaN targets with 0 (no crash)
                t = np.nan_to_num(t, nan=0.0)
                target_arrays[h] = t

            # Temporal weighting (recent data weighted higher)
            temporal_weights = np.linspace(0.5, 1.5, len(X_norm)).astype(np.float32)

            # Purged train/val split
            gap_days = 265  # Worst-case: 12m horizon
            val_size = max(252, n_available // 5)
            split_point = len(X_norm) - val_size - gap_days - self.WINDOW_SIZE

            if split_point < self.WINDOW_SIZE + min_sequences // 2:
                split_point = len(X_norm) - val_size - self.WINDOW_SIZE
                gap_days = 0

            # Build datasets
            train_X = X_norm[:split_point + self.WINDOW_SIZE]
            train_targets = {h: t[:split_point + self.WINDOW_SIZE] for h, t in target_arrays.items()}
            train_weights = temporal_weights[:split_point + self.WINDOW_SIZE]

            val_start = split_point + self.WINDOW_SIZE + gap_days
            val_X = X_norm[val_start - self.WINDOW_SIZE:]
            val_targets = {h: t[val_start - self.WINDOW_SIZE:] for h, t in target_arrays.items()}

            try:
                train_ds = CrashSequenceDataset(
                    train_X, train_targets, self.WINDOW_SIZE, train_weights,
                )
                val_ds = CrashSequenceDataset(
                    val_X, val_targets, self.WINDOW_SIZE,
                )
            except ValueError as e:
                return {"success": False, "reason": str(e)}

            if len(train_ds) < min_sequences // 2 or len(val_ds) < 50:
                return {"success": False, "reason": f"Insufficient sequences: train={len(train_ds)}, val={len(val_ds)}"}

            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

            n_horizons = len(avail_horizons)
            self._horizons = avail_horizons

            # Train LSTM
            print(f"  [LSTM] Training on {len(train_ds)} sequences, {n_features} features...")
            self.lstm_model = LSTMCrashModel(
                input_dim=n_features,
                hidden_dim=self.hidden_dim,
                n_horizons=n_horizons,
            ).to(self._device)
            lstm_metrics = self._train_model(
                self.lstm_model, train_loader, val_loader,
                n_epochs, lr, patience, "LSTM",
            )

            # Train TCN
            print(f"  [TCN] Training on {len(train_ds)} sequences, {n_features} features...")
            self.tcn_model = TCNCrashModel(
                input_dim=n_features,
                n_horizons=n_horizons,
            ).to(self._device)
            tcn_metrics = self._train_model(
                self.tcn_model, train_loader, val_loader,
                n_epochs, lr, patience, "TCN",
            )

            self.is_trained = True
            return {
                "success": True,
                "n_train_sequences": len(train_ds),
                "n_val_sequences": len(val_ds),
                "n_features": n_features,
                "lstm_val_loss": lstm_metrics["best_val_loss"],
                "tcn_val_loss": tcn_metrics["best_val_loss"],
                "lstm_epochs": lstm_metrics["epochs_trained"],
                "tcn_epochs": tcn_metrics["epochs_trained"],
            }

        def _train_model(
            self,
            model: nn.Module,
            train_loader: DataLoader,
            val_loader: DataLoader,
            n_epochs: int,
            lr: float,
            patience: int,
            model_name: str,
        ) -> dict:
            """Train a single PyTorch model with early stopping."""
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=patience // 3,
            )
            criterion = nn.BCELoss(reduction='none')

            best_val_loss = float('inf')
            best_state = None
            no_improve = 0

            for epoch in range(n_epochs):
                # Training
                model.train()
                train_loss = 0.0
                n_train = 0
                for x, y, w in train_loader:
                    x, y, w = x.to(self._device), y.to(self._device), w.to(self._device)
                    optimizer.zero_grad()
                    pred = model(x)
                    loss_per_sample = criterion(pred, y).mean(dim=1)  # Average over horizons
                    loss = (loss_per_sample * w).mean()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    train_loss += loss.item() * len(x)
                    n_train += len(x)

                # Validation
                model.eval()
                val_loss = 0.0
                n_val = 0
                with torch.no_grad():
                    for x, y, w in val_loader:
                        x, y = x.to(self._device), y.to(self._device)
                        pred = model(x)
                        loss = criterion(pred, y).mean()
                        val_loss += loss.item() * len(x)
                        n_val += len(x)

                avg_train = train_loss / max(n_train, 1)
                avg_val = val_loss / max(n_val, 1)
                scheduler.step(avg_val)

                if avg_val < best_val_loss:
                    best_val_loss = avg_val
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    no_improve = 0
                else:
                    no_improve += 1

                if no_improve >= patience:
                    break

            # Restore best model
            if best_state is not None:
                model.load_state_dict(best_state)

            epochs_trained = epoch + 1
            print(f"  [{model_name}] Trained {epochs_trained} epochs, "
                  f"best val loss: {best_val_loss:.4f}")

            return {"best_val_loss": best_val_loss, "epochs_trained": epochs_trained}

        def predict_proba(self, features: pd.DataFrame, horizon: str = "12m") -> np.ndarray:
            """Predict crash probability by averaging LSTM and TCN outputs.

            Args:
                features: Feature matrix. Uses last WINDOW_SIZE rows as the
                    input sequence. If more rows provided, builds sequences
                    from the last portion.

            Returns:
                Array of crash probabilities.
            """
            if not self.is_trained:
                raise RuntimeError("TemporalEnsemble not trained — call train() first")

            if horizon not in self._horizons:
                horizon = self._horizons[-1]  # Default to longest horizon
            h_idx = self._horizons.index(horizon)

            # Normalize using training statistics
            X = features[self.feature_names].values if isinstance(features, pd.DataFrame) else features
            X_norm = (X.astype(np.float32) - self._train_mean) / self._train_std
            X_norm = np.nan_to_num(X_norm, nan=0.0, posinf=0.0, neginf=0.0)

            # Build sequences
            if len(X_norm) < self.WINDOW_SIZE:
                # Pad with zeros at the beginning
                pad = np.zeros((self.WINDOW_SIZE - len(X_norm), X_norm.shape[1]), dtype=np.float32)
                X_norm = np.vstack([pad, X_norm])

            # Create sequences: one per row that has enough history
            n_sequences = len(X_norm) - self.WINDOW_SIZE + 1
            sequences = np.stack([
                X_norm[i: i + self.WINDOW_SIZE] for i in range(n_sequences)
            ])
            x_tensor = torch.from_numpy(sequences).to(self._device)

            # Predict with both models
            preds = []
            for model in [self.lstm_model, self.tcn_model]:
                if model is not None:
                    model.eval()
                    with torch.no_grad():
                        out = model(x_tensor)
                        preds.append(out[:, h_idx].cpu().numpy())

            if not preds:
                return np.full(n_sequences, 0.12)

            # Average ensemble predictions
            ensemble = np.mean(preds, axis=0)
            return np.clip(ensemble, 0.02, 0.98)

        def predict_individual(self, features: pd.DataFrame, horizon: str = "12m") -> dict:
            """Predict crash probability from each model individually.

            Returns:
                Dict with keys "lstm" and "tcn", each containing an np.ndarray.
                Useful for the meta-stacker which needs independent model signals.
            """
            if not self.is_trained:
                raise RuntimeError("TemporalEnsemble not trained")

            if horizon not in self._horizons:
                horizon = self._horizons[-1]
            h_idx = self._horizons.index(horizon)

            X = features[self.feature_names].values if isinstance(features, pd.DataFrame) else features
            X_norm = (X.astype(np.float32) - self._train_mean) / self._train_std
            X_norm = np.nan_to_num(X_norm, nan=0.0, posinf=0.0, neginf=0.0)

            if len(X_norm) < self.WINDOW_SIZE:
                pad = np.zeros((self.WINDOW_SIZE - len(X_norm), X_norm.shape[1]), dtype=np.float32)
                X_norm = np.vstack([pad, X_norm])

            n_sequences = len(X_norm) - self.WINDOW_SIZE + 1
            sequences = np.stack([
                X_norm[i: i + self.WINDOW_SIZE] for i in range(n_sequences)
            ])
            x_tensor = torch.from_numpy(sequences).to(self._device)

            result = {}
            for name, model in [("lstm", self.lstm_model), ("tcn", self.tcn_model)]:
                if model is not None:
                    model.eval()
                    with torch.no_grad():
                        out = model(x_tensor)
                        result[name] = np.clip(out[:, h_idx].cpu().numpy(), 0.02, 0.98)
                else:
                    result[name] = np.full(n_sequences, 0.12)

            return result

        def predict_all_horizons(self, features: pd.DataFrame) -> dict:
            """Predict crash probability at all trained horizons."""
            if not self.is_trained:
                return {}
            results = {}
            for horizon in self._horizons:
                results[horizon] = self.predict_proba(features, horizon)
            return results

else:
    # Fallback when PyTorch is not installed
    class CrashSequenceDataset:
        pass

    class LSTMCrashModel:
        pass

    class TCNCrashModel:
        pass

    class TemporalEnsemble:
        """Fallback when PyTorch is not installed."""

        WINDOW_SIZE = 60
        HORIZONS = ["3m", "6m", "12m"]

        def __init__(self, hidden_dim: int = 128, random_state: int = 42):
            self.is_trained = False
            self._horizons = []

        def train(self, features, targets, train_end_idx=None, min_sequences=400, **kwargs):
            return {"success": False, "reason": "PyTorch not installed"}

        def predict_proba(self, features, horizon: str = "12m"):
            n = len(features) if hasattr(features, '__len__') else 1
            return np.full(n, 0.12)

        def predict_all_horizons(self, features):
            return {}


__all__ = [
    "CrashSequenceDataset", "LSTMCrashModel", "TCNCrashModel",
    "TemporalEnsemble",
]
