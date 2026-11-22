"""PyTorch LSTM Autoencoder for Sequential API Anomaly Detection.

Inspects multi-step Open Banking call sequences (e.g. /consent -> /accounts -> /transfer)
and flags sequence skips, rate disruptions, and structural deviations via reconstruction error.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import json
import random
import numpy as np
import structlog
import torch
import torch.nn as nn
from data.attack_traffic import generate_all_attacks_dataset
from data.normal_traffic import generate_normal_session
from ml.config import ml_settings
from ml.feature_engineering import extract_features_from_request, feature_dict_to_vector
from ml.sliding_window import tracker

logger = structlog.get_logger(__name__)


class LSTMAutoencoder(nn.Module):
    """LSTM Autoencoder for sequential feature reconstruction."""

    def __init__(
        self,
        input_dim: int = 12,
        hidden_dim: int = 16,
        latent_dim: int = 8,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # Encoder: compress sequence to latent representation
        self.encoder_lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.encoder_fc = nn.Linear(hidden_dim, latent_dim)

        # Decoder: reconstruct full sequence from latent state
        self.decoder_fc = nn.Linear(latent_dim, hidden_dim)
        self.decoder_lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.output_fc = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass reconstructing input sequence.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim).

        Returns:
            Reconstructed tensor of identical shape.
        """
        _, seq_len, _ = x.size()
        _, (h_n, _) = self.encoder_lstm(x)
        latent = self.encoder_fc(h_n.squeeze(0))

        decoder_in = self.decoder_fc(latent).unsqueeze(1).repeat(1, seq_len, 1)
        decoder_out, _ = self.decoder_lstm(decoder_in)
        return self.output_fc(decoder_out)


def predict_lstm_score(
    model: LSTMAutoencoder,
    input_data: Union[List[float], List[List[float]], np.ndarray],
    threshold_scale: float = 0.25,
) -> float:
    """Compute normalized reconstruction anomaly score in range [0.0, 1.0]."""
    model.eval()
    with torch.no_grad():
        try:
            arr = np.array(input_data, dtype=np.float32)
            if arr.ndim == 1:
                tensor = torch.tensor(arr).unsqueeze(0).unsqueeze(0).repeat(1, 3, 1)
            elif arr.ndim == 2:
                tensor = torch.tensor(arr).unsqueeze(0)
            elif arr.ndim == 3:
                tensor = torch.tensor(arr)
            else:
                return 0.5

            reconstructed = model(tensor)
            mse = float(torch.mean((reconstructed - tensor) ** 2).item())
            score = float(1.0 - np.exp(-mse / max(1e-4, threshold_scale)))
            return float(max(0.0, min(1.0, score)))
        except Exception as exc:
            logger.error("lstm_prediction_failed", error=str(exc))
            return 0.5


def build_normal_sequences(num_sessions: int = 300) -> np.ndarray:
    """Assemble sequences of feature vectors from compliant sessions."""
    sequences: List[List[List[float]]] = []
    tracker.clear()
    current_sim_time = 1_700_000_000.0
    for _ in range(num_sessions):
        session_requests = generate_normal_session(base_time=current_sim_time)
        session_vectors: List[List[float]] = []
        for req in session_requests:
            features = extract_features_from_request(req)
            session_vectors.append(feature_dict_to_vector(features))
            if req.timestamp:
                current_sim_time = req.timestamp
        current_sim_time += random.uniform(30.0, 300.0)
        if session_vectors:
            sequences.append(session_vectors)
    return np.array(sequences, dtype=np.float32)


def evaluate_lstm_reconstruction(
    model: LSTMAutoencoder,
    eval_sequences: np.ndarray,
) -> Dict[str, float]:
    """Evaluate reconstruction error and attack separation."""
    normal_scores = [predict_lstm_score(model, seq) for seq in eval_sequences]
    avg_normal = float(np.mean(normal_scores)) if normal_scores else 0.0

    attack_vectors, _, _ = generate_all_attacks_dataset(reset_tracker_state=True)
    attack_scores = [predict_lstm_score(model, v) for v in attack_vectors]
    avg_attack = float(np.mean(attack_scores)) if attack_scores else 0.0

    return {
        "avg_normal_score": avg_normal,
        "avg_attack_score": avg_attack,
        "separation_margin": avg_attack - avg_normal,
    }


def train_lstm_autoencoder(
    num_sessions: int = 300,
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 0.005,
    save_artifact: bool = True,
    output_path: Optional[Path] = None,
) -> Tuple[LSTMAutoencoder, Dict[str, Any]]:
    """Train LSTM Autoencoder on normal interaction sequences."""
    target_path = output_path or ml_settings.lstm_model_path
    target_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("building_lstm_training_sequences", sessions=num_sessions)
    train_data = build_normal_sequences(num_sessions=num_sessions)

    model = LSTMAutoencoder(input_dim=12, hidden_dim=16, latent_dim=8)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    tensor_data = torch.tensor(train_data, dtype=torch.float32)
    dataset = torch.utils.data.TensorDataset(tensor_data)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    initial_loss: float = 0.0
    final_loss: float = 0.0
    for epoch in range(epochs):
        epoch_losses: List[float] = []
        for (batch_x,) in loader:
            optimizer.zero_grad()
            reconstructed = model(batch_x)
            loss = criterion(reconstructed, batch_x)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        avg_loss = float(np.mean(epoch_losses))
        if epoch == 0:
            initial_loss = avg_loss
        final_loss = avg_loss

    eval_data = build_normal_sequences(num_sessions=30)
    eval_metrics = evaluate_lstm_reconstruction(model, eval_data)
    eval_metrics.update({"initial_loss": initial_loss, "final_loss": final_loss})

    logger.info(
        "lstm_training_complete",
        initial_loss=round(initial_loss, 4),
        final_loss=round(final_loss, 4),
        avg_normal=round(eval_metrics["avg_normal_score"], 4),
        avg_attack=round(eval_metrics["avg_attack_score"], 4),
        margin=round(eval_metrics["separation_margin"], 4),
    )

    if save_artifact:
        torch.save(model.state_dict(), target_path)
        logger.info("lstm_artifact_saved", path=str(target_path))

    return model, eval_metrics


def load_lstm_autoencoder(
    model_path: Optional[Path] = None,
    input_dim: int = 12,
    hidden_dim: int = 16,
    latent_dim: int = 8,
) -> Optional[LSTMAutoencoder]:
    """Load persisted PyTorch LSTM Autoencoder weights from disk."""
    path = model_path or ml_settings.lstm_model_path
    try:
        if not path.exists():
            logger.warning("lstm_weights_not_found", path=str(path))
            return None
        model = LSTMAutoencoder(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
        state_dict = torch.load(path, map_location=torch.device("cpu"), weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()
        return model
    except Exception as exc:
        logger.error("failed_to_load_lstm_model", path=str(path), error=str(exc))
        return None


if __name__ == "__main__":
    train_lstm_autoencoder(save_artifact=True)
