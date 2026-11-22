"""Unit tests for ML Anomaly Detection Models."""

from pathlib import Path
import pytest
from data.attack_traffic import generate_sql_injection_attack
from data.normal_traffic import generate_normal_dataset
from ml.config import ml_settings
from ml.feature_engineering import extract_features_from_request, feature_dict_to_vector
from ml.train_isolation_forest import (
    load_isolation_forest,
    predict_if_score,
    train_isolation_forest,
)
from ml.train_rf import (
    load_random_forest,
    predict_rf_probability,
    train_random_forest,
)
from ml.train_lstm import (
    load_lstm_autoencoder,
    predict_lstm_score,
    train_lstm_autoencoder,
)
from ml.train_all import (
    fit_and_save_scaler,
    verify_artifacts_exist,
)


def test_isolation_forest_training_and_persistence(tmp_path: Path) -> None:
    """Verify Isolation Forest training, evaluation separation, and serialization."""
    temp_artifact = tmp_path / "test_if.joblib"
    model, metrics = train_isolation_forest(
        num_sessions=60,
        contamination=0.03,
        save_artifact=True,
        output_path=temp_artifact,
    )

    # Verify model trained and artifact was saved
    assert model is not None
    assert temp_artifact.exists()
    assert temp_artifact.stat().st_size > 0

    # Verify metrics exhibit positive separation
    assert metrics["training_samples"] > 0
    assert metrics["avg_attack_score"] > metrics["avg_normal_score"]
    assert metrics["separation_margin"] > 0.0


def test_isolation_forest_scoring_bounds() -> None:
    """Verify predicted anomaly scores are bounded in [0.0, 1.0] and attacks score high."""
    # Quick model training for verification
    model, _ = train_isolation_forest(num_sessions=40, save_artifact=False)

    normal_vectors = generate_normal_dataset(num_sessions=5, reset_tracker_state=True)
    normal_scores = []
    for v in normal_vectors:
        score = predict_if_score(model, v)
        assert 0.0 <= score <= 1.0
        normal_scores.append(score)

    avg_normal = sum(normal_scores) / len(normal_scores)
    assert 0.0 <= avg_normal <= 1.0

    # Check attack vector receives higher score (extreme outlier)
    attack_req = generate_sql_injection_attack()[0]
    import json
    features = extract_features_from_request(
        endpoint=attack_req.endpoint,
        method=attack_req.method,
        headers=attack_req.headers,
        body_bytes=json.dumps(attack_req.body).encode(),
    )
    attack_vec = feature_dict_to_vector(features)
    attack_score = predict_if_score(model, attack_vec)
    assert 0.0 <= attack_score <= 1.0
    assert attack_score >= 0.80




def test_isolation_forest_loader(tmp_path: Path) -> None:
    """Verify loading from non-existent and existing model paths."""
    missing_path = tmp_path / "non_existent.joblib"
    assert load_isolation_forest(missing_path) is None

    # Save real model and reload
    real_path = tmp_path / "valid_if.joblib"
    train_isolation_forest(num_sessions=20, save_artifact=True, output_path=real_path)
    loaded = load_isolation_forest(real_path)
    assert loaded is not None


def test_random_forest_training_and_metrics(tmp_path: Path) -> None:
    """Verify Random Forest training, persistence, and real classification metrics."""
    test_artifact = tmp_path / "test_rf.joblib"
    model, metrics = train_random_forest(
        num_normal_sessions=60,
        attack_multipliers=2,
        test_size=0.25,
        save_artifact=True,
        output_path=test_artifact,
    )

    assert model is not None
    assert test_artifact.exists()
    assert test_artifact.stat().st_size > 0

    # Verify verified sklearn metrics
    assert metrics["precision"] >= 0.85
    assert metrics["recall"] >= 0.85
    assert metrics["f1_score"] >= 0.85
    assert metrics["roc_auc"] >= 0.90
    assert metrics["test_samples"] > 0


def test_random_forest_probability_scoring() -> None:
    """Verify RF probability bounds and anomaly discrimination."""
    model, _ = train_random_forest(
        num_normal_sessions=50,
        attack_multipliers=2,
        save_artifact=False,
    )

    # Test normal traffic vectors produce low anomaly probabilities
    normal_vectors = generate_normal_dataset(num_sessions=5, reset_tracker_state=True)
    for v in normal_vectors:
        prob = predict_rf_probability(model, v)
        assert 0.0 <= prob <= 1.0
        assert prob < 0.40

    # Test attack traffic produces high anomaly probability
    attack_req = generate_sql_injection_attack()[0]
    import json
    features = extract_features_from_request(
        endpoint=attack_req.endpoint,
        method=attack_req.method,
        headers=attack_req.headers,
        body_bytes=json.dumps(attack_req.body).encode(),
    )
    attack_vec = feature_dict_to_vector(features)
    attack_prob = predict_rf_probability(model, attack_vec)
    assert 0.0 <= attack_prob <= 1.0
    assert attack_prob > 0.70


def test_random_forest_loader(tmp_path: Path) -> None:
    """Verify error safety of Random Forest loader."""
    missing_path = tmp_path / "non_existent_rf.joblib"
    assert load_random_forest(missing_path) is None

    real_path = tmp_path / "valid_rf.joblib"
    train_random_forest(num_normal_sessions=20, attack_multipliers=1, save_artifact=True, output_path=real_path)
    loaded = load_random_forest(real_path)
    assert loaded is not None


def test_lstm_autoencoder_training_and_convergence(tmp_path: Path) -> None:
    """Verify LSTM Autoencoder trains, converges (loss decreases), and persists weights."""
    test_artifact = tmp_path / "test_lstm.pt"
    model, metrics = train_lstm_autoencoder(
        num_sessions=50,
        epochs=12,
        batch_size=16,
        save_artifact=True,
        output_path=test_artifact,
    )

    assert model is not None
    assert test_artifact.exists()
    assert test_artifact.stat().st_size > 0

    # Verify loss decreased from epoch 0 to final
    assert metrics["final_loss"] < metrics["initial_loss"]
    assert metrics["avg_attack_score"] > metrics["avg_normal_score"]


def test_lstm_autoencoder_scoring_bounds() -> None:
    """Verify LSTM reconstruction score is properly bounded in [0.0, 1.0]."""
    model, _ = train_lstm_autoencoder(
        num_sessions=30,
        epochs=8,
        save_artifact=False,
    )

    # Test 1D vector input
    normal_vector = [0.0] * 12
    score_1d = predict_lstm_score(model, normal_vector)
    assert 0.0 <= score_1d <= 1.0

    # Test 2D sequence input
    seq_input = [[0.0] * 12, [0.0] * 12, [0.0] * 12]
    score_2d = predict_lstm_score(model, seq_input)
    assert 0.0 <= score_2d <= 1.0


def test_lstm_autoencoder_loader(tmp_path: Path) -> None:
    """Verify error safety of LSTM Autoencoder model weight loading."""
    missing_path = tmp_path / "non_existent_lstm.pt"
    assert load_lstm_autoencoder(missing_path) is None

    real_path = tmp_path / "valid_lstm.pt"
    train_lstm_autoencoder(num_sessions=20, epochs=5, save_artifact=True, output_path=real_path)
    loaded = load_lstm_autoencoder(real_path)
    assert loaded is not None


def test_fit_and_save_scaler(tmp_path: Path) -> None:
    """Verify feature scaler fitting, serialization, and transform shape."""
    scaler_path = tmp_path / "test_scaler.joblib"
    scaler = fit_and_save_scaler(num_sessions=30, output_path=scaler_path)
    assert scaler is not None
    assert scaler_path.exists()
    assert scaler_path.stat().st_size > 0

    # Test transformation
    dummy_vec = [[0.0] * 12]
    transformed = scaler.transform(dummy_vec)
    assert transformed.shape == (1, 12)


def test_artifacts_verification() -> None:
    """Verify verify_artifacts_exist returns dictionary with expected keys."""
    status = verify_artifacts_exist()
    assert "scaler" in status
    assert "isolation_forest" in status
    assert "random_forest" in status
    assert "lstm_autoencoder" in status




