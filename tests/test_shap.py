"""Unit tests for SHAP Explainability Engine and Feature Attribution."""

import time
from pathlib import Path
import pytest
from backend.models import SHAPExplanation
from ml.config import ml_settings
from ml.shap_explainer import (
    SecuritySHAPExplainer,
    get_shap_explainer,
    load_saved_shap_explainer,
    save_shap_explainer,
)
from ml.train_rf import load_random_forest


def test_shap_explainer_initialization() -> None:
    """Verify that SecuritySHAPExplainer loads RF model and builds TreeExplainer."""
    explainer = get_shap_explainer(force_reload=True)
    assert explainer is not None
    assert explainer.model is not None
    assert explainer.explainer is not None
    assert len(explainer.feature_names) == 12


def test_real_shap_attributions_on_attack_vectors() -> None:
    """Verify real non-zero SHAP values identify the active attack feature."""
    explainer = get_shap_explainer()

    # Create vector with pronounced SQL injection signature
    # Feature 7 = payload_sql_score, Feature 11 = endpoint_risk_weight
    sql_vec = [0.0] * 12
    sql_vec[7] = 1.0  # payload_sql_score
    sql_vec[10] = 1.0  # is_post_method
    sql_vec[11] = 1.0  # endpoint_risk_weight

    explanations = explainer.explain_vector(sql_vec, max_features=3)
    assert len(explanations) > 0
    assert all(isinstance(e, SHAPExplanation) for e in explanations)

    # Features identified must carry positive contribution
    features_identified = [e.feature for e in explanations]
    assert any(f in features_identified for f in ["payload_sql_score", "endpoint_risk_weight"])
    for e in explanations:
        assert e.contribution > 0.0
        assert len(e.description) > 0


def test_shap_explanation_human_readable_mapping() -> None:
    """Verify explanations map to exact human-readable domain descriptions."""
    explainer = get_shap_explainer()

    # Scope mismatch vector
    scope_vec = [0.0] * 12
    scope_vec[0] = 1.0  # scope_mismatch
    scope_vec[11] = 1.0  # endpoint_risk_weight

    explanations = explainer.explain_vector(scope_vec, max_features=2)
    assert len(explanations) > 0
    top_explanation = explanations[0]
    expected_desc = ml_settings.feature_descriptions.get(top_explanation.feature)
    assert top_explanation.description == expected_desc


def test_shap_explainer_serialization_and_loader(tmp_path: Path) -> None:
    """Verify TreeExplainer artifact saving and reloading."""
    explainer = get_shap_explainer()
    test_path = tmp_path / "test_shap.joblib"

    saved_path = save_shap_explainer(explainer, output_path=test_path)
    assert saved_path.exists()
    assert saved_path.stat().st_size > 0

    reloaded = load_saved_shap_explainer(test_path)
    assert reloaded is not None
    assert reloaded.explainer is not None


def test_shap_inference_latency_budget() -> None:
    """Verify single-sample TreeSHAP attribution completes well within latency budget (< 50ms)."""
    explainer = get_shap_explainer()
    test_vec = [0.5] * 12

    # Warmup
    _ = explainer.explain_vector(test_vec)

    # Benchmark
    t0 = time.perf_counter()
    for _ in range(10):
        _ = explainer.explain_vector(test_vec)
    avg_latency_ms = ((time.perf_counter() - t0) / 10.0) * 1000.0

    # TreeExplainer is fast (< 20ms typically, SLA threshold 50ms)
    assert avg_latency_ms < 50.0

