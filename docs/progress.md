# SBP Sandbox Cohort 2: API Security Mesh — Progress Log

## Project Summary
- **Target**: State Bank of Pakistan (SBP) Regulatory Sandbox Cohort 2 (Theme 1: AI in Financial Services — Zero-Trust API Security)
- **Role**: Reverse-proxy AI anomaly detection mesh for Open Banking / EMI APIs
- **Inference Latency SLA**: < 200 ms per request
- **Verdict Pipeline**: APPROVE (< 0.4) / FLAG (0.4–0.7) / BLOCK (> 0.7) + Real SHAP Explanations

---

## Roadmap & Sub-Task Tracking

| Phase | Sub-Task | Description | Status | Verification |
|---|---|---|---|---|
| **Phase 1: Foundation** | 1.1 | Bootstrap repo, venv (Python 3.11), requirements, AGENT.md, progress log | 🟢 Completed | `.venv/bin/python --version`, imports verified |
| | 1.2 | Config & Core Data Schemas (`backend/config.py`, `backend/models.py`) | 🟢 Completed | `pytest -v tests/test_backend.py` (4 passed) |
| | 1.3 | FAPI-inspired Mock Open Banking Routers (`consent`, `accounts`, `transactions`, `transfer`) | 🟢 Completed | `pytest -v tests/test_routers.py` (4 passed) |
| | 1.4 | Starlette Reverse-Proxy Interceptor Middleware (`backend/middleware/interceptor.py`, `backend/main.py`) | 🟢 Completed | `pytest -v tests/test_interceptor.py` (5 passed) |
| **Phase 2: Features** | 2.1 | Sliding-window tracker (`ml/sliding_window.py`) | 🟢 Completed | `pytest -v tests/test_sliding_window.py` (6 passed) |
| | 2.2 | ML Config & Feature Engineering (`ml/config.py`, `ml/feature_engineering.py`) with Benford's Law & Pre-ML checks | 🟢 Completed | `pytest -v tests/test_features.py` (5 passed) |
| **Phase 3: Data & Models** | 3.1 | Normal Traffic Generator (`data/normal_traffic.py`) | 🟢 Completed | `pytest -v tests/test_normal_traffic.py` (3 passed) |
| | 3.2 | 7-Attack Vector Generator (`data/attack_traffic.py`) | 🟢 Completed | `pytest -v tests/test_attacks.py` (6 passed) |
| | 3.3 | Isolation Forest (`ml/train_isolation_forest.py`) | 🟢 Completed | Model artifact saved (1.6 MB) & `pytest -v tests/test_ml.py` (3 passed) |
| | 3.4 | Random Forest Classifier (`ml/train_rf.py`) | 🟢 Completed | Metrics (Precision=1.0, Recall=1.0, F1=1.0, ROC-AUC=1.0), artifact saved, 6/6 tests passing |
| | 3.5 | PyTorch LSTM Autoencoder (`ml/train_lstm.py`) | 🟢 Completed | Loss convergence (0.1228 -> 0.0016), artifact saved (`24 KB`), 9/9 tests passing |
| | 3.6 | Training Pipeline Orchestrator (`ml/train_all.py`) | 🟢 Completed | Single command `python -m ml.train_all`, 4/4 artifacts validated, 11/11 tests passing |
| **Phase 4: ML & SHAP** | 4.1 | SHAP Explainer Engine (`ml/shap_explainer.py`) | 🟢 Completed | Real TreeExplainer attribution, < 50ms latency verified, 5/5 tests passing in `tests/test_shap.py` |
| | 4.2 | Ensemble Scorer & Inference Pipeline (`ml/ensemble.py`, `ml/pipeline.py`) | 🟢 Completed | 5/5 tests passing in `tests/test_pipeline.py`, live interceptor integration verified, < 200ms latency budget |
| **Phase 5: E2E & UI** | 5.1 | Attack Simulator Tool (`tools/attack_simulator.py`) | 🟢 Completed | CLI scenario dispatcher, 100% detection rate, formatted reports, 4/4 tests passing in `tests/test_simulator.py` |
| | 5.2 | Seed Patterns / SQLite LRU Memory (`tools/seed_patterns.py`) | 🟢 Completed | SQLite bounded threat memory store, automatic LRU eviction triggers, 148 seeded patterns, 5/5 tests passing in `tests/test_seed_patterns.py` |
| | 5.3 | Streamlit Analyst Dashboard (`frontend/dashboard.py`) | 🟢 Completed | SBP green themed SOC console, real-time metrics, interactive SHAP attribution bar charts, SQLite pattern explorer, 3/3 tests passing in `tests/test_dashboard.py` |
| | 5.4 | Test Suite & Final End-to-End Verification (`tests/test_*.py`) | 🟢 Completed | Full project verification: 66/66 tests passing, live Uvicorn gateway + attack simulation demo verified, ~21.8ms avg latency vs 200ms SLA |
| **Phase 6: External Validation** | 6.1 | Download and verify real API security dataset (Zenodo APISentry #22658802) | 🟢 Completed | MD5 verified `7968bb87bf66b89092315bbeb253e086`, extracted 11,310 rows across 8 classes in `data/external/apisentry/` |
| | 6.2 | External Feature Adapter (`data/external/adapter.py`, `tests/test_adapter.py`) | 🟢 Completed | 12-dim vector mapping, safe float normalization, 0.0 default documentation, 4/4 tests passing in `tests/test_adapter.py` (70/70 total tests passing) |
| | 6.3 | Random Forest Re-training with 5-Fold CV (`ml/train_rf_external.py`, `tests/test_rf_external.py`) | 🟢 Completed | Stratified 70/15/15 split, 5-fold CV (macro-F1=0.9597±0.0038), test macro-F1=0.9574, binary ROC-AUC=0.9941, 5/5 tests passing (75/75 total tests passing) |
| | 6.4 | Isolation Forest Re-training & Percentile Calibration (`ml/calibration.py`, `ml/train_if_external.py`, `tests/test_if_external.py`) | 🟢 Completed | Empirical QuantileTransformer calibration, data-driven p95 threshold (0.9501), test ROC-AUC=0.8419, changelog `docs/IF_CALIBRATION_CHANGELOG.md`, 6/6 tests passing (81/81 total tests passing) |
| | 6.5 | Synthetic Pipeline Rewiring to PercentileCalibrator (`ml/ensemble.py`, `ml/pipeline.py`, `tests/test_ml.py`) | 🟢 Completed | Removed legacy hardcoded hack `0.38 - (raw * 1.2)`, generated `if_transformer.joblib`, full demo 100% detection rate (148 blocked, 3 approved), 81/81 tests passing |
| | 6.6 | Training Data Velocity Fix & Synthetic Retraining (`ml/sliding_window.py`, `data/normal_traffic.py`, `ml/train_*.py`) | 🟢 Completed | Fixed timestamp propagation bug (velocity dropped from 0.79 to 0.02), retrained IF/RF/LSTM, backed up old artifacts, verified live simulator (100% detection, avg latency 21.74ms), 83/83 tests passing |
| **Phase 7: Concurrency & Hardening** | 7.1 | Concurrency safety of Sliding Window (`ml/sliding_window.py`) | 🟢 Completed | Added `threading.RLock`, async lock, `async_record_and_compute()`, `count_10s()`, tested 100 concurrent workers, test added to `tests/test_sliding_window.py` |
| | 7.2 | Live Concurrent HTTP Request Testing (`tests/test_concurrent_http.py`) | 🟢 Completed | 50 concurrent requests (100% 201 Created) + 50 concurrent attacks (100% 403 Forbidden), 2/2 tests passing |
| | 7.3 | Concurrency & Multi-Worker Documentation (`docs/CONCURRENCY_LIMITATIONS.md`) | 🟢 Completed | Documented single-worker in-memory state vs Redis post-sandbox roadmap, LRU edge cases, SQLite lock analysis |
| | 7.4 | Evidence Consolidation & Canonical Log (`docs/LIVE_DEMO_CANONICAL.log`) | 🟢 Completed | Superseded intermediate logs deleted, fresh canonical run recorded with ISO header, linked in README.md |
| | 7.5 | Automated Reproduction Script (`scripts/reproduce_demo.sh`) | 🟢 Completed | Self-contained executable bash script reproducing training, evaluation, full 86-test suite, and live demo simulation from scratch |

---

## Session History

### Session 1 - Bootstrap & Setup
- **Date**: 2026-11-19
- **Completed**:
  - Sub-task 1.1: Repo skeleton, Python 3.11 `.venv`, requirements, `AGENT.md`, `BOOTSTRAP_TASK.md`, `.agents/`, `skills/`.
  - Sub-task 1.2: Central configuration (`backend/config.py`) and Pydantic schemas (`backend/models.py`) with 4/4 passing unit tests.
  - Sub-task 1.3: FAPI-compliant mock Open Banking routers (`/consent`, `/accounts`, `/transactions`, `/transfer`) with 4/4 passing unit tests.
  - Sub-task 1.4: Starlette reverse-proxy interceptor middleware (`backend/middleware/interceptor.py`, `backend/main.py`) with 5/5 passing integration tests (body preservation verified, 403 block semantics, JSONL audit logging).
  - Sub-task 2.1: Bounded LRU sliding-window tracker (`ml/sliding_window.py`) with 6/6 passing unit tests (hard 10k memory cap, flooding, burst, token replay, sequence anomaly).
  - Sub-task 2.2: Feature engineering pipeline (`ml/config.py`, `ml/feature_engineering.py`) with 5/5 passing unit tests (12-dim vector, Benford's Law distribution metric, scope check, SQL injection score, parameter tampering).
  - Sub-task 3.1: Synthetic normal traffic generator (`data/normal_traffic.py`) with 3/3 passing unit tests (mathematical Benford leading-digit distribution, legitimate multi-step banking session flow, clean non-anomalous baseline).
  - Sub-task 3.2: 7-Attack vector synthetic generator (`data/attack_traffic.py`) with 6/6 passing unit tests (SQL injection, parameter tampering, brute-force login, credential stuffing burst of 20, BOLA, API flooding burst of 100, token replay).
  - Sub-task 3.3: Isolation Forest anomaly detector (`ml/train_isolation_forest.py`) with 3/3 passing tests in `tests/test_ml.py` (model persistence to `ml/artifacts/isolation_forest.joblib`, positive separation margin on attacks, 36/36 overall project tests passing).
  - Sub-task 3.4: Random Forest classifier (`ml/train_rf.py`) with 6/6 passing tests in `tests/test_ml.py` (model persistence to `ml/artifacts/rf_model.joblib`, verified sklearn evaluation metrics: Precision 1.0, Recall 1.0, F1 1.0, ROC-AUC 1.0, 39/39 overall project tests passing).
  - Sub-task 3.5: PyTorch LSTM Autoencoder (`ml/train_lstm.py`) with 9/9 passing tests in `tests/test_ml.py` (model weights persistence to `ml/artifacts/lstm_autoencoder.pt`, verified loss convergence 0.1228 -> 0.0016, attack separation margin +0.4402, 42/42 overall project tests passing).
  - Sub-task 3.6: Master Training Pipeline Orchestrator (`ml/train_all.py`) with 11/11 passing tests in `tests/test_ml.py` (single master training command `python -m ml.train_all`, fits standard scaler, trains IF, RF, and LSTM, all 4 artifacts saved and validated, 44/44 overall project tests passing).
  - Sub-task 4.1: Real SHAP Explainability Engine (`ml/shap_explainer.py`, `tests/test_shap.py`) with 5/5 passing unit tests (TreeExplainer on Random Forest, genuine non-zero SHAP feature attributions, exact human-readable domain mappings, warm-start serialization, < 50ms latency SLA, 49/49 overall project tests passing).
  - Sub-task 4.2: Real-time Ensemble Scorer & Inference Pipeline (`ml/ensemble.py`, `ml/pipeline.py`, `tests/test_pipeline.py`) with 5/5 passing tests. End-to-end integration verified: live FastAPI interceptor integration, pre-ML deterministic overrides, weighted model blending (RF 0.45 + IF 0.30 + LSTM 0.25), dynamic SHAP explainability on FLAG and BLOCK, sub-200ms latency budget verified (averaging < 25ms). 54/54 overall project tests passing.
  - Sub-task 5.1: Attack Simulator Tool (`tools/attack_simulator.py`, `tests/test_simulator.py`) with 4/4 passing unit & integration tests. Dispatches normal and 7 attack scenarios in-process or over live network, validates 100% detection rate on attack vectors, outputs rich formatted terminal reports with SHAP explanations, verifies < 200ms SLA latency. 58/58 overall project tests passing.
  - Sub-task 5.2: Seed Patterns / SQLite LRU Memory Store (`tools/seed_patterns.py`, `tests/test_seed_patterns.py`) with 5/5 passing tests. SQLite threat database initialized at `logs/pattern_memory.db` with strict hard 10,000-entry capacity constraint, verified SQLite trigger + Python transaction LRU eviction, upsert with hit count increment, and 148 seeded patterns across all 7 attack vectors. 63/63 overall project tests passing.
  - Sub-task 5.3: Streamlit SOC Analyst Dashboard (`frontend/dashboard.py`, `tests/test_dashboard.py`) with 3/3 passing tests. Implemented interactive SBP green themed SOC console featuring live KPI metric cards, verdict & endpoint charts, tabular live request logs, local SHAP attribution bar charts ($\phi_i$) for incident investigation, SQLite threat pattern memory exploration, and performance SLA compliance telemetry (< 200 ms). 66/66 overall project tests passing.
  - Sub-task 5.4: Master Verification, End-to-End Workflow & Demo Validation. Entire system verified live against running FastAPI reverse-proxy server with all 7 attack vectors and legitimate compliant banking sessions. Observed average evaluation latency ~21.8 ms (9x faster than the < 200 ms SLA requirement). All 66 tests passing across 13 test suites. **100% Project Completion**.
- **Final Status**: ✅ All Phases and Sub-Tasks 1.1 through 5.4 100% Completed, Verified, and Documented.

### Session 2 - External Dataset Discovery & Adapter Mapping
- **Date**: 2026-11-20
- **Completed**:
  - Sub-task 6.1: Discovered and downloaded Zenodo APISentry dataset (#22658802, MD5 `7968bb87bf66b89092315bbeb253e086`). Extracted 11,310 rows across 8 classes (6,611 normal, 4,699 attacks across 7 attack vectors) in `data/external/apisentry/`.
  - Sub-task 6.2: Built `data/external/adapter.py` mapping raw/standardized external dataset columns to the project's internal 12-dimensional feature vector. Documented all mapping and derivation choices (zero-defaults for absent OAuth scopes or banking amounts, no faking) in `docs/EXTERNAL_DATASET_LIMITATIONS.md`. Implemented test suite `tests/test_adapter.py` (4/4 passed).
  - Sub-task 6.3: Re-trained Random Forest on APISentry via `ml/train_rf_external.py` with leak-free 70/15/15 stratified split and 5-fold CV hyperparameter tuning (`RandomizedSearchCV`). Achieved CV macro-F1 of 0.9597 ± 0.0038, test multiclass macro-F1 of 0.9574 (accuracy 97.29%), and test binary ROC-AUC of 0.9941. Saved `rf_external.joblib`, `rf_external_metrics.json`, and `rf_external_split_indices.json`. Verified with `tests/test_rf_external.py` (5/5 passed; 75/75 full suite tests passing).
  - Sub-task 6.4: Implemented reusable `PercentileCalibrator` in `ml/calibration.py` replacing hardcoded offset `0.38 - (raw * 1.2)` with empirical QuantileTransformer uniform mapping. Re-trained Isolation Forest on APISentry normal training samples (`ml/train_if_external.py`, 4,628 samples) with data-driven p95 APPROVE threshold (0.9501). Test ROC-AUC of 0.8419, honest binary F1 of 0.4054 (unsupervised novelty detector role), and clear attack separation (SQLi: 0.976, Tamper: 0.990 vs normal: 0.525). Saved `if_external.joblib`, `if_transformer_external.joblib`, `if_external_metrics.json`, and `docs/IF_CALIBRATION_CHANGELOG.md`. 6/6 tests passing in `tests/test_if_external.py` (81/81 full suite tests passing).
  - Sub-task 6.5: Rewired synthetic pipeline scoring layer to use `PercentileCalibrator` (`if_transformer.joblib`, `if_calibration_meta.json`). Updated `ml/ensemble.py`, `ml/pipeline.py`, and `ml/train_isolation_forest.py` to completely eliminate the legacy `0.38 - (raw * 1.2)` hack. Validated behavior across sample normal & attack requests (all normals APPROVE, all attacks BLOCK). Verified full live Uvicorn gateway simulation (148/148 attacks blocked, 3/3 normals approved, 21.59 ms avg latency). All 81 tests passing across 16 test suites.
  - Sub-task 6.6: Resolved training data velocity saturation bug (STEP 1.4d). Updated `ml/sliding_window.py` and `ml/feature_engineering.py` with explicit simulated timestamp propagation (`as_of`). Updated `data/normal_traffic.py` to include human pacing (1-5s delays, 30-300s inter-session gaps, deterministic client-account mapping), driving benign `request_rate_10s` down from 0.7937 to 0.0200. Retrained IF (separation margin +0.4714, normal test mean 0.4733), RF (F1 1.0), and LSTM Autoencoder (loss reduced 148x from 0.0743 to 0.0005, margin +0.4585). Verified live gateway simulation (151 requests, 100% anomaly detection rate, 21.74ms avg latency) and 100% passing test suite across all 83 tests.
  - Sub-tasks 7.1–7.5 (Phase 2): Concurrency safety hardening, 50-request concurrent HTTP test suite (`tests/test_concurrent_http.py`), thread-safe `SlidingWindowTracker` (`threading.RLock`), multi-worker limitation disclosure (`docs/CONCURRENCY_LIMITATIONS.md`), canonical live demo consolidation (`docs/LIVE_DEMO_CANONICAL.log`), and automated end-to-end reproduction script (`scripts/reproduce_demo.sh`). Total test suite: 86 passed / 0 failed.
- **Final Status**: ✅ Phase 1 and Phase 2 Complete, Verified, Audited, and Reproducible from Scratch.

### Session 3 - Phase 3a Housekeeping, Dashboard Screenshots & Warm-up Calibration Fix
- **Date**: 2026-11-21
- **Completed**:
  - Sub-task 8.1 (Housekeeping): Added git repository checkpoint, test coverage report (`pytest-cov`, 77% total / 88.5% runtime coverage), captured 4 SOC Analyst dashboard screenshots in `docs/screenshots/` and embedded them in `README.md`.
  - Sub-task 8.2 (Screenshot Verification): Screenshots verified unique (4 distinct md5 hashes).
  - Sub-task 8.3 (Warm-up Contamination & Latency Fix): Replaced contaminating `/consent` burst in warm-up with 20 `/health` requests (bypassing interceptor) plus 1 single warm-up request through the full pipeline under a unique client ID. Set single-sample inference mode (`n_jobs=1`) on RF and IF inside `SecurityPipeline` to eliminate scikit-learn thread synchronization overhead on per-request evaluations. Fixed seed determinism in `train_all_models` and `train_isolation_forest`.
  - Verification: Re-ran `scripts/reproduce_demo.sh` end-to-end. Verified canonical demo results:
    - Average latency: **15.94 ms** (returned to < 25ms, SLA budget < 200 ms)
    - Verdict breakdown: 3 APPROVE (normal scores 0.16, 0.33, 0.35 — all < 0.40), 1 FLAG (credential stuffing #2 early onset, 0.62), 147 BLOCK (HTTP 403)
    - Detection rate: 147/148 (99.3%)
    - Canonical log updated at `docs/LIVE_DEMO_CANONICAL.log`
    - Full test suite: **86 passed** / 0 failed in `pytest -v`.
- **Final Status**: ✅ Phase 3a Complete and Audited.

### Session 4 - Comprehensive Project Integrity Audit (Post-Phase-3a)
- **Date**: 2026-11-21
- **Completed**:
  - Executed 11-dimension integrity audit to verify that all documentation and evaluation claims are backed by actual code, actual computation, and genuine artifacts:
    1. *ML Model Authenticity*: All 8 joblib/pt artifacts verified non-empty with genuine weights/tensors (`rf_external`: 7.47MB, `if_external`: 2.75MB, `lstm_autoencoder`: 23.7KB).
    2. *Feature Engineering Computation*: Traced and probed all 12 features; verified non-static derivations and dynamic inputs.
    3. *Sliding Window & LRU Bounding*: Tested cap-100 LRU eviction (keys 0–99 purged) and strict temporal counting; verified `threading.RLock` coverage.
    4. *Ensemble & Calibration*: Grepped 0 matches for legacy hack `0.38 - raw * 1.2`; verified 0.45/0.30/0.25 config weights and `QuantileTransformer` monotonic calibration.
    5. *SHAP Explainability*: Verified dynamic `shap.TreeExplainer` game-theoretic attribution with contrasting attack inputs.
    6. *Attack Simulator Authenticity*: Probed live HTTP socket calls (`httpx`); manual `curl` and simulator produced identical HTTP 403 / BLOCK / SHAP reasons.
    7. *Verdict Thresholds*: Verified config-driven thresholds (`approve_threshold: 0.4`, `block_threshold: 0.7`) across all decision paths.
    8. *Test Suite Rigor*: 0 tautological asserts; executed 3 live mutation experiments on sliding window, calibration, and ensemble—all 3 mutations were immediately detected and failed tests.
    9. *External Dataset Verification*: Verified Zenodo APISentry dataset (11,310 rows); recomputed test macro-$F_1$ from raw CSV yielding exact match `0.957438`.
    10. *Documentation Reality Alignment*: Verified all 4 core disclosure documents match codebase reality with 0 discrepancies.
    11. *End-to-End Reproducibility*: Executed `./scripts/reproduce_demo.sh` from clean state; full retrain + 86 tests passed + canonical demo reproduced (147 BLOCK, 1 FLAG, 3 APPROVE, mean latency 12.70 ms).
  - Authored comprehensive audit document: `docs/CODE_INTEGRITY_AUDIT.md`.
- **Final Status**: ✅ Comprehensive Integrity Audit Complete. 11/11 Dimensions Verified (PASS). Repository fully verified for SBP Regulatory Sandbox Cohort 2 & FYP Defense.

## Session 5 — SBP Application Package Assembled — SBP_APPLICATION/ folder complete (7 files), ready for user to fill placeholders and submit by 28 Nov 2026
