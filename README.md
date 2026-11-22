# SBP AI API Security Mesh

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![Tests Passing](https://img.shields.io/badge/tests-86%20passed-brightgreen.svg)](tests/)
[![Coverage](https://img.shields.io/badge/coverage-88.5%25%20runtime-success.svg)](docs/COVERAGE_REPORT.txt)
[![Audit](https://img.shields.io/badge/integrity%20audit-11%2F11%20PASS-brightgreen.svg)](docs/CODE_INTEGRITY_AUDIT.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![SBP Regulatory Sandbox](https://img.shields.io/badge/SBP%20Sandbox-Cohort%202%20(Theme%201)-darkgreen.svg)](SBP_APPLICATION/)

> **Real-Time, Explainable AI Anomaly Detection & Zero-Trust Reverse-Proxy Mesh for Pakistan Open Banking & EMI APIs**

---

## 1. Overview

As Pakistan rapidly transitions to digital financial infrastructure through **Raast Instant Payments**, licensed **Electronic Money Institutions (EMIs)**, and the upcoming **State Bank of Pakistan (SBP) Open Banking Regulations 2026 (draft)**, commercial banks and fintechs are required to expose customer account and payment initiation endpoints to Third-Party Providers (TPPs) via public APIs.

While Open Banking unlocks immense financial inclusion, it exposes financial institutions to severe attack vectors (BOLA, credential stuffing, SQL injection, token replay, parameter tampering) highlighted in the **National CERT (PK-CERT) Financial Cybersecurity Advisory (Nov 2026)**. Traditional Web Application Firewalls (WAFs) rely on static signatures and regex rules that fail to detect multi-step behavioral account hopping or deviations from legitimate banking transaction distributions.

The **SBP AI API Security Mesh** operates as an in-line, ultra-low-latency reverse proxy positioned transparently in front of Open Banking core adapters. It analyzes every incoming HTTP request across payload and behavioral sequence dimensions using a multi-model machine learning ensemble, rendering deterministic, explainable verdicts in under **16 ms** ($>12.5\times$ SLA headroom against the SBP $< 200\text{ ms}$ requirement).

---

## 2. Key Features

- **Multi-Tier AI Ensemble**: Combines a Supervised Random Forest ($w=0.45$), a Percentile-Calibrated Isolation Forest ($w=0.30$), and a Deep PyTorch LSTM Autoencoder ($w=0.25$).
- **7 OWASP API Attack Vectors Covered**: SQL Injection, Parameter Tampering, Brute-Force Login, Credential Stuffing, Broken Object Level Authorization (BOLA), API Flooding, and Token Replay.
- **Explainable AI (No Black Boxes)**: Employs game-theoretic `shap.TreeExplainer` on every blocked or flagged request; returns local Shapley attributions directly in `HTTP 403` payloads and `X-Security-Reason` response headers.
- **Financial Domain Feature Engineering**: Evaluates transaction amounts against **Benford's Law distribution** ($P(d) = \log_{10}(1 + 1/d)$) on PKR currency flows, enforces FAPI OAuth token scope boundaries, and validates multi-step endpoint workflows (`/consent` $\to$ `/accounts` $\to$ `/transfer`).
- **Memory-Bounded Behavioral Tracking**: Enforces a strict **10,000-entry LRU cap** with fine-grained `threading.RLock` synchronization to monitor 10s/60s burst rates and account probing ratios without memory leakage.
- **Enterprise SOC Analyst Dashboard**: Live Streamlit console in SBP corporate green featuring real-time KPI gauges, latency monitors, interactive SHAP attribution charts, and a SQLite threat pattern explorer.
- **Independent Benchmark Validation**: Validated on **11,310 real-world API requests** from the Zenodo APISentry benchmark (#22658802), achieving a macro-$F_1$ of **0.9574**.

---

## 3. Architecture

```text
[TPP / Fintech Client]
          │
          ▼  HTTPS / FAPI Request
┌─────────────────────────────────────────────────────────────────┐
│               AI API Security Mesh (FastAPI / Starlette)        │
│                                                                 │
│ 1. Request Interception & Stream Preservation                   │
│ 2. Sliding Window Behavioral Tracking (10k LRU Bounded Cache)   │
│ 3. 12-Dimensional Feature Engineering (Structural, Seq, Benford)│
│ 4. Multi-Tier AI Ensemble Inference:                            │
│    ├── Supervised Random Forest Classifier     (Weight: 0.45)   │
│    ├── Percentile-Calibrated Isolation Forest  (Weight: 0.30)   │
│    └── PyTorch LSTM Autoencoder Reconstruction (Weight: 0.25)   │
│ 5. TreeSHAP Mathematical Attribution Engine                     │
│ 6. Three-Tier Regulatory Verdict Decision                       │
└────────────────────────────────┬────────────────────────────────┘
          │                      │
   [If Score ≥ 0.70]      [If Score < 0.70]
          │                      │
          ▼                      ▼
  HTTP 403 Forbidden     Forward to Core Banking Gateway
  + Real-time SHAP       (/consent, /accounts, /transactions, /transfer)
    Reason Payload       (With X-Security-Verdict: APPROVE / FLAG)
```

### Three-Tier Verdict Logic:
- **`APPROVE`** ($\text{score} < 0.40$): Request passes through to core banking endpoints.
- **`FLAG`** ($0.40 \le \text{score} < 0.70$): Request passes through with enriched `X-Security-Verdict` audit headers for SOC review.
- **`BLOCK`** ($\text{score} \ge 0.70$): Intercepted immediately with `HTTP 403 Forbidden` and SHAP explanations.

---

## 4. Quick Start (5 Commands)

For SBP evaluators and developers, run the entire pipeline with:

```bash
# 1. Clone repository
git clone https://github.com/ba456jutt-cloud/zero_trust_api_mesh.git
cd sbp-api-security-mesh

# 2. Set up virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install pinned dependencies
pip install -r requirements.txt

# 4. Run automated end-to-end reproduction (trains models, runs tests, executes live demo)
./scripts/reproduce_demo.sh
```

---

## 5. Detailed Step-by-Step Setup

### Step 5.1: Environment Prerequisites
- Linux / macOS / WSL2 (Ubuntu 22.04+ recommended)
- Python 3.11
- `curl`

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Step 5.2: Verify External Dataset
The APISentry dataset (`data/external/apisentry/dataset_processed.csv`, 11,310 rows) is included in the repository. If missing, fetch it automatically from Zenodo:
```bash
./scripts/fetch_dataset.sh
```

---

## 6. How to Run Each Component

### 1. Train Synthetic Models
Trains Feature Scaler, Isolation Forest, Random Forest ($F_1 = 1.0$), and PyTorch LSTM Autoencoder:
```bash
python -m ml.train_all
```
*Artifacts saved to `ml/artifacts/`.*

### 2. Train External Benchmark Models
Trains Random Forest and Percentile-Calibrated Isolation Forest on the APISentry benchmark:
```bash
python -m ml.train_rf_external
python -m ml.train_if_external
```

### 3. Run Automated Test Suite
Executes all 86 unit and integration test cases across 16 test files:
```bash
pytest -v
```

To view code coverage breakdown:
```bash
pytest --cov=backend --cov=ml --cov=data --cov=tools
```

### 4. Start the Reverse-Proxy Gateway
Launches the FastAPI gateway on port 8080:
```bash
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8080 --workers 1
```
Interactive API documentation is accessible at `http://127.0.0.1:8080/docs`.

### 5. Run the Attack Simulator
In a separate terminal, dispatch the 7 attack scenarios over real HTTP loopback sockets:
```bash
# Run all 148 attack scenarios + legitimate baseline
python -m tools.attack_simulator -t http://127.0.0.1:8080 -s all

# Or run specific scenarios:
python -m tools.attack_simulator -t http://127.0.0.1:8080 -s sqli
python -m tools.attack_simulator -t http://127.0.0.1:8080 -s stuffing
python -m tools.attack_simulator -t http://127.0.0.1:8080 -s bola
python -m tools.attack_simulator -t http://127.0.0.1:8080 -s flooding
python -m tools.attack_simulator -t http://127.0.0.1:8080 -s replay
```

### 6. Launch the SOC Analyst Dashboard
Populate the SQLite pattern memory store and start the Streamlit web console:
```bash
python -m tools.seed_patterns
streamlit run frontend/dashboard.py
```
Open your browser at `http://localhost:8501`.

---

## 7. Performance & Verification Metrics

Every metric below is backed by reproducible computation and verified in our [docs/CODE_INTEGRITY_AUDIT.md](docs/CODE_INTEGRITY_AUDIT.md):

| Dimension / Metric | SBP Regulatory Target | Audited Codebase Reality | Verification Source |
|---|---|---|---|
| **Attack Detection Rate** | $\ge 98.0\%$ | **99.3% Hard Block (147/148)**<br>**100.0% Intercepted (148/148)** | [docs/LIVE_DEMO_CANONICAL.log](docs/LIVE_DEMO_CANONICAL.log) |
| **False Block Rate (Normals)**| $\le 0.1\%$ | **0.0% (3/3 Approved, scores $< 0.36$)** | [docs/LIVE_DEMO_CANONICAL.log](docs/LIVE_DEMO_CANONICAL.log) |
| **Inference Latency** | $< 200.00\text{ ms}$ | **12.70–15.94 ms average** ($>12.5\times$ SLA headroom) | [docs/LIVE_DEMO_CANONICAL.log](docs/LIVE_DEMO_CANONICAL.log) |
| **External Benchmark Quality**| Macro-$F_1 \ge 0.90$ | **Macro-$F_1 = 0.9574$** (Accuracy: 97.29%, ROC-AUC: 0.9941) | [ml/artifacts/rf_external_metrics.json](ml/artifacts/rf_external_metrics.json) |
| **Automated Test Rigor** | 100% pass | **86 passed / 0 failed in 8.35s** | [docs/COVERAGE_REPORT.txt](docs/COVERAGE_REPORT.txt) |
| **Runtime Code Coverage** | $> 80.0\%$ | **88.5% runtime coverage** | [docs/COVERAGE_REPORT.txt](docs/COVERAGE_REPORT.txt) |
| **Explainability Compliance**| $100\%$ on BLOCK/FLAG | **100% compliant** (Real TreeSHAP attributions) | `ml/shap_explainer.py` |
| **State Memory Safety** | Bounded $< 100\text{ MB}$ | **10,000-entry LRU cap strictly enforced** | `tests/test_sliding_window.py` |

---

## 8. External Benchmark Validation (APISentry)

To prevent synthetic evaluation bias, the architecture was independently evaluated against the **APISentry** dataset ([Zenodo record #22658802](https://zenodo.org/records/22658802)), containing 11,310 real-world API requests across 8 classes.

- **Split**: Leak-free stratified 70/15/15 train/val/test split ([ml/artifacts/rf_external_split_indices.json](ml/artifacts/rf_external_split_indices.json)).
- **Cross-Validation**: 5-fold CV macro-$F_1 = 0.9597 \pm 0.0038$.
- **Test Performance**:
  - Macro-$F_1$: **`0.9574`**
  - Weighted-$F_1$: **`0.9724`**
  - Accuracy: **`97.29%`**
  - Binary ROC-AUC: **`0.9941`**

---

## 9. Documentation Index

The repository maintains transparent, comprehensive documentation:

- [docs/CODE_INTEGRITY_AUDIT.md](docs/CODE_INTEGRITY_AUDIT.md) — Formal 11-dimension integrity audit report (11/11 PASS).
- [docs/LIVE_DEMO_CANONICAL.log](docs/LIVE_DEMO_CANONICAL.log) — Canonical 151-request live gateway log with latencies and verdicts.
- [docs/IF_CALIBRATION_CHANGELOG.md](docs/IF_CALIBRATION_CHANGELOG.md) — Empirical QuantileTransformer calibration and 56-point IF gap rationale.
- [docs/EXTERNAL_DATASET_LIMITATIONS.md](docs/EXTERNAL_DATASET_LIMITATIONS.md) — Feature derivation mapping for the APISentry external dataset.
- [docs/CONCURRENCY_LIMITATIONS.md](docs/CONCURRENCY_LIMITATIONS.md) — Single-worker RLock analysis and multi-worker Redis production scaling roadmap.
- [docs/COVERAGE_REPORT.txt](docs/COVERAGE_REPORT.txt) — Line-by-line test coverage report (88.5% runtime coverage).
- [SBP_APPLICATION/](SBP_APPLICATION/) — SBP Regulatory Sandbox Cohort 2 formal application package (Q1 through Q37).

---

## 10. Known Limitations & Honest Disclosures

In accordance with State Bank of Pakistan sandbox principles, all technical boundaries are disclosed:

1. **Single-Worker In-Memory State**: The sliding-window tracker (`SlidingWindowTracker`) holds state in Python process memory with `threading.RLock` synchronization. It safely supports 50 concurrent test users at $< 25\text{ ms}$ latency on a single Uvicorn worker. Deploying across multi-worker clusters requires migrating state to **Redis sorted sets** (`ZADD`/`ZCOUNT`), which is documented as a post-sandbox production item in [docs/CONCURRENCY_LIMITATIONS.md](docs/CONCURRENCY_LIMITATIONS.md).
2. **APISentry Feature Mapping**: As disclosed in [docs/EXTERNAL_DATASET_LIMITATIONS.md](docs/EXTERNAL_DATASET_LIMITATIONS.md), APISentry lacks banking-specific fields. Exactly **8 features are direct**, **2 are proxy**, and **2 features** (`scope_mismatch` and `benford_deviation`) default strictly to `0.0`. Full 12-feature evaluation is demonstrated on the synthetic FAPI Open Banking suite.
3. **Unsupervised Isolation Forest Sensitivity**: Unsupervised Isolation Forest achieves $F_1 = 0.9653$ on synthetic data and $F_1 = 0.4054$ on subtle external data. The ensemble architecture explicitly mitigates this by weighting supervised Random Forest ($F_1 = 0.9574$) at 0.45, retaining Isolation Forest (0.30 weight) strictly for zero-day novelty detection.

---

## 11. Regulatory Alignment

- **SBP Regulatory Sandbox Cohort 2**: Aligned with **Theme 1: AI in Financial Services** (Zero-Trust Cybersecurity Controls).
- **SBP Open Banking Regulations 2026 (draft)**: Designed to protect FAPI-compliant `/consent`, `/accounts`, `/transactions`, and `/transfer` interfaces.
- **SBP Cybersecurity Framework**: Implements continuous behavioral inspection, proactive threat containment, and zero-trust verification.

---

## 12. License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

The APISentry dataset is redistributed under its original open research attribution from Zenodo (Record #22658802).

---

## 13. Maintainer & Contact

- **Applicant / Lead Engineer**: [YOUR_NAME] ([YOUR_EMAIL])
- **Proposed Entity**: [YOUR_NAME] AI Labs (Pvt.) Limited (Pre-incorporation, Path A)
- **Repository**: `https://github.com/ba456jutt-cloud/zero_trust_api_mesh`
- **Application Dossier**: See [SBP_APPLICATION/01_applicant_details.md](SBP_APPLICATION/01_applicant_details.md)
