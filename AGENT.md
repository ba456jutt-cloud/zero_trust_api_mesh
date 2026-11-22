# AI-Powered API Security Mesh — Pakistan Open Banking (SBP Sandbox Cohort 2)

## Project Context
Real-time AI anomaly detection layer for Open Banking APIs in Pakistan.
Sits as a reverse proxy in front of bank/EMI APIs, inspects every request
(payload + behavioural), and returns APPROVE / FLAG / BLOCK verdict with
SHAP-based explanations.

Target: SBP Regulatory Sandbox Cohort 2 (Theme 1: AI in Financial Services —
zero-trust cybersecurity controls) + FYP / MS portfolio.

Current phase: SIMULATION (x86 Linux, mock APIs, synthetic traffic). Real bank
integration comes after sandbox acceptance. No real customer data at any point.

## Language & Stack
- Backend: Python 3.11 + FastAPI + Uvicorn + structlog
- Interception: Starlette middleware (reverse-proxy pattern)
- ML: scikit-learn (Isolation Forest, Random Forest), PyTorch (LSTM Autoencoder),
  SHAP (explainability)
- Data: synthetic only
- Dashboard: Streamlit (analyst view) — React variant later
- Testing: pytest + pytest-asyncio + httpx
- Storage: SQLite for pattern-memory store (no external DB in v1)
- Logs: JSON lines to `logs/verdicts.jsonl` (dashboard reads this)

## Build & Test Commands
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn backend.main:app --reload --port 8080
python -m ml.train_all              # train all models → ml/artifacts/
pytest -v                           # full test suite
python -m tools.attack_simulator --scenario all   # end-to-end demo
streamlit run frontend/dashboard.py
```

## File Structure
```text
api-security-mesh/
├── AGENT.md
├── BOOTSTRAP_TASK.md
├── README.md
├── requirements.txt
├── docs/
│   └── progress.md
├── backend/
│   ├── main.py
│   ├── config.py
│   ├── models.py
│   ├── middleware/
│   │   └── interceptor.py
│   └── routers/
│       ├── accounts.py
│       ├── transactions.py
│       ├── transfer.py
│       └── consent.py
├── ml/
│   ├── config.py
│   ├── sliding_window.py
│   ├── feature_engineering.py
│   ├── train_isolation_forest.py
│   ├── train_lstm.py
│   ├── train_rf.py
│   ├── train_all.py
│   ├── ensemble.py
│   ├── shap_explainer.py
│   ├── pipeline.py
│   └── artifacts/           (gitignored)
├── data/
│   ├── normal_traffic.py
│   └── attack_traffic.py
├── tools/
│   ├── attack_simulator.py
│   └── seed_patterns.py
├── frontend/
│   └── dashboard.py
├── tests/
│   ├── test_backend.py
│   ├── test_features.py
│   ├── test_ml.py
│   ├── test_attacks.py
│   └── test_shap.py
├── logs/                    (gitignored)
├── .agents/
│   └── agents/
│       ├── backend-architect.md
│       ├── ml-engineer.md
│       └── security-analyst.md
└── skills/
    ├── fastapi-scaffold/SKILL.md
    ├── ml-pipeline/SKILL.md
    └── shap-explainability/SKILL.md
```

## Coding Standards
- Python 3.11+, type hints on every public function
- snake_case functions/variables, PascalCase classes
- Async I/O in backend (FastAPI async def) — no blocking calls in middleware
- Max 60 lines per function
- Every external call (model load, SHAP explain, file write) must handle exceptions
- Use structlog (JSON); no print() in library code (CLI tools excepted)
- Thresholds/feature lists centralized in backend/config.py and ml/config.py

## Hard Constraints
- Inference latency target: < 200 ms per request (APISentry baseline)
- Middleware must not drop the request body — it must be re-readable after verdict
- No real customer data. Synthetic only. Ever.
- Mock endpoints must follow FAPI-inspired shapes (consent_id, X-Consent-Scope header simulating OAuth scope + mTLS presence)
- Every BLOCK / FLAG must carry a SHAP explanation — no silent blocks
- Pattern-memory store: cap 10,000 entries with LRU eviction (no unbounded growth)
- No dynamic allocation on the hot request path (middleware must be stateless, except for sliding-window counters held in a bounded structure)

## Security Rules (detection semantics)
- Verdict thresholds (configurable in backend/config.py):
  - ensemble_score < 0.4 → APPROVE (pass through)
  - 0.4 ≤ score ≤ 0.7 → FLAG (pass through + log + X-Security-Verdict header)
  - ensemble_score > 0.7 → BLOCK (403 + JSON explanation)

- 7 attack types that MUST be covered in data/attack_traffic.py:
  1. SQL injection
  2. Parameter tampering
  3. Brute-force login
  4. Credential stuffing (burst of 20)
  5. BOLA — broken object level authorization
  6. API flooding (burst of 100)
  7. Token replay

- Mandatory pre-ML checks (in feature_engineering, not just ML):
  - scope_mismatch: token scope (accounts:read / transfer:write) must match endpoint requirement
  - Benford's Law deviation on /transfer amounts is a first-class feature
  - Endpoint sequence anomaly: normal flow = consent → accounts → transfer; any skip is flagged

## Task Workflow
1. Read this file (AGENT.md)
2. Check docs/progress.md for last session state
3. Confirm current sub-task with user
4. Execute ONE file OR one logical group per turn
5. Run the relevant verification command
6. Show verification output (test pass, curl response, metrics table)
7. STOP — wait for user "continue"
8. Update docs/progress.md

## Anti-Patterns
❌ Multiple files in one turn (exception: bootstrap steps that explicitly allow it)
❌ Hardcoded verdicts (e.g. if endpoint == "/transfer": return "BLOCK")
❌ Fake SHAP values (must come from actual shap.Explainer calls)
❌ Training on test data / leaking attack signatures into normal training data
❌ Middleware that logs "BLOCK" but still calls call_next()
❌ Real PII or real bank data in any test
❌ Adding a model without registering it in ml/train_all.py
❌ Shipping without SHAP explanation on BLOCK verdicts

## Skills Available
Live in skills/. Load only when relevant.
- fastapi-scaffold — reverse proxy middleware, mock Open Banking endpoints, Pydantic schemas
- ml-pipeline — feature engineering, model training, ensemble weighting, evaluation
- shap-explainability — TreeExplainer / KernelExplainer setup, human-readable feature names

## Communication
Roman Urdu + English (technical terms). Direct, no fluff. Tables and code blocks.
When a parameter is genuinely unclear, ask — but do not ask about parameters already defined in this file or in backend/config.py / ml/config.py.

## Verification Philosophy
Every step must be independently verifiable by a human examiner:
- Each module has a test that fails if the module is stubbed
- Every ML metric is produced from actual sklearn.metrics calls, not hardcoded
- Every SHAP value is produced from actual shap.Explainer calls, not hardcoded
- Attack simulator must actually send malformed HTTP requests (visible in uvicorn access log)
- Final demo must be reproducible by running the documented commands on a clean checkout

