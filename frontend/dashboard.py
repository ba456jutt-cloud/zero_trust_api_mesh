"""State Bank of Pakistan (SBP) Regulatory Sandbox Cohort 2.

AI-Powered API Security Mesh — SOC Analyst & Regulatory Dashboard.
Provides real-time zero-trust inspection telemetry, interactive SHAP
explainability drill-downs, and SQLite threat pattern memory exploration.
"""

from datetime import datetime
import json
from pathlib import Path
import sys

# Ensure repository root is in sys.path
_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from typing import Any, Dict, List, Optional
import pandas as pd
import streamlit as st
from backend.config import VerdictEnum, settings
from tools.seed_patterns import PatternMemoryStore


# ---------------------------------------------------------------------------
# Data Loading & Ingestion
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3.0)
def load_verdict_logs(log_path_str: str) -> pd.DataFrame:
    """Read JSONL verdict logs produced by reverse-proxy interceptor middleware."""
    log_file = Path(log_path_str)
    if not log_file.exists():
        return pd.DataFrame()

    records: List[Dict[str, Any]] = []
    try:
        with open(log_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    records.append(data)
                except Exception:
                    continue
    except Exception:
        return pd.DataFrame()

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    if "timestamp" in df.columns:
        df["datetime"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.sort_values(by="datetime", ascending=False).reset_index(drop=True)
    return df


def get_pattern_store() -> PatternMemoryStore:
    """Retrieve SQLite threat pattern memory store instance."""
    return PatternMemoryStore()


# ---------------------------------------------------------------------------
# Streamlit Dashboard Rendering
# ---------------------------------------------------------------------------
def render_dashboard() -> None:
    """Render full interactive SOC dashboard layout and components."""
    st.set_page_config(
        page_title="SBP API Security Mesh — SOC Dashboard",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Custom Styling
    st.markdown(
        """
        <style>
        .main-header {
            font-size: 26px;
            font-weight: 700;
            color: #0b6623;
            margin-bottom: 2px;
        }
        .sub-header {
            font-size: 14px;
            color: #555555;
            margin-bottom: 20px;
        }
        .metric-card {
            background-color: #f8f9fa;
            border-radius: 8px;
            padding: 14px;
            border-left: 4px solid #0b6623;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar
    with st.sidebar:
        st.markdown("## 🏛️ **State Bank of Pakistan**")
        st.markdown("**Regulatory Sandbox Cohort 2**")
        st.caption("Theme 1: AI in Financial Services — Zero-Trust API Security")

        st.divider()
        st.subheader("⚙️ Control Panel")
        if st.button("🔄 Manual Refresh"):
            st.cache_data.clear()
            st.rerun()

        st.divider()
        st.subheader("🔍 Telemetry Filters")
        verdict_filter = st.multiselect(
            "Filter by Verdict",
            options=["ALL", "APPROVE", "FLAG", "BLOCK"],
            default=["ALL"],
        )

        endpoint_filter = st.selectbox(
            "Target Endpoint",
            options=["ALL", "/consent", "/accounts", "/transactions", "/transfer"],
            index=0,
        )

        min_score = st.slider("Min Anomaly Score", min_value=0.0, max_value=1.0, value=0.0, step=0.05)

    # Header
    st.markdown('<div class="main-header">🛡️ SBP AI-Powered API Security Mesh — SOC Gateway</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Inline Zero-Trust Traffic Inspection, Real-Time Ensemble ML Anomaly Detection & Genuine SHAP Explainability</div>',
        unsafe_allow_html=True,
    )

    # Load Logs
    log_path = str(settings.verdicts_log_path)
    df_raw = load_verdict_logs(log_path)

    if df_raw.empty:
        st.warning("⚠️ No traffic logged in `logs/verdicts.jsonl` yet. Run `python -m tools.attack_simulator --in-process` to generate live telemetry.")
        return

    # Filter DataFrame
    df = df_raw.copy()
    if "ALL" not in verdict_filter and verdict_filter:
        df = df[df["verdict"].isin(verdict_filter)]
    if endpoint_filter != "ALL":
        df = df[df["endpoint"] == endpoint_filter]
    if "anomaly_score" in df.columns:
        df = df[df["anomaly_score"] >= min_score]

    # KPI Metrics
    total_reqs = len(df_raw)
    approve_count = len(df_raw[df_raw["verdict"] == "APPROVE"])
    flag_count = len(df_raw[df_raw["verdict"] == "FLAG"])
    block_count = len(df_raw[df_raw["verdict"] == "BLOCK"])
    avg_latency = df_raw["latency_ms"].mean() if "latency_ms" in df_raw.columns else 0.0
    block_rate = (block_count / total_reqs * 100.0) if total_reqs > 0 else 0.0

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        st.metric("Total Inspected", f"{total_reqs:,}")
    with col2:
        st.metric("Approved (200/201)", f"{approve_count:,}", delta=f"{approve_count/total_reqs*100:.1f}%")
    with col3:
        st.metric("Flagged (Review)", f"{flag_count:,}", delta=f"{flag_count/total_reqs*100:.1f}%", delta_color="off")
    with col4:
        st.metric("Blocked (HTTP 403)", f"{block_count:,}", delta=f"-{block_rate:.1f}%", delta_color="inverse")
    with col5:
        st.metric("Avg Latency", f"{avg_latency:.1f} ms", delta="< 200 ms SLA Target", delta_color="normal")
    with col6:
        st.metric("Block Rate", f"{block_rate:.1f}%", delta="Threat Containment")

    st.divider()

    # Tabs
    tab_traffic, tab_shap, tab_memory, tab_compliance = st.tabs(
        [
            "📊 Live Traffic & Visualizations",
            "🔬 SHAP Attribution & Threat Investigation",
            "🧠 SQLite Threat Pattern Memory",
            "📜 Regulatory Compliance & SLA Telemetry",
        ]
    )

    with tab_traffic:
        col_chart1, col_chart2 = st.columns([1, 1])
        with col_chart1:
            st.subheader("Verdict Breakdown")
            verdict_counts = df_raw["verdict"].value_counts().reset_index()
            verdict_counts.columns = ["Verdict", "Count"]
            st.bar_chart(data=verdict_counts.set_index("Verdict"), use_container_width=True)

        with col_chart2:
            st.subheader("Target Endpoint Activity")
            endpoint_counts = df_raw["endpoint"].value_counts().reset_index()
            endpoint_counts.columns = ["Endpoint", "Count"]
            st.bar_chart(data=endpoint_counts.set_index("Endpoint"), use_container_width=True)

        st.subheader(f"Recent Request Telemetry ({len(df)} records matching filters)")
        display_cols = ["timestamp", "client_id", "http_method", "endpoint", "anomaly_score", "verdict", "latency_ms", "request_id"]
        display_df = df[[c for c in display_cols if c in df.columns]].copy()
        st.dataframe(display_df, use_container_width=True, height=360)

    with tab_shap:
        st.subheader("Local SHAP Feature Attribution Drill-Down")
        st.caption("Inspect genuine mathematical TreeSHAP attributions for flagged and blocked transactions.")

        candidates = df_raw[df_raw["verdict"].isin(["BLOCK", "FLAG"])].copy()
        if candidates.empty:
            st.info("No BLOCKED or FLAGGED requests recorded yet to examine.")
        else:
            req_options = [
                f"{r.get('timestamp', '')[:19]} | {r.get('verdict')} | {r.get('http_method')} {r.get('endpoint')} | Score: {r.get('anomaly_score')} | ID: {r.get('request_id', '')[:8]}"
                for _, r in candidates.iterrows()
            ]
            selected_idx = st.selectbox("Select Request to Investigate:", range(len(req_options)), format_func=lambda i: req_options[i])
            selected_record = candidates.iloc[selected_idx]

            col_meta1, col_meta2, col_meta3, col_meta4 = st.columns(4)
            with col_meta1:
                st.write(f"**Client ID:** `{selected_record.get('client_id')}`")
                st.write(f"**Endpoint:** `{selected_record.get('endpoint')}`")
            with col_meta2:
                st.write(f"**HTTP Method:** `{selected_record.get('http_method')}`")
                st.write(f"**Verdict:** `{selected_record.get('verdict')}`")
            with col_meta3:
                st.write(f"**Anomaly Score:** `{selected_record.get('anomaly_score')}`")
                st.write(f"**Latency:** `{selected_record.get('latency_ms', 0):.2f} ms`")
            with col_meta4:
                st.write(f"**Timestamp:** `{selected_record.get('timestamp')}`")
                st.write(f"**Request ID:** `{selected_record.get('request_id')}`")

            st.divider()

            explanations = selected_record.get("explanations", [])
            if explanations and isinstance(explanations, list):
                st.markdown(r"#### **SHAP Feature Contributions ($\phi_i$)**")
                exp_df = pd.DataFrame(explanations)
                if "contribution" in exp_df.columns and "feature" in exp_df.columns:
                    chart_df = exp_df[["feature", "contribution"]].sort_values(by="contribution", ascending=True)
                    st.bar_chart(chart_df.set_index("feature"), use_container_width=True)

                    st.markdown("#### **Regulatory & Technical Explanations**")
                    for _, row in exp_df.iterrows():
                        st.markdown(
                            f"- **`{row.get('feature')}`** (SHAP Attribution: `+{row.get('contribution', 0.0):.4f}`): "
                            f"_{row.get('description', '')}_"
                        )
            else:
                st.info("No detailed SHAP attributions found for this record.")

    with tab_memory:
        st.subheader("Persistent SQLite Threat Pattern Memory (`logs/pattern_memory.db`)")
        st.caption("Bounded storage constraint: Maximum 10,000 unique threat signatures with automatic LRU eviction.")

        store = get_pattern_store()
        pattern_count = store.count_patterns()
        st.write(f"**Total Patterns in Memory:** `{pattern_count:,} / 10,000` (Bounded Hard Cap)")

        recent_patterns = store.get_recent_patterns(limit=100)
        if recent_patterns:
            pat_df = pd.DataFrame(recent_patterns)
            display_pat_cols = ["pattern_hash", "client_id", "endpoint", "attack_type", "anomaly_score", "hit_count", "last_seen", "created_at"]
            st.dataframe(pat_df[[c for c in display_pat_cols if c in pat_df.columns]], use_container_width=True, height=350)
        else:
            st.info("Pattern memory store is currently empty. Run `python -m tools.seed_patterns` to populate initial signatures.")

    with tab_compliance:
        st.subheader("State Bank of Pakistan (SBP) Compliance Telemetry")
        col_sla1, col_sla2 = st.columns(2)
        with col_sla1:
            st.markdown("#### **Performance SLA Telemetry**")
            st.markdown("- **SLA Budget Target:** `< 200.0 ms` per API request")
            st.markdown(f"- **Current Average Latency:** `{avg_latency:.2f} ms`")
            st.markdown(f"- **Max Observed Latency:** `{df_raw['latency_ms'].max():.2f} ms`")
            st.markdown(f"- **Min Observed Latency:** `{df_raw['latency_ms'].min():.2f} ms`")
            sla_compliance = (len(df_raw[df_raw["latency_ms"] < 200.0]) / len(df_raw) * 100.0) if len(df_raw) > 0 else 100.0
            st.success(f"✅ **SLA Compliance Rate:** {sla_compliance:.2f}% requests served under 200 ms")

        with col_sla2:
            st.markdown("#### **Zero-Trust Security Controls**")
            st.markdown("- **Inline Reverse Proxy Interception:** Every HTTP request inspected before upstream dispatch")
            st.markdown("- **Payload Body Preservation:** Request body non-destructively forwarded to banking endpoints")
            st.markdown("- **Local Model Execution:** 100% on-premises offline ML inference (Zero cloud leakage)")
            st.markdown("- **Privacy & PII Protection:** Zero customer data or real accounts utilized (100% synthetic)")

    st.divider()
    st.caption("State Bank of Pakistan (SBP) Regulatory Sandbox Cohort 2 | AI-Powered API Security Mesh | Engineered for Pakistan Open Banking")


if __name__ == "__main__" or any("streamlit" in arg for arg in sys.argv):
    render_dashboard()
