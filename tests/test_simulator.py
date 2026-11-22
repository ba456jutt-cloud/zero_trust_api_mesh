"""Unit and Integration tests for Attack Simulator CLI Tool."""

import pytest
from backend.config import VerdictEnum
from tools.attack_simulator import AttackSimulator, parse_args


def test_simulator_cli_parser() -> None:
    """Verify CLI parser handles valid arguments and defaults."""
    import sys
    sys_argv_backup = sys.argv
    try:
        sys.argv = ["attack_simulator.py", "--scenario", "sqli", "--in-process"]
        args = parse_args()
        assert args.scenario == "sqli"
        assert args.in_process is True
        assert args.target == "http://127.0.0.1:8080"
    finally:
        sys.argv = sys_argv_backup


@pytest.mark.asyncio
async def test_simulator_normal_scenario_in_process() -> None:
    """Verify normal compliant banking traffic executes cleanly with APPROVE verdicts."""
    simulator = AttackSimulator(in_process=True)
    report = await simulator.execute_and_report(scenario="normal")

    assert report.total_requests == 3
    assert report.approved_count == 3
    assert report.blocked_count == 0
    assert report.avg_latency_ms < 200.0
    for r in report.results:
        assert r.verdict == VerdictEnum.APPROVE.value
        assert r.status_code in [200, 201]
        assert r.anomaly_score < 0.40


@pytest.mark.asyncio
async def test_simulator_sqli_scenario_in_process() -> None:
    """Verify SQL injection attack scenario is intercepted, blocked (403), with SHAP attribution."""
    simulator = AttackSimulator(in_process=True)
    report = await simulator.execute_and_report(scenario="sqli")

    assert report.total_requests >= 1
    assert report.blocked_count >= 1
    assert report.detection_rate == 100.0
    assert report.avg_latency_ms < 200.0

    sqli_result = report.results[0]
    assert sqli_result.status_code == 403
    assert sqli_result.verdict == VerdictEnum.BLOCK.value
    assert sqli_result.anomaly_score >= 0.70
    assert len(sqli_result.explanations) > 0
    assert any(e.get("feature") == "payload_sql_score" for e in sqli_result.explanations)


@pytest.mark.asyncio
async def test_simulator_parameter_tampering_scenario_in_process() -> None:
    """Verify parameter tampering attack scenario is blocked with high anomaly score."""
    simulator = AttackSimulator(in_process=True)
    report = await simulator.execute_and_report(scenario="tampering")

    assert report.total_requests >= 1
    assert report.blocked_count >= 1
    tamper_result = report.results[0]
    assert tamper_result.status_code == 403
    assert tamper_result.verdict == VerdictEnum.BLOCK.value
    assert tamper_result.anomaly_score >= 0.70

