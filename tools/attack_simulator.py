"""State Bank of Pakistan (SBP) Sandbox: Attack Simulator Tool.

Dispatches synthetic normal and malicious traffic scenarios against the
AI Security Mesh Gateway (live network or in-process ASGI) and verifies
inline blocking, ensemble risk scores, and SHAP explainability.
"""

import argparse
import asyncio
from dataclasses import dataclass, field
import json
import time
from typing import Any, Dict, List, Optional
import httpx
from backend.config import VerdictEnum, settings
from backend.main import create_app
from data.attack_traffic import (
    generate_api_flooding_attack,
    generate_bola_attack,
    generate_bruteforce_login_attack,
    generate_credential_stuffing_attack,
    generate_parameter_tampering_attack,
    generate_sql_injection_attack,
    generate_token_replay_attack,
)
from data.normal_traffic import SyntheticHTTPRequest, generate_normal_session
from ml.pipeline import register_pipeline_with_interceptor


@dataclass
class ScenarioResult:
    """Summary record for a single executed HTTP request."""

    scenario: str
    method: str
    endpoint: str
    status_code: int
    verdict: str
    anomaly_score: float
    latency_ms: float
    is_blocked: bool
    explanations: List[Dict[str, Any]] = field(default_factory=list)
    client_id: Optional[str] = None


@dataclass
class SimulationReport:
    """Consolidated metrics across all executed scenarios."""

    total_requests: int = 0
    approved_count: int = 0
    flagged_count: int = 0
    blocked_count: int = 0
    attack_requests: int = 0
    attacks_blocked: int = 0
    avg_latency_ms: float = 0.0
    results: List[ScenarioResult] = field(default_factory=list)

    @property
    def detection_rate(self) -> float:
        """Percentage of attack requests that were blocked (HTTP 403)."""
        if self.attack_requests == 0:
            return 100.0
        return (self.attacks_blocked / self.attack_requests) * 100.0


class AttackSimulator:
    """Orchestrates traffic dispatch and evaluation against the Security Mesh."""

    def __init__(
        self,
        target_url: str = "http://127.0.0.1:8080",
        in_process: bool = False,
        timeout: float = 10.0,
    ) -> None:
        self.target_url = target_url.rstrip("/")
        self.in_process = in_process
        self.timeout = timeout
        self.app = create_app() if in_process else None
        if self.in_process:
            register_pipeline_with_interceptor()

    async def _dispatch_request(
        self,
        client: httpx.AsyncClient,
        req: SyntheticHTTPRequest,
        scenario_name: str,
    ) -> ScenarioResult:
        """Send a single request to the gateway and parse security mesh headers."""
        url = f"{self.target_url}{req.endpoint}"
        body_bytes = json.dumps(req.body).encode("utf-8") if req.body else b""

        t0 = time.perf_counter()
        try:
            if req.body:
                response = await client.request(
                    method=req.method,
                    url=url,
                    headers=req.headers,
                    json=req.body,
                )
            else:
                response = await client.request(
                    method=req.method,
                    url=url,
                    headers=req.headers,
                )
            round_trip_ms = (time.perf_counter() - t0) * 1000.0
            status_code = response.status_code

            # Extract mesh telemetry from headers or body
            mesh_verdict = response.headers.get(
                settings.header_verdict,
                response.headers.get("x-mesh-verdict", "UNKNOWN")
            )
            mesh_score_str = response.headers.get("x-mesh-risk-score")
            mesh_latency_str = response.headers.get("x-mesh-latency-ms")

            explanations: List[Dict[str, Any]] = []
            anomaly_score = float(mesh_score_str) if mesh_score_str else 0.0
            latency_ms = float(mesh_latency_str) if mesh_latency_str else round_trip_ms

            if status_code == 403:
                try:
                    data = response.json()
                    mesh_verdict = data.get("verdict", "BLOCK")
                    anomaly_score = data.get("anomaly_score", anomaly_score)
                    explanations = data.get("explanations", [])
                except Exception:
                    pass

            is_blocked = status_code == 403 or mesh_verdict == VerdictEnum.BLOCK.value

            return ScenarioResult(
                scenario=scenario_name,
                method=req.method,
                endpoint=req.endpoint,
                status_code=status_code,
                verdict=mesh_verdict,
                anomaly_score=anomaly_score,
                latency_ms=round(latency_ms, 2),
                is_blocked=is_blocked,
                explanations=explanations,
                client_id=req.client_id,
            )

        except Exception as exc:
            return ScenarioResult(
                scenario=scenario_name,
                method=req.method,
                endpoint=req.endpoint,
                status_code=500,
                verdict="ERROR",
                anomaly_score=1.0,
                latency_ms=0.0,
                is_blocked=False,
                explanations=[{"error": str(exc)}],
                client_id=req.client_id,
            )

    async def run_scenario(self, scenario: str) -> List[ScenarioResult]:
        """Execute selected traffic scenario."""
        scenario_map = {
            "sqli": (lambda: generate_sql_injection_attack(), True),
            "tampering": (lambda: generate_parameter_tampering_attack(), True),
            "bruteforce": (lambda: generate_bruteforce_login_attack(), True),
            "stuffing": (lambda: generate_credential_stuffing_attack(), True),
            "bola": (lambda: generate_bola_attack(), True),
            "flooding": (lambda: generate_api_flooding_attack(), True),
            "replay": (lambda: generate_token_replay_attack(), True),
            "normal": (lambda: generate_normal_session(client_id="tpp_live_legit_01"), False),
        }

        transport = httpx.ASGITransport(app=self.app) if self.in_process else None
        async with httpx.AsyncClient(transport=transport, timeout=self.timeout) as client:
            if scenario == "all":
                results: List[ScenarioResult] = []
                # Execute normal baseline first, then all 7 attack vectors
                scenarios_to_run = ["normal", "sqli", "tampering", "bruteforce", "stuffing", "bola", "flooding", "replay"]
                for sc in scenarios_to_run:
                    gen_fn, _ = scenario_map[sc]
                    reqs = gen_fn()
                    for r in reqs:
                        res = await self._dispatch_request(client, r, sc)
                        results.append(res)
                return results

            elif scenario in scenario_map:
                gen_fn, _ = scenario_map[scenario]
                reqs = gen_fn()
                results = []
                for r in reqs:
                    res = await self._dispatch_request(client, r, scenario)
                    results.append(res)
                return results

            else:
                raise ValueError(f"Unknown scenario '{scenario}'. Choose from: {list(scenario_map.keys()) + ['all']}")

    async def execute_and_report(self, scenario: str = "all") -> SimulationReport:
        """Execute simulation, calculate aggregated metrics, and print formatted report."""
        results = await self.run_scenario(scenario)

        report = SimulationReport()
        report.total_requests = len(results)
        report.results = results

        total_latency = 0.0
        for r in results:
            total_latency += r.latency_ms
            if r.verdict == VerdictEnum.APPROVE.value:
                report.approved_count += 1
            elif r.verdict == VerdictEnum.FLAG.value:
                report.flagged_count += 1
            elif r.verdict == VerdictEnum.BLOCK.value or r.is_blocked:
                report.blocked_count += 1

            if r.scenario != "normal":
                report.attack_requests += 1
                if r.is_blocked:
                    report.attacks_blocked += 1

        report.avg_latency_ms = round(total_latency / len(results), 2) if results else 0.0
        self.print_console_summary(report)
        return report

    def print_console_summary(self, report: SimulationReport) -> None:
        """Display formatted terminal dashboard of simulation results."""
        print("\n" + "=" * 90)
        print("  SBP REGULATORY SANDBOX — AI API SECURITY MESH SIMULATION REPORT")
        print("=" * 90)
        print(f"  Target Gateway : {self.target_url} (in-process: {self.in_process})")
        print(f"  Total Requests : {report.total_requests}")
        print(f"  Approved       : {report.approved_count} (Legitimate API traffic)")
        print(f"  Flagged        : {report.flagged_count} (Suspicious / Under Review)")
        print(f"  Blocked        : {report.blocked_count} (HTTP 403 Forbidden)")
        print(f"  Attack Vectors : {report.attacks_blocked}/{report.attack_requests} Blocked "
              f"({report.detection_rate:.1f}% Detection Rate)")
        print(f"  Avg Latency    : {report.avg_latency_ms:.2f} ms (SLA Target: < 200.00 ms)")
        print("-" * 90)

        # Print top blocked attacks with SHAP explanations
        print(f"{'SCENARIO':<12} | {'METHOD':<6} | {'STATUS':<6} | {'VERDICT':<8} | {'SCORE':<6} | {'LATENCY':<8} | {'TOP SHAP REASON'}")
        print("-" * 90)
        for r in report.results:
            top_reason = ""
            if r.explanations:
                exp = r.explanations[0]
                feat = exp.get("feature", "unknown")
                contrib = exp.get("contribution", 0.0)
                top_reason = f"{feat} (+{contrib:.2f})"
            print(f"{r.scenario:<12} | {r.method:<6} | {r.status_code:<6} | {r.verdict:<8} | {r.anomaly_score:<6.2f} | {r.latency_ms:<5.1f} ms | {top_reason}")
        print("=" * 90 + "\n")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for attack simulator."""
    parser = argparse.ArgumentParser(
        description="State Bank of Pakistan (SBP) Sandbox API Security Mesh Attack Simulator"
    )
    parser.add_argument(
        "-s",
        "--scenario",
        default="all",
        choices=["all", "normal", "sqli", "tampering", "bruteforce", "stuffing", "bola", "flooding", "replay"],
        help="Attack scenario to execute (default: all)",
    )
    parser.add_argument(
        "-t",
        "--target",
        default="http://127.0.0.1:8080",
        help="Target API gateway URL (default: http://127.0.0.1:8080)",
    )
    parser.add_argument(
        "--in-process",
        action="store_true",
        help="Execute directly against in-memory FastAPI app without requiring running server",
    )
    return parser.parse_args()


def main() -> None:
    """CLI Entrypoint."""
    args = parse_args()
    simulator = AttackSimulator(
        target_url=args.target,
        in_process=args.in_process,
    )
    asyncio.run(simulator.execute_and_report(scenario=args.scenario))


if __name__ == "__main__":
    main()

