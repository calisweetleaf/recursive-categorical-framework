"""
URSMIF Alpha-1 Stress Benchmark Suite
====================================
Pushes URSMIF toward operational limits while recording stability, resource
usage, and failure thresholds. Generates:
- Detailed terminal output
- Markdown report
- JSON manifest + JSON event log
"""

import sys
import os
import json
import time
import math
import random
import string
import argparse
import tracemalloc
import gc
import psutil
from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

# Rich for colored TUI output
try:
    from rich.console import Console
    from rich.table import Table
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

# Import URSMIF components
try:
    import importlib.util
    spec = importlib.util.spec_from_file_location("ursmif_theory", "ursmif-theory.py")
    ursmif_theory = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ursmif_theory)

    MetaADHDMonitor = ursmif_theory.MetaADHDMonitor
    SystemState = ursmif_theory.SystemState
    ResonanceProfile = ursmif_theory.ResonanceProfile
    AttentionState = ursmif_theory.AttentionState
    ComplexityAnalyzer = ursmif_theory.ComplexityAnalyzer
except Exception as e:
    print(f"ERROR: Failed to import ursmif_theory: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class StressConfig:
    graph_sizes: List[int] = field(default_factory=lambda: [100, 1000, 10000, 100000, 250000, 500000, 1000000])
    recursion_depths: List[int] = field(default_factory=lambda: [1, 3, 5, 10, 20, 50])
    max_kb_entries: int = 200000
    max_outputs: int = 2000
    max_time_s: float = 15.0
    max_memory_mb: float = 512.0
    contradiction_ratios: List[float] = field(default_factory=lambda: [0.0, 0.1, 0.3, 0.6, 0.9])
    entropy_sizes: List[int] = field(default_factory=lambda: [10, 100, 500, 1000, 2500, 5000])
    saturation_iterations: int = 2000
    fuzz_cases: int = 300
    seed: int = 1337


@dataclass
class StressCaseResult:
    test_name: str
    case_id: str
    status: str
    execution_time_ms: float
    memory_peak_mb: float
    memory_delta_mb: float
    cpu_percent: float
    patterns_detected: int
    limit_triggered: Optional[str]
    exception: Optional[str]
    details: Dict[str, Any]
    timestamp: str


@dataclass
class StressManifest:
    run_id: str
    timestamp: str
    python_version: str
    system_info: Dict[str, Any]
    test_configuration: Dict[str, Any]
    results: List[Dict[str, Any]]
    summary: Dict[str, Any]


@dataclass
class OperationResult:
    patterns_detected: int = 0
    details: Dict[str, Any] = field(default_factory=dict)
    limit_triggered: Optional[str] = None


# =============================================================================
# TUI Output
# =============================================================================

class StressTUI:
    def __init__(self):
        self.use_rich = RICH_AVAILABLE
        self.console = Console() if self.use_rich else None
        self.theme_primary = "bold bright_blue"
        self.theme_success = "bold green"
        self.theme_warning = "bold yellow"
        self.theme_error = "bold red"

    def _print(self, text: str):
        if self.use_rich:
            self.console.print(text)
        else:
            print(text)

    def print_header(self, title: str):
        if self.use_rich:
            self.console.print(f"[{self.theme_primary}]{'=' * 78}[/{self.theme_primary}]")
            self.console.print(f"[{self.theme_primary}]{title}[/{self.theme_primary}]")
            self.console.print(f"[{self.theme_primary}]{'=' * 78}[/{self.theme_primary}]")
        else:
            print("=" * 78)
            print(title)
            print("=" * 78)

    def print_subheader(self, title: str):
        if self.use_rich:
            self.console.print(f"[{self.theme_primary}]{title}[/{self.theme_primary}]")
        else:
            print(title)

    def print_metric(self, label: str, value: str):
        self._print(f"  {label}: {value}")

    def print_success(self, message: str):
        if self.use_rich:
            self.console.print(f"[{self.theme_success}]OK {message}[/{self.theme_success}]")
        else:
            print(f"OK {message}")

    def print_warning(self, message: str):
        if self.use_rich:
            self.console.print(f"[{self.theme_warning}]WARN {message}[/{self.theme_warning}]")
        else:
            print(f"WARN {message}")

    def print_error(self, message: str):
        if self.use_rich:
            self.console.print(f"[{self.theme_error}]FAIL {message}[/{self.theme_error}]")
        else:
            print(f"FAIL {message}")

    def print_table(self, title: str, rows: List[Dict[str, Any]], columns: List[str]):
        if not self.use_rich:
            print(title)
            for row in rows:
                print(" | ".join(str(row.get(col, "")) for col in columns))
            return

        table = Table(title=title, box=box.SIMPLE)
        for col in columns:
            table.add_column(col)
        for row in rows:
            table.add_row(*[str(row.get(col, "")) for col in columns])
        self.console.print(table)


# =============================================================================
# Stress Benchmark Suite
# =============================================================================

class URSMIFStressBenchmark:
    def __init__(self, config: StressConfig):
        self.config = config
        self.tui = StressTUI()
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path("results")
        self.output_dir.mkdir(exist_ok=True)
        self.process = psutil.Process(os.getpid())
        self.results: List[StressCaseResult] = []
        self.events: List[Dict[str, Any]] = []
        self.complexity_analyzer = ComplexityAnalyzer()

    def _new_monitor(self) -> MetaADHDMonitor:
        return MetaADHDMonitor()

    def _log_event(self, event: str, details: Dict[str, Any]):
        self.events.append({
            "timestamp": datetime.now().isoformat(),
            "event": event,
            "details": details
        })

    def _measure_operation(self, operation_fn, deadline: float) -> Dict[str, Any]:
        self.process.cpu_percent(None)
        mem_before = self.process.memory_info().rss
        tracemalloc.start()
        start = time.perf_counter()
        outcome = operation_fn(deadline)
        elapsed = time.perf_counter() - start
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        mem_after = self.process.memory_info().rss
        cpu_percent = self.process.cpu_percent(None)

        return {
            "outcome": outcome,
            "execution_time_ms": round(elapsed * 1000, 4),
            "memory_peak_mb": round(peak / (1024 ** 2), 4),
            "memory_delta_mb": round((mem_after - mem_before) / (1024 ** 2), 4),
            "cpu_percent": round(cpu_percent, 2)
        }

    def _build_kb(self, kb_entries: int, contradiction_ratio: float = 0.0) -> set:
        if kb_entries <= 0:
            return set()
        contradictions = int(kb_entries * contradiction_ratio / 2)
        kb = set()
        for i in range(contradictions):
            prop = f"c{i}"
            kb.add((prop, True))
            kb.add((prop, False))
        remaining = kb_entries - (contradictions * 2)
        for i in range(remaining):
            kb.add((f"p{i}", True))
        return kb

    def _random_outputs(self, count: int, token_len: int = 8) -> List[str]:
        outputs = []
        alphabet = string.ascii_letters + string.digits
        for _ in range(max(1, count)):
            token = "".join(random.choices(alphabet, k=token_len))
            outputs.append(token)
        return outputs

    def _record_result(self, test_name: str, case_id: str, metrics: Dict[str, Any],
                       outcome: OperationResult, exception: Optional[str] = None):
        status = "ok"
        if exception:
            status = "error"
        elif outcome.limit_triggered:
            status = "limit"

        result = StressCaseResult(
            test_name=test_name,
            case_id=case_id,
            status=status,
            execution_time_ms=metrics["execution_time_ms"],
            memory_peak_mb=metrics["memory_peak_mb"],
            memory_delta_mb=metrics["memory_delta_mb"],
            cpu_percent=metrics["cpu_percent"],
            patterns_detected=outcome.patterns_detected,
            limit_triggered=outcome.limit_triggered,
            exception=exception,
            details=outcome.details,
            timestamp=datetime.now().isoformat()
        )
        self.results.append(result)

    def test_scaling_saturation(self):
        test_name = "scaling_saturation"
        self.tui.print_header("TEST 1: Scaling Saturation")
        for graph_size in self.config.graph_sizes:
            for depth in self.config.recursion_depths:
                case_id = f"n{graph_size}_d{depth}"
                self.tui.print_subheader(f"Case {case_id}")
                kb_entries = min(graph_size, self.config.max_kb_entries)
                outputs_count = min(self.config.max_outputs, max(5, graph_size // 1000))
                deadline = time.perf_counter() + self.config.max_time_s
                monitor = self._new_monitor()

                def op(deadline_ts: float) -> OperationResult:
                    patterns_total = 0
                    kb = self._build_kb(kb_entries, contradiction_ratio=0.1)
                    for i in range(depth):
                        if time.perf_counter() > deadline_ts:
                            return OperationResult(
                                patterns_detected=patterns_total,
                                details={"graph_size": graph_size, "recursion_depth": depth},
                                limit_triggered="time_limit"
                            )
                        state = SystemState(
                            outputs=[f"out_{j}" for j in range(outputs_count)],
                            knowledge_base=kb,
                            self_references=random.randint(0, min(50, kb_entries // 100 + 1)),
                            timestamp=time.time(),
                            recursion_depth=i
                        )
                        patterns = monitor.monitor(state)
                        patterns_total += len(patterns)
                    return OperationResult(
                        patterns_detected=patterns_total,
                        details={
                            "graph_size": graph_size,
                            "recursion_depth": depth,
                            "kb_entries": kb_entries,
                            "outputs_count": outputs_count
                        }
                    )

                self._log_event("case_start", {"test": test_name, "case_id": case_id})
                exception = None
                try:
                    metrics = self._measure_operation(op, deadline)
                    outcome = metrics["outcome"]
                    if metrics["memory_peak_mb"] > self.config.max_memory_mb:
                        outcome.limit_triggered = outcome.limit_triggered or "memory_limit"
                except Exception as exc:
                    metrics = {
                        "execution_time_ms": 0.0,
                        "memory_peak_mb": 0.0,
                        "memory_delta_mb": 0.0,
                        "cpu_percent": 0.0,
                        "outcome": OperationResult()
                    }
                    outcome = metrics["outcome"]
                    exception = f"{type(exc).__name__}: {exc}"
                self._record_result(test_name, case_id, metrics, outcome, exception)
                if exception:
                    self.tui.print_error(f"{case_id} -> {exception}")
                elif outcome.limit_triggered:
                    self.tui.print_warning(f"{case_id} -> limit {outcome.limit_triggered}")
                else:
                    self.tui.print_success(f"{case_id} -> {metrics['execution_time_ms']:.2f} ms")
                self._log_event("case_end", {
                    "test": test_name,
                    "case_id": case_id,
                    "status": "error" if exception else "limit" if outcome.limit_triggered else "ok"
                })
                gc.collect()

    def test_recursion_storm(self):
        test_name = "recursion_storm"
        self.tui.print_header("TEST 2: Recursion Storm")
        depths = [5, 10, 25, 50, 75, 100]
        for depth in depths:
            case_id = f"depth_{depth}"
            self.tui.print_subheader(f"Case {case_id}")
            deadline = time.perf_counter() + self.config.max_time_s
            monitor = self._new_monitor()

            def op(deadline_ts: float) -> OperationResult:
                patterns_total = 0
                for i in range(depth):
                    if time.perf_counter() > deadline_ts:
                        return OperationResult(
                            patterns_detected=patterns_total,
                            details={"depth": depth, "iteration": i},
                            limit_triggered="time_limit"
                        )
                    kb = self._build_kb(2000, contradiction_ratio=0.2)
                    state = SystemState(
                        outputs=[f"storm_{i}_{j}" for j in range(20)],
                        knowledge_base=kb,
                        self_references=random.randint(10, 40),
                        timestamp=time.time(),
                        attention_state=AttentionState.SUPERPOSITION,
                        recursion_depth=i
                    )
                    patterns, meta_state = monitor.meta_monitor(state)
                    patterns_total += len(patterns)
                return OperationResult(
                    patterns_detected=patterns_total,
                    details={
                        "depth": depth,
                        "recursion_stack": len(monitor.recursion_stack),
                        "meta_stability_samples": len(monitor.meta_stability_history)
                    }
                )

            self._log_event("case_start", {"test": test_name, "case_id": case_id})
            exception = None
            try:
                metrics = self._measure_operation(op, deadline)
                outcome = metrics["outcome"]
                if metrics["memory_peak_mb"] > self.config.max_memory_mb:
                    outcome.limit_triggered = outcome.limit_triggered or "memory_limit"
            except Exception as exc:
                metrics = {
                    "execution_time_ms": 0.0,
                    "memory_peak_mb": 0.0,
                    "memory_delta_mb": 0.0,
                    "cpu_percent": 0.0,
                    "outcome": OperationResult()
                }
                outcome = metrics["outcome"]
                exception = f"{type(exc).__name__}: {exc}"
            self._record_result(test_name, case_id, metrics, outcome, exception)
            if exception:
                self.tui.print_error(f"{case_id} -> {exception}")
            elif outcome.limit_triggered:
                self.tui.print_warning(f"{case_id} -> limit {outcome.limit_triggered}")
            else:
                self.tui.print_success(f"{case_id} -> {metrics['execution_time_ms']:.2f} ms")
            self._log_event("case_end", {
                "test": test_name,
                "case_id": case_id,
                "status": "error" if exception else "limit" if outcome.limit_triggered else "ok"
            })
            gc.collect()

    def test_contradiction_cascade(self):
        test_name = "contradiction_cascade"
        self.tui.print_header("TEST 3: Contradiction Cascade")
        sizes = [1000, 5000, 20000, 50000]
        for kb_size in sizes:
            for ratio in self.config.contradiction_ratios:
                case_id = f"kb{kb_size}_r{int(ratio*100)}"
                self.tui.print_subheader(f"Case {case_id}")
                deadline = time.perf_counter() + self.config.max_time_s
                monitor = self._new_monitor()

                def op(deadline_ts: float) -> OperationResult:
                    if time.perf_counter() > deadline_ts:
                        return OperationResult(limit_triggered="time_limit")
                    kb_entries = min(kb_size, self.config.max_kb_entries)
                    kb = self._build_kb(kb_entries, contradiction_ratio=ratio)
                    state = SystemState(
                        outputs=[f"conflict_{i}" for i in range(30)],
                        knowledge_base=kb,
                        self_references=random.randint(5, 20),
                        timestamp=time.time(),
                        recursion_depth=1
                    )
                    patterns = monitor.monitor(state)
                    contradiction_count = sum(1 for p in patterns if p.pattern_type == "contradiction")
                    return OperationResult(
                        patterns_detected=len(patterns),
                        details={
                            "kb_entries": kb_entries,
                            "contradiction_ratio": ratio,
                            "contradictions_detected": contradiction_count
                        }
                    )

                self._log_event("case_start", {"test": test_name, "case_id": case_id})
                exception = None
                try:
                    metrics = self._measure_operation(op, deadline)
                    outcome = metrics["outcome"]
                    if metrics["memory_peak_mb"] > self.config.max_memory_mb:
                        outcome.limit_triggered = outcome.limit_triggered or "memory_limit"
                except Exception as exc:
                    metrics = {
                        "execution_time_ms": 0.0,
                        "memory_peak_mb": 0.0,
                        "memory_delta_mb": 0.0,
                        "cpu_percent": 0.0,
                        "outcome": OperationResult()
                    }
                    outcome = metrics["outcome"]
                    exception = f"{type(exc).__name__}: {exc}"
                self._record_result(test_name, case_id, metrics, outcome, exception)
                if exception:
                    self.tui.print_error(f"{case_id} -> {exception}")
                elif outcome.limit_triggered:
                    self.tui.print_warning(f"{case_id} -> limit {outcome.limit_triggered}")
                else:
                    self.tui.print_success(f"{case_id} -> {metrics['execution_time_ms']:.2f} ms")
                self._log_event("case_end", {
                    "test": test_name,
                    "case_id": case_id,
                    "status": "error" if exception else "limit" if outcome.limit_triggered else "ok"
                })
                gc.collect()

    def test_entropy_flood(self):
        test_name = "entropy_flood"
        self.tui.print_header("TEST 4: Entropy Flood")
        for count in self.config.entropy_sizes:
            case_id = f"entropy_{count}"
            self.tui.print_subheader(f"Case {case_id}")
            deadline = time.perf_counter() + self.config.max_time_s
            monitor = self._new_monitor()

            def op(deadline_ts: float) -> OperationResult:
                if time.perf_counter() > deadline_ts:
                    return OperationResult(limit_triggered="time_limit")
                outputs = self._random_outputs(count, token_len=12)
                state = SystemState(
                    outputs=outputs,
                    knowledge_base=set(),
                    self_references=0,
                    timestamp=time.time(),
                    resonance_profile=ResonanceProfile(
                        novelty=1.0, interest=0.2, challenge=0.3, urgency=0.1, emotional_salience=0.1
                    )
                )
                patterns = monitor.monitor(state)
                return OperationResult(
                    patterns_detected=len(patterns),
                    details={"output_count": count, "entropy": round(state.entropy, 4)}
                )

            self._log_event("case_start", {"test": test_name, "case_id": case_id})
            exception = None
            try:
                metrics = self._measure_operation(op, deadline)
                outcome = metrics["outcome"]
                if metrics["memory_peak_mb"] > self.config.max_memory_mb:
                    outcome.limit_triggered = outcome.limit_triggered or "memory_limit"
            except Exception as exc:
                metrics = {
                    "execution_time_ms": 0.0,
                    "memory_peak_mb": 0.0,
                    "memory_delta_mb": 0.0,
                    "cpu_percent": 0.0,
                    "outcome": OperationResult()
                }
                outcome = metrics["outcome"]
                exception = f"{type(exc).__name__}: {exc}"
            self._record_result(test_name, case_id, metrics, outcome, exception)
            if exception:
                self.tui.print_error(f"{case_id} -> {exception}")
            elif outcome.limit_triggered:
                self.tui.print_warning(f"{case_id} -> limit {outcome.limit_triggered}")
            else:
                entropy_val = outcome.details.get("entropy", "n/a")
                self.tui.print_success(f"{case_id} -> entropy {entropy_val}")
            self._log_event("case_end", {
                "test": test_name,
                "case_id": case_id,
                "status": "error" if exception else "limit" if outcome.limit_triggered else "ok"
            })
            gc.collect()

    def test_pattern_saturation(self):
        test_name = "pattern_saturation"
        self.tui.print_header("TEST 5: Pattern Saturation")
        case_id = f"iterations_{self.config.saturation_iterations}"
        self.tui.print_subheader(f"Case {case_id}")
        deadline = time.perf_counter() + self.config.max_time_s
        monitor = self._new_monitor()

        def op(deadline_ts: float) -> OperationResult:
            patterns_total = 0
            state = SystemState(
                outputs=["repeat"] * 10,
                knowledge_base={("loop", True)},
                self_references=10,
                timestamp=time.time(),
                recursion_depth=1
            )
            for i in range(self.config.saturation_iterations):
                if time.perf_counter() > deadline_ts:
                    return OperationResult(
                        patterns_detected=patterns_total,
                        details={"iterations": i, "pattern_cache": len(monitor.detected_patterns)},
                        limit_triggered="time_limit"
                    )
                patterns = monitor.monitor(state)
                patterns_total += len(patterns)
            return OperationResult(
                patterns_detected=patterns_total,
                details={
                    "iterations": self.config.saturation_iterations,
                    "pattern_cache": len(monitor.detected_patterns)
                }
            )

        self._log_event("case_start", {"test": test_name, "case_id": case_id})
        exception = None
        try:
            metrics = self._measure_operation(op, deadline)
            outcome = metrics["outcome"]
            if metrics["memory_peak_mb"] > self.config.max_memory_mb:
                outcome.limit_triggered = outcome.limit_triggered or "memory_limit"
        except Exception as exc:
            metrics = {
                "execution_time_ms": 0.0,
                "memory_peak_mb": 0.0,
                "memory_delta_mb": 0.0,
                "cpu_percent": 0.0,
                "outcome": OperationResult()
            }
            outcome = metrics["outcome"]
            exception = f"{type(exc).__name__}: {exc}"
        self._record_result(test_name, case_id, metrics, outcome, exception)
        if exception:
            self.tui.print_error(f"{case_id} -> {exception}")
        elif outcome.limit_triggered:
            self.tui.print_warning(f"{case_id} -> limit {outcome.limit_triggered}")
        else:
            self.tui.print_success(
                f"{case_id} -> cache {outcome.details.get('pattern_cache', 0)}"
            )
        self._log_event("case_end", {
            "test": test_name,
            "case_id": case_id,
            "status": "error" if exception else "limit" if outcome.limit_triggered else "ok"
        })
        gc.collect()

    def test_fuzzing(self):
        test_name = "state_fuzzing"
        self.tui.print_header("TEST 6: State Fuzzing")
        monitor = self._new_monitor()
        failures = 0
        deadline = time.perf_counter() + self.config.max_time_s

        def op(deadline_ts: float) -> OperationResult:
            nonlocal failures
            patterns_total = 0
            for i in range(self.config.fuzz_cases):
                if time.perf_counter() > deadline_ts:
                    return OperationResult(
                        patterns_detected=patterns_total,
                        details={"cases": i, "failures": failures},
                        limit_triggered="time_limit"
                    )
                kb_entries = random.randint(0, min(self.config.max_kb_entries, 5000))
                outputs_count = random.randint(1, 50)
                kb = self._build_kb(kb_entries, contradiction_ratio=random.choice(self.config.contradiction_ratios))
                outputs = [f"fuzz_{random.randint(0, 9999)}" for _ in range(outputs_count)]
                state = SystemState(
                    outputs=outputs,
                    knowledge_base=kb,
                    self_references=random.randint(0, 100),
                    timestamp=time.time(),
                    recursion_depth=random.randint(0, 15)
                )
                try:
                    patterns = monitor.monitor(state)
                    patterns_total += len(patterns)
                except Exception:
                    failures += 1
            return OperationResult(
                patterns_detected=patterns_total,
                details={"cases": self.config.fuzz_cases, "failures": failures}
            )

        case_id = f"fuzz_{self.config.fuzz_cases}"
        self._log_event("case_start", {"test": test_name, "case_id": case_id})
        exception = None
        try:
            metrics = self._measure_operation(op, deadline)
            outcome = metrics["outcome"]
            if metrics["memory_peak_mb"] > self.config.max_memory_mb:
                outcome.limit_triggered = outcome.limit_triggered or "memory_limit"
        except Exception as exc:
            metrics = {
                "execution_time_ms": 0.0,
                "memory_peak_mb": 0.0,
                "memory_delta_mb": 0.0,
                "cpu_percent": 0.0,
                "outcome": OperationResult()
            }
            outcome = metrics["outcome"]
            exception = f"{type(exc).__name__}: {exc}"
        self._record_result(test_name, case_id, metrics, outcome, exception)
        if exception:
            self.tui.print_error(f"{case_id} -> {exception}")
        elif outcome.limit_triggered:
            self.tui.print_warning(f"{case_id} -> limit {outcome.limit_triggered}")
        else:
            self.tui.print_success(
                f"{case_id} -> failures {outcome.details.get('failures', 0)}"
            )
        self._log_event("case_end", {
            "test": test_name,
            "case_id": case_id,
            "status": "error" if exception else "limit" if outcome.limit_triggered else "ok"
        })
        gc.collect()

    def _summarize(self) -> Dict[str, Any]:
        summary = {
            "total_cases": len(self.results),
            "ok_cases": 0,
            "limit_cases": 0,
            "error_cases": 0,
            "max_execution_ms": 0.0,
            "max_memory_mb": 0.0
        }
        for result in self.results:
            if result.status == "ok":
                summary["ok_cases"] += 1
            elif result.status == "limit":
                summary["limit_cases"] += 1
            else:
                summary["error_cases"] += 1
            summary["max_execution_ms"] = max(summary["max_execution_ms"], result.execution_time_ms)
            summary["max_memory_mb"] = max(summary["max_memory_mb"], result.memory_peak_mb)
        return summary

    def _report_sections(self) -> Dict[str, List[StressCaseResult]]:
        grouped: Dict[str, List[StressCaseResult]] = {}
        for result in self.results:
            grouped.setdefault(result.test_name, []).append(result)
        return grouped

    def _generate_markdown_report(self, manifest: StressManifest) -> str:
        report = [
            "# URSMIF Stress Benchmark Report",
            "",
            f"Run ID: `{manifest.run_id}`",
            f"Timestamp: {manifest.timestamp}",
            f"Python: {manifest.python_version.split()[0]}",
            "",
            "## Summary",
            f"- Total cases: {manifest.summary['total_cases']}",
            f"- OK: {manifest.summary['ok_cases']}",
            f"- Limits triggered: {manifest.summary['limit_cases']}",
            f"- Errors: {manifest.summary['error_cases']}",
            f"- Max execution time (ms): {manifest.summary['max_execution_ms']:.4f}",
            f"- Max memory peak (MB): {manifest.summary['max_memory_mb']:.4f}",
            "",
            "## System Info",
            f"- Platform: {manifest.system_info['platform']}",
            f"- CPU cores: {manifest.system_info['cpu_count']}",
            f"- Memory total GB: {manifest.system_info['memory_total_gb']}",
            ""
        ]
        for test_name, results in self._report_sections().items():
            report.append(f"## {test_name.replace('_', ' ').title()}")
            report.append("")
            report.append("| Case | Status | Time (ms) | Mem Peak (MB) | CPU % | Patterns | Limit |")
            report.append("|------|--------|-----------|---------------|-------|----------|-------|")
            for result in results:
                report.append(
                    f"| {result.case_id} | {result.status} | {result.execution_time_ms:.4f} | "
                    f"{result.memory_peak_mb:.4f} | {result.cpu_percent:.2f} | "
                    f"{result.patterns_detected} | {result.limit_triggered or ''} |"
                )
            report.append("")
        return "\n".join(report)

    def write_outputs(self):
        manifest = StressManifest(
            run_id=self.run_id,
            timestamp=datetime.now().isoformat(),
            python_version=sys.version,
            system_info={
                "platform": sys.platform,
                "cpu_count": psutil.cpu_count(),
                "memory_total_gb": round(psutil.virtual_memory().total / 1024 ** 3, 2),
                "memory_available_gb": round(psutil.virtual_memory().available / 1024 ** 3, 2)
            },
            test_configuration=asdict(self.config),
            results=[asdict(r) for r in self.results],
            summary=self._summarize()
        )

        manifest_path = self.output_dir / f"stress_manifest_{self.run_id}.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(asdict(manifest), f, indent=2)

        report_path = self.output_dir / f"stress_report_{self.run_id}.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(self._generate_markdown_report(manifest))

        log_path = self.output_dir / f"stress_event_log_{self.run_id}.json"
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(self.events, f, indent=2)

        self.tui.print_success(f"JSON manifest: {manifest_path}")
        self.tui.print_success(f"Markdown report: {report_path}")
        self.tui.print_success(f"Event log: {log_path}")

    def run(self):
        random.seed(self.config.seed)
        self.tui.print_header("URSMIF Alpha-1 Stress Benchmark Suite")
        self.tui.print_metric("Run ID", self.run_id)
        self.tui.print_metric("Timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        self.tui.print_metric("Max KB", str(self.config.max_kb_entries))
        self.tui.print_metric("Max Time (s)", str(self.config.max_time_s))
        self.tui.print_metric("Max Memory (MB)", str(self.config.max_memory_mb))

        self.test_scaling_saturation()
        self.test_recursion_storm()
        self.test_contradiction_cascade()
        self.test_entropy_flood()
        self.test_pattern_saturation()
        self.test_fuzzing()

        self.tui.print_header("Generating Output Files")
        self.write_outputs()
        self.tui.print_header("Stress Benchmark Complete")
        self.tui.print_success(f"Total Cases: {len(self.results)}")


def _parse_args() -> StressConfig:
    parser = argparse.ArgumentParser(description="URSMIF Stress Benchmark")
    parser.add_argument("--max-kb", type=int, default=200000, help="Max KB entries to allocate")
    parser.add_argument("--max-outputs", type=int, default=2000, help="Max outputs per state")
    parser.add_argument("--max-time-s", type=float, default=15.0, help="Max time per case")
    parser.add_argument("--max-memory-mb", type=float, default=512.0, help="Max peak memory per case")
    parser.add_argument("--fuzz-cases", type=int, default=300, help="Number of fuzz cases")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed")
    args = parser.parse_args()

    if args.max_kb < 0:
        raise ValueError("--max-kb must be non-negative")
    if args.max_outputs < 1:
        raise ValueError("--max-outputs must be >= 1")
    if args.max_time_s <= 0:
        raise ValueError("--max-time-s must be > 0")
    if args.max_memory_mb <= 0:
        raise ValueError("--max-memory-mb must be > 0")
    if args.fuzz_cases < 0:
        raise ValueError("--fuzz-cases must be non-negative")

    config = StressConfig(
        max_kb_entries=args.max_kb,
        max_outputs=args.max_outputs,
        max_time_s=args.max_time_s,
        max_memory_mb=args.max_memory_mb,
        fuzz_cases=args.fuzz_cases,
        seed=args.seed
    )
    return config


if __name__ == "__main__":
    try:
        config = _parse_args()
        suite = URSMIFStressBenchmark(config)
        suite.run()
    except Exception as exc:
        print(f"ERROR: Stress benchmark failed: {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
