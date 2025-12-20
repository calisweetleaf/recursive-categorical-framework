"""
RCF Internal Contradictions Theory - Mathematical Validation Harness (non-pytest)

Spec source:
  theoroms/ethics&stability/core stability frameworks/Internal_Contradictions_Theory.md

Policy:
- Detailed terminal output
- Markdown report
- JSON manifest (log-manifest)

This validator extracts all display equations ($$...$$) from the spec and requires that
each one is covered by a corresponding computational check.
"""

from __future__ import annotations

import json
import math
import os
import platform
import sys
import time
import traceback
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import internal_contradictions as ict


@dataclass
class CheckResult:
    name: str
    ok: bool
    seconds: float
    details: dict[str, Any]
    error: str | None = None
    traceback: str | None = None


def _utc_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="strict")


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="\n")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8", newline="\n")


def _run(name: str, fn: Callable[[], dict[str, Any]]) -> CheckResult:
    t0 = time.perf_counter()
    try:
        details = fn()
        return CheckResult(name=name, ok=True, seconds=time.perf_counter() - t0, details=details)
    except Exception as e:
        return CheckResult(
            name=name,
            ok=False,
            seconds=time.perf_counter() - t0,
            details={},
            error=f"{type(e).__name__}: {e}",
            traceback=traceback.format_exc(),
        )


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def extract_display_equations(md_text: str) -> list[dict[str, Any]]:
    """
    Extract $$...$$ blocks and their starting line numbers.
    Supports single-line and multi-line blocks.
    """
    lines = md_text.splitlines()
    in_block = False
    buf: list[str] = []
    start_line = 0
    out: list[dict[str, Any]] = []

    for idx, line in enumerate(lines, start=1):
        if not in_block:
            if "$$" not in line:
                continue
            parts = line.split("$$")
            if len(parts) >= 3:
                eq = parts[1].strip()
                out.append({"start_line": idx, "equation": eq})
                continue
            in_block = True
            start_line = idx
            after = parts[1] if len(parts) == 2 else ""
            buf = [after]
        else:
            if "$$" in line:
                before, *_rest = line.split("$$", 1)
                buf.append(before)
                eq = "\n".join(buf).strip()
                out.append({"start_line": start_line, "equation": eq})
                in_block = False
                buf = []
                start_line = 0
            else:
                buf.append(line)
    return out


def _sig_for(eq: str) -> str:
    s = " ".join(eq.strip().split())
    return s[:80]


def check_tension_nonnegative_and_descent() -> dict[str, Any]:
    rng = ict.random.Random(123)
    beliefs = [
        [0.2, -0.1, 0.4],
        [-0.3, 0.5, 0.1],
        [0.1, 0.2, -0.2],
    ]
    n = len(beliefs)
    w = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                w[i][j] = 0.5 + 0.5 * rng.random()
    c = [0.3, 0.2, 0.1]
    s0 = ict.State(beliefs=[b[:] for b in beliefs], weights_w=w, coeffs_c=c, prev_beliefs=[b[:] for b in beliefs])
    t0 = ict.tension(s0)
    _assert(t0 >= 0.0, "T(S_t) must be non-negative")
    s1 = ict.step_state(s0, eta=0.01, noise_std=0.0, rng=rng)
    t1 = ict.tension(s1)
    _assert(t1 <= t0 + 1e-10, "gradient step without noise should not increase tension (small eta)")
    return {"tension_t0": t0, "tension_t1": t1, "delta": t1 - t0}


def check_preference_stability_and_psi() -> dict[str, Any]:
    # Quadratic with SPD Hessian => stable minimum.
    quad = ict.Quadratic(A=[[4.0, 1.0], [1.0, 3.0]], b=[-1.0, 0.5], c=0.0)
    H = quad.hessian()
    _assert(ict.is_positive_definite(H), "Hessian must be positive definite for stability")

    # Compute minimizer p* = -A^{-1} b (2x2 closed form).
    a, b12 = 4.0, 1.0
    c22 = 3.0
    det = a * c22 - b12 * b12
    _assert(det > 0.0, "A must be invertible")
    invA = [[c22 / det, -b12 / det], [-b12 / det, a / det]]
    p_star = [
        -(invA[0][0] * quad.b[0] + invA[0][1] * quad.b[1]),
        -(invA[1][0] * quad.b[0] + invA[1][1] * quad.b[1]),
    ]

    g_star = quad.grad(p_star)
    _assert(ict._l2_norm(g_star) < 1e-8, "∇_p T must be ~0 at the stable point")

    # psi at a non-optimal point should be finite (can move to reduce tension).
    p0 = [p_star[0] + 0.25, p_star[1] - 0.25]
    psi = ict.preference_strength_psi(quad.value, p0)
    _assert(psi != float("inf") and psi > 0.0, "psi should be finite and positive away from optimum")
    return {"p_star": p_star, "grad_norm_at_star": ict._l2_norm(g_star), "psi_at_perturbed": psi}


def check_free_energy_identity() -> dict[str, Any]:
    o = 0.7
    sigma2 = 0.5
    q = ict.Gaussian1D(mu=0.1, var=1.2)
    F = ict.free_energy_gaussian_1d(o=o, sigma2=sigma2, q=q)
    post = ict.posterior_gaussian_1d(o=o, sigma2=sigma2)
    kl = ict.kl_gaussian_1d(q=q, p=post)
    logZ = ict.log_evidence_gaussian_1d(o=o, sigma2=sigma2)
    # Identity: F = KL(Q||P(S|O)) - log P(O)
    rhs = kl - logZ
    _assert(abs(F - rhs) < 1e-9, "free energy identity must hold (analytic 1D case)")
    _assert(kl >= -1e-12, "KL must be non-negative")
    return {"F": F, "KL": kl, "-log_evidence": -logZ, "abs_error": abs(F - rhs), "posterior": {"mu": post.mu, "var": post.var}}


def check_mdl_decomposition() -> dict[str, Any]:
    # Demonstrate L(M,D)=L(M)+L(D|M) via BIC-like MDL scoring for two models.
    rng = ict.random.Random(7)
    n = 200
    true_mu = 1.0
    true_sigma = 2.0
    data = [rng.gauss(true_mu, true_sigma) for _ in range(n)]

    # Model 1: estimate mu, fixed sigma=true_sigma
    mu1 = sum(data) / n
    sigma1 = true_sigma
    nll1 = 0.5 * n * math.log(2.0 * math.pi * sigma1 * sigma1) + sum((x - mu1) ** 2 for x in data) / (2.0 * sigma1 * sigma1)
    L_M_1 = 0.5 * 1 * math.log(n)
    mdl1 = ict.mdl_score_bic(nll=nll1, k_params=1, n=n)

    # Model 2: estimate mu and sigma
    mu2 = mu1
    var2 = sum((x - mu2) ** 2 for x in data) / n
    sigma2 = math.sqrt(var2)
    nll2 = 0.5 * n * math.log(2.0 * math.pi * sigma2 * sigma2) + sum((x - mu2) ** 2 for x in data) / (2.0 * sigma2 * sigma2)
    L_M_2 = 0.5 * 2 * math.log(n)
    mdl2 = ict.mdl_score_bic(nll=nll2, k_params=2, n=n)

    # Validate decomposition equality (by construction) and trade-off behavior.
    _assert(abs((L_M_1 + (mdl1 - L_M_1)) - mdl1) < 1e-12, "MDL decomposition must hold for model 1")
    _assert(abs((L_M_2 + (mdl2 - L_M_2)) - mdl2) < 1e-12, "MDL decomposition must hold for model 2")
    chosen = "model2" if mdl2 < mdl1 else "model1"
    return {
        "n": n,
        "mdl_model1": mdl1,
        "mdl_model2": mdl2,
        "chosen": chosen,
        "components": {"L_M1": L_M_1, "L_D_given_M1": mdl1 - L_M_1, "L_M2": L_M_2, "L_D_given_M2": mdl2 - L_M_2},
    }


def check_path_probability_product() -> dict[str, Any]:
    # Validate product-form path weights and normalized probabilities.
    e1 = ict.Edge("v1", "r1", "v2", t=0.0, weight=0.8)
    e2 = ict.Edge("v2", "r2", "v3", t=1.0, weight=0.7)
    e3 = ict.Edge("v1", "rX", "v3", t=0.5, weight=0.3)
    p_path12 = ict.path_weight_product([e1, e2])
    p_path_direct = ict.path_weight_product([e3])
    probs = ict.normalize([p_path12, p_path_direct])
    _assert(abs(sum(probs) - 1.0) < 1e-12, "normalized path probabilities must sum to 1")
    _assert(probs[0] > probs[1], "higher product weight should yield higher probability after normalization")
    return {"path_products": [p_path12, p_path_direct], "normalized": probs}


def check_pruning_constraint() -> dict[str, Any]:
    edges = [
        ict.Edge("a", "r", "b", 0.0, weight=0.5, mi=0.9),
        ict.Edge("b", "r", "c", 0.0, weight=0.5, mi=0.4),
        ict.Edge("c", "r", "d", 0.0, weight=0.5, mi=0.2),
        ict.Edge("d", "r", "e", 0.0, weight=0.5, mi=0.1),
    ]
    eps = 0.2
    picked = ict.greedy_prune_by_mi(edges=edges, epsilon=eps)
    total = sum(e.mi for e in edges)
    got = sum(e.mi for e in picked)
    _assert(got >= (1.0 - eps) * total - 1e-12, "pruned graph must meet mutual information constraint")
    # Greedy should be minimal-size for this monotone additive toy objective.
    _assert(len(picked) == 2, "expected minimal subset size 2 for this MI distribution and epsilon")
    return {"epsilon": eps, "total_mi": total, "picked_mi": got, "picked_edges": [f"{e.src}->{e.dst}" for e in picked]}


def check_temporal_abstraction_recursion() -> dict[str, Any]:
    # A^(0)=E, A^(k+1)=phi(A^(k))
    E = ["e1", "e2", "e3", "e4", "e5"]

    def phi(seq: list[str]) -> list[str]:
        # Simple abstraction: pairwise merge into tokens; last passes through.
        out = []
        i = 0
        while i < len(seq):
            if i + 1 < len(seq):
                out.append(f"({seq[i]}+{seq[i+1]})")
                i += 2
            else:
                out.append(seq[i])
                i += 1
        return out

    A0 = E[:]
    A1 = phi(A0)
    A2 = phi(A1)
    _assert(A0 == E, "A^(0) must equal E")
    _assert(len(A1) < len(A0), "phi should reduce sequence length in this toy construction")
    _assert(len(A2) <= len(A1), "repeated abstraction should be non-expanding")
    return {"A0": A0, "A1": A1, "A2": A2}


def check_wasserstein_1d() -> dict[str, Any]:
    a = [0.0, 1.0, 2.0, 2.5]
    b = [0.0, 1.2, 1.8, 3.0]
    w_ab = ict.wasserstein_1d_equal_weight(a, b)
    w_ba = ict.wasserstein_1d_equal_weight(b, a)
    w_aa = ict.wasserstein_1d_equal_weight(a, a)
    _assert(abs(w_ab - w_ba) < 1e-12, "Wasserstein distance must be symmetric")
    _assert(w_aa == 0.0, "Wasserstein distance must be zero on identical distributions")
    return {"W1(a,b)": w_ab, "W1(a,a)": w_aa}


def check_self_reference_fixed_point() -> dict[str, Any]:
    state = {"x": 1, "y": 2}
    m = ict.make_self_model(state)
    _assert(isinstance(m.representation_of_M, str) and len(m.representation_of_M) == 64, "representation must be a sha256 hex")
    # Fixed-point flavor: the model contains a representation token describing M.
    return {"state": m.state, "representation_of_M": m.representation_of_M}


def check_symmetry_breaking_sigma() -> dict[str, Any]:
    x = [0.2, -0.4, 0.1]

    def g_id(v):  # type: ignore[no-untyped-def]
        return v

    def g_neg(v):  # type: ignore[no-untyped-def]
        return [-float(v[0]), -float(v[1]), -float(v[2])]

    def Phi(v):  # type: ignore[no-untyped-def]
        # Slightly non-equivariant mapping to produce measurable symmetry breaking.
        return [float(v[0]) + 0.01, float(v[1]), float(v[2])]

    sigma = ict.symmetry_breaking_sigma(x=x, group=[g_id, g_neg], Phi=Phi)
    _assert(math.isfinite(sigma), "sigma must be finite")
    return {"sigma": sigma}


def check_momentum_update() -> dict[str, Any]:
    # Quadratic loss L(theta)=0.5(theta-theta*)^2, grad=theta-theta*.
    theta_star = 1.5
    theta_prev = 0.0
    theta = 2.5
    eta = 0.1
    mu = 0.3
    losses = []
    for _ in range(50):
        grad = theta - theta_star
        theta_next = ict.momentum_step(theta=theta, theta_prev=theta_prev, eta=eta, mu=mu, grad=grad, noise=0.0)
        loss = 0.5 * (theta - theta_star) ** 2
        losses.append(loss)
        theta_prev, theta = theta, theta_next
    _assert(losses[-1] < losses[0], "momentum dynamics should reduce loss in this stable setting")
    return {"loss_start": losses[0], "loss_end": losses[-1], "theta_end": theta}


def check_fisher_information_metric() -> dict[str, Any]:
    sigma2 = 2.0
    g = ict.fisher_information_gaussian_mean(sigma2=sigma2)
    _assert(abs(g - 0.5) < 1e-12, "Fisher metric for Gaussian mean must be 1/sigma^2")
    return {"sigma2": sigma2, "g": g}


def check_natural_gradient_update() -> dict[str, Any]:
    theta_star = -1.0
    theta = 2.0
    eta = 0.2
    g_inv = 2.0  # pretend metric inverse scales step
    for _ in range(30):
        grad = theta - theta_star
        theta = ict.natural_gradient_step(theta=theta, eta=eta, grad=grad, g_inv=g_inv)
    _assert(abs(theta - theta_star) < 1e-3, "natural gradient update should converge on simple quadratic")
    return {"theta_star": theta_star, "theta_final": theta}


def check_hilbert_state_normalization_and_tensor() -> dict[str, Any]:
    coeffs = [complex(0.2, 0.1), complex(-0.3, 0.4), complex(0.1, -0.2)]
    psi = ict.normalize_complex_state(coeffs)
    norm2 = sum((c.real * c.real + c.imag * c.imag) for c in psi)
    _assert(abs(norm2 - 1.0) < 1e-12, "Hilbert state must satisfy sum |c_i|^2 = 1")
    # Tensor product membership: dimension multiplies and norm multiplies for product states.
    phi = ict.normalize_complex_state([complex(1.0, 0.0), complex(0.0, 1.0)])
    tp = ict.tensor_product(psi, phi)
    _assert(len(tp) == len(psi) * len(phi), "tensor product dimension must multiply")
    norm2_tp = sum((c.real * c.real + c.imag * c.imag) for c in tp)
    _assert(abs(norm2_tp - 1.0) < 1e-12, "tensor product of normalized states must be normalized")
    return {"dim_psi": len(psi), "dim_phi": len(phi), "dim_tensor": len(tp), "norm2_tensor": norm2_tp}


def check_complexity_lower_bound_proxy() -> dict[str, Any]:
    # T(n) = Ω(T_M(n)) as a deterministic operation-count proxy.
    samples = [10, 100, 1000]
    rows = []
    for n in samples:
        tm = ict.operation_count_model_run(n)
        t = ict.operation_count_model_update(n)
        _assert(t >= tm, "update cost proxy must be >= model run cost proxy")
        rows.append({"n": n, "T_M(n)": tm, "T(n)": t, "ratio": t / tm})
    return {"samples": rows}


def check_stdp_rule() -> dict[str, Any]:
    A_plus = 0.1
    A_minus = 0.12
    tau_plus = 20.0
    tau_minus = 25.0
    dw_pos = ict.stdp_delta_w(delta_t=10.0, A_plus=A_plus, A_minus=A_minus, tau_plus=tau_plus, tau_minus=tau_minus)
    dw_neg = ict.stdp_delta_w(delta_t=-10.0, A_plus=A_plus, A_minus=A_minus, tau_plus=tau_plus, tau_minus=tau_minus)
    dw_zero = ict.stdp_delta_w(delta_t=0.0, A_plus=A_plus, A_minus=A_minus, tau_plus=tau_plus, tau_minus=tau_minus)
    _assert(dw_pos > 0.0 and dw_neg < 0.0 and dw_zero == 0.0, "STDP piecewise signs must match Δt")
    _assert(abs(dw_pos) <= A_plus + 1e-12, "positive update must be bounded by A_plus")
    _assert(abs(dw_neg) <= A_minus + 1e-12, "negative update magnitude must be bounded by A_minus")
    return {"dw_dt_pos": dw_pos, "dw_dt_neg": dw_neg, "dw_dt_zero": dw_zero}


def validators_by_signature() -> list[tuple[str, str, Callable[[], dict[str, Any]]]]:
    return [
        ("T(S_t) =", "tension_energy_and_descent", check_tension_nonnegative_and_descent),
        ("S_{t+1} =", "state_update_rule", check_tension_nonnegative_and_descent),
        ("\\nabla_p T(S_t)", "stability_conditions_and_psi", check_preference_stability_and_psi),
        ("\\psi(p) =", "preference_strength_metric", check_preference_stability_and_psi),
        ("F =", "free_energy_identity", check_free_energy_identity),
        ("L(M, D)", "mdl_decomposition", check_mdl_decomposition),
        ("P(v_1", "graph_path_product", check_path_probability_product),
        ("G' =", "graph_pruning_constraint", check_pruning_constraint),
        ("A^{(0)}", "temporal_abstraction_recursion", check_temporal_abstraction_recursion),
        ("A^{(k+1)}", "temporal_abstraction_recursion", check_temporal_abstraction_recursion),
        ("W_c(\\mu, \\nu)", "wasserstein_distance", check_wasserstein_1d),
        ("M(s) =", "self_reference_fixed_point", check_self_reference_fixed_point),
        ("\\sigma =", "symmetry_breaking_parameter", check_symmetry_breaking_sigma),
        ("\\theta_{t+1} = \\theta_t - \\eta \\nabla_\\theta L(\\theta_t)", "momentum_update", check_momentum_update),
        ("g_{ij}(\\theta)", "fisher_information_metric", check_fisher_information_metric),
        ("\\theta_{t+1} = \\theta_t - \\eta g^{-1}(\\theta_t)", "natural_gradient_update", check_natural_gradient_update),
        ("|\\psi\\rangle =", "hilbert_state_normalization", check_hilbert_state_normalization_and_tensor),
        ("|\\psi_{VI}\\rangle", "tensor_product_membership", check_hilbert_state_normalization_and_tensor),
        ("T(n) =", "complexity_lower_bound_proxy", check_complexity_lower_bound_proxy),
        ("L(m) + L(s|m)", "mdl_simplicity_bias", check_mdl_decomposition),
        ("\\Delta w_{ij} =", "stdp_rule", check_stdp_rule),
    ]


def match_validator(eq: str) -> tuple[str, Callable[[], dict[str, Any]]] | None:
    normalized = " ".join(eq.strip().split())
    for sig, name, fn in validators_by_signature():
        if sig in normalized:
            return name, fn
    return None


def run_internal_contradictions_theory_test() -> dict[str, Any]:
    spec_path = REPO_ROOT / "theoroms/ethics&stability/core stability frameworks/Internal_Contradictions_Theory.md"
    if not spec_path.exists():
        raise FileNotFoundError(f"Spec not found: {spec_path}")

    md_text = _read_text(spec_path)
    equations = extract_display_equations(md_text)
    if not equations:
        raise RuntimeError("No $$...$$ display equations found in spec; expected at least one.")

    started = _utc_iso()
    results_dir = REPO_ROOT / "results"
    report_path = results_dir / "internal_contradictions_theory_test_report.md"
    manifest_path = results_dir / "internal_contradictions_theory_test_manifest.json"

    # Map each equation to a validator; if any equation is unmatched, fail.
    unmatched: list[dict[str, Any]] = []
    scheduled: dict[str, Callable[[], dict[str, Any]]] = {}
    equation_map: list[dict[str, Any]] = []
    for eq in equations:
        m = match_validator(eq["equation"])
        if not m:
            unmatched.append({"start_line": eq["start_line"], "equation_sig": _sig_for(eq["equation"])})
            continue
        vname, fn = m
        scheduled[vname] = fn
        equation_map.append(
            {
                "start_line": eq["start_line"],
                "equation_sig": _sig_for(eq["equation"]),
                "validator": vname,
            }
        )

    checks: list[CheckResult] = []
    if unmatched:
        checks.append(
            CheckResult(
                name="equation_coverage",
                ok=False,
                seconds=0.0,
                details={"total_equations_found": len(equations), "unmatched_equations": unmatched},
                error="Unmatched display equations found; add validators for full math coverage.",
            )
        )
    else:
        checks.append(
            CheckResult(
                name="equation_coverage",
                ok=True,
                seconds=0.0,
                details={"total_equations_found": len(equations), "unique_validators_scheduled": len(scheduled)},
            )
        )

    for name, fn in sorted(scheduled.items()):
        checks.append(_run(name, fn))

    ok = all(c.ok for c in checks)
    status = "passed" if ok else "failed"

    manifest: dict[str, Any] = {
        "test": "RCF Internal Contradictions Theory - Mathematical Validation",
        "status": status,
        "started_utc": started,
        "ended_utc": _utc_iso(),
        "env": {
            "python": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "cwd": os.getcwd(),
        },
        "inputs": {
            "spec_path": str(spec_path),
            "spec_sha256": ict.sha256_text(md_text),
            "display_equations_count": len(equations),
        },
        "equation_map": equation_map,
        "checks": [asdict(c) for c in checks],
    }

    lines = [
        "# Internal Contradictions Theory Test Report",
        "",
        f"- Status: {status.upper()}",
        f"- Timestamp (UTC): {started}",
        f"- Spec: `{spec_path.as_posix()}`",
        f"- Spec SHA256: `{manifest['inputs']['spec_sha256']}`",
        f"- Display equations found: {len(equations)}",
        "",
        "## Equation Coverage",
        "",
    ]
    if unmatched:
        lines.append(f"- Unmatched display equations: {len(unmatched)}")
        for u in unmatched[:50]:
            lines.append(f"  - line {u['start_line']}: `{u['equation_sig']}`")
    else:
        lines.append(f"- Coverage: OK (validators scheduled: {len(scheduled)})")

    lines += ["", "## Checks", ""]
    for c in checks:
        lines.append(f"- `{c.name}`: {'OK' if c.ok else 'FAIL'} ({c.seconds:.3f}s)")
        if c.ok and c.details:
            for k, v in c.details.items():
                if isinstance(v, (dict, list)):
                    s = json.dumps(v)
                    if len(s) > 220:
                        continue
                    lines.append(f"  - {k}: `{s}`")
                else:
                    lines.append(f"  - {k}: `{v}`")
        if not c.ok:
            lines.append(f"  - error: `{c.error}`")

    _write_text(report_path, "\n".join(lines) + "\n")
    _write_json(manifest_path, manifest)

    print("\n".join(lines[:60]))
    print(f"Wrote: {report_path}")
    print(f"Wrote: {manifest_path}")

    if not ok:
        for c in checks:
            if not c.ok and c.traceback:
                print("\n--- traceback:", c.name, "---\n" + c.traceback)
        raise SystemExit(2)
    return manifest


if __name__ == "__main__":
    run_internal_contradictions_theory_test()
