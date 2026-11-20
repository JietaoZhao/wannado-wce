"""Cryptographically Sealed WCE Verification Dossier & CI PR Regression Gate."""

from __future__ import annotations

import hashlib
import hmac
import json
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _canonical_json(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _get_git_provenance(cwd: str | Path = ".") -> dict[str, Any]:
    root = Path(cwd).resolve()
    info: dict[str, Any] = {"git_commit": None, "git_branch": None, "git_dirty": None}
    try:
        sha = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
        if sha.returncode == 0:
            info["git_commit"] = sha.stdout.strip()
        br = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--abbrev-ref", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
        if br.returncode == 0:
            info["git_branch"] = br.stdout.strip()
        st = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
        if st.returncode == 0:
            info["git_dirty"] = bool(st.stdout.strip())
    except (OSError, subprocess.SubprocessError) as exc:
        info["git_error"] = str(exc)
    return info


@dataclass
class WCEVerificationDossier:
    """Tamper-evident 3GPP/IMT-2030 physics & link-level verification evidence dossier."""

    schema_version: str
    generated_at_utc: str
    target_name: str
    target_mode: str
    overall_passed: bool
    git_provenance: dict[str, Any]
    checks: dict[str, bool]
    red_flags: list[str]
    wce_spec: dict[str, Any] | None = None
    phy_validation: dict[str, Any] | None = None
    harq_amc: dict[str, Any] | None = None
    calibration_report: dict[str, Any] | None = None
    regression_comparison: dict[str, Any] | None = None
    payload_sha256: str = ""
    hmac_sha256: str | None = None
    saved_paths: dict[str, str] = field(default_factory=dict)

    def body_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "generated_at_utc": self.generated_at_utc,
            "target_name": self.target_name,
            "target_mode": self.target_mode,
            "overall_passed": self.overall_passed,
            "git_provenance": self.git_provenance,
            "checks": self.checks,
            "red_flags": self.red_flags,
            "wce_spec": self.wce_spec,
            "phy_validation": self.phy_validation,
            "harq_amc": self.harq_amc,
            "calibration_report": self.calibration_report,
            "regression_comparison": self.regression_comparison,
        }

    def seal(self, signing_key: str | bytes | None = None) -> WCEVerificationDossier:
        """Compute canonical SHA-256 content digest and optional HMAC-SHA256 signature."""
        canon = _canonical_json(self.body_dict())
        self.payload_sha256 = hashlib.sha256(canon).hexdigest()
        if signing_key is not None:
            key_b = signing_key.encode("utf-8") if isinstance(signing_key, str) else signing_key
            self.hmac_sha256 = hmac.new(key_b, canon, hashlib.sha256).hexdigest()
        else:
            self.hmac_sha256 = None
        return self

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def format_markdown(self) -> str:
        verdict = "PASS" if self.overall_passed else "FAIL"
        lines = [
            f"# wannado-wce Verification Dossier ({verdict})",
            "",
            f"- **Generated (UTC)**: `{self.generated_at_utc}`",
            f"- **Target**: `{self.target_name}` (`{self.target_mode}`)",
            f"- **Payload SHA-256**: `{self.payload_sha256}`",
            f"- **HMAC-SHA256 Signed**: `{self.hmac_sha256 is not None}`",
            "",
            "## Gate Checks",
        ]
        for k, v in self.checks.items():
            lines.append(f"- `{k}`: {'OK' if v else 'FAIL'}")
        if self.red_flags:
            lines.append("\n## Red Flags")
            lines.extend(f"- `{rf}`" for rf in self.red_flags)
        return "\n".join(lines) + "\n"

    def save(self, out_dir: str | Path) -> dict[str, Path]:
        """Write ``wce_dossier.json`` and ``wce_dossier.md`` inside ``out_dir``."""
        root = Path(out_dir).resolve()
        root.mkdir(parents=True, exist_ok=True)
        if not self.payload_sha256:
            self.seal()
        json_p = root / "wce_dossier.json"
        md_p = root / "wce_dossier.md"
        self.saved_paths = {"json": str(json_p), "markdown": str(md_p)}
        json_p.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        md_p.write_text(self.format_markdown(), encoding="utf-8")
        return {"json": json_p, "markdown": md_p}


def build_wce_dossier(
    verify_report: Any,
    *,
    wce_spec: Any = None,
    calibration_report: Any = None,
    harq_result: Any = None,
    regression_comparison: dict[str, Any] | None = None,
    signing_key: str | bytes | None = None,
    cwd: str | Path = ".",
) -> WCEVerificationDossier:
    """Build and cryptographically seal a WCE verification evidence dossier."""
    rep = verify_report.to_dict() if hasattr(verify_report, "to_dict") else dict(verify_report)
    spec_dict = wce_spec.to_dict() if hasattr(wce_spec, "to_dict") else wce_spec
    cal_dict = calibration_report.to_dict() if hasattr(calibration_report, "to_dict") else calibration_report
    harq_dict = harq_result.to_dict() if hasattr(harq_result, "to_dict") else harq_result

    dossier = WCEVerificationDossier(
        schema_version="wce.dossier.v1",
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        target_name=str(rep.get("target_name", "unknown")),
        target_mode=str(rep.get("target_mode", "unknown")),
        overall_passed=bool(rep.get("passed", False)),
        git_provenance=_get_git_provenance(cwd),
        checks=dict(rep.get("checks") or {}),
        red_flags=list(rep.get("red_flags") or []),
        wce_spec=spec_dict,
        phy_validation=rep.get("phy_validation"),
        harq_amc=harq_dict or rep.get("harq_amc"),
        calibration_report=cal_dict or rep.get("calibration_report"),
        regression_comparison=regression_comparison or rep.get("regression_comparison"),
    )
    return dossier.seal(signing_key=signing_key)


def verify_wce_dossier_integrity(
    dossier_or_path: WCEVerificationDossier | dict[str, Any] | str | Path,
    *,
    signing_key: str | bytes | None = None,
) -> dict[str, Any]:
    """Verify SHA-256 content hash and optional HMAC-SHA256 signature on a saved WCE dossier."""
    if isinstance(dossier_or_path, (str, Path)):
        raw = json.loads(Path(dossier_or_path).read_text(encoding="utf-8"))
    elif isinstance(dossier_or_path, WCEVerificationDossier):
        raw = dossier_or_path.to_dict()
    else:
        raw = dict(dossier_or_path)

    body = {
        "schema_version": raw.get("schema_version"),
        "generated_at_utc": raw.get("generated_at_utc"),
        "target_name": raw.get("target_name"),
        "target_mode": raw.get("target_mode"),
        "overall_passed": raw.get("overall_passed"),
        "git_provenance": raw.get("git_provenance"),
        "checks": raw.get("checks"),
        "red_flags": raw.get("red_flags"),
        "wce_spec": raw.get("wce_spec"),
        "phy_validation": raw.get("phy_validation"),
        "harq_amc": raw.get("harq_amc"),
        "calibration_report": raw.get("calibration_report"),
        "regression_comparison": raw.get("regression_comparison"),
    }
    canon = _canonical_json(body)
    expected_sha = hashlib.sha256(canon).hexdigest()
    sha_ok = hmac.compare_digest(str(raw.get("payload_sha256", "")), expected_sha)

    hmac_ok: bool | None = None
    if signing_key is not None:
        sig = raw.get("hmac_sha256")
        if not isinstance(sig, str) or not sig:
            hmac_ok = False
        else:
            key_b = signing_key.encode("utf-8") if isinstance(signing_key, str) else signing_key
            expected_hmac = hmac.new(key_b, canon, hashlib.sha256).hexdigest()
            hmac_ok = hmac.compare_digest(sig, expected_hmac)

    return {
        "valid": bool(sha_ok and (hmac_ok is not False)),
        "sha256_valid": sha_ok,
        "hmac_valid": hmac_ok,
    }


def _extract_summary_kpis(rep: dict[str, Any]) -> dict[str, float | None]:
    comp = rep.get("composed_rigor") or {}
    stats = comp.get("seed_stats") or {}
    mean_se = stats.get("mean")
    if mean_se is None and rep.get("rigor_reports"):
        first = next(iter(rep["rigor_reports"].values()))
        mean_se = (first.get("seed_stats") or {}).get("mean")
    phy = rep.get("phy_validation") or {}
    harq = rep.get("harq_amc") or {}
    bler = harq.get("residual_bler") if harq else phy.get("coded_bler")
    evm = phy.get("evm_pct")
    return {
        "mean_metric": float(mean_se) if mean_se is not None else None,
        "bler": float(bler) if bler is not None else None,
        "evm_pct": float(evm) if evm is not None else None,
    }


def compare_wce_verdicts(
    baseline_or_path: dict[str, Any] | str | Path,
    current_report: Any,
    *,
    max_se_drop_pct: float = 5.0,
    max_bler_increase: float = 0.03,
    max_evm_increase_pct: float = 3.0,
) -> dict[str, Any]:
    """Compare current WCE verify verdict against a baseline JSON/dict to catch PR regressions."""
    if isinstance(baseline_or_path, (str, Path)):
        base = json.loads(Path(baseline_or_path).read_text(encoding="utf-8"))
    else:
        base = baseline_or_path.to_dict() if hasattr(baseline_or_path, "to_dict") else dict(baseline_or_path)
    cur = current_report.to_dict() if hasattr(current_report, "to_dict") else dict(current_report)

    b_kpi = _extract_summary_kpis(base)
    c_kpi = _extract_summary_kpis(cur)
    regressions: list[str] = []

    if b_kpi["mean_metric"] is not None and c_kpi["mean_metric"] is not None and abs(b_kpi["mean_metric"]) > 1e-9:
        drop_pct = ((b_kpi["mean_metric"] - c_kpi["mean_metric"]) / abs(b_kpi["mean_metric"])) * 100.0
        if drop_pct > float(max_se_drop_pct):
            regressions.append(f"regression:metric_drop({drop_pct:.2f}%>{float(max_se_drop_pct):.2f}%)")

    if b_kpi["bler"] is not None and c_kpi["bler"] is not None:
        bler_delta = c_kpi["bler"] - b_kpi["bler"]
        if bler_delta > float(max_bler_increase):
            regressions.append(f"regression:bler_increase({bler_delta:+.4f}>{float(max_bler_increase):.4f})")

    if b_kpi["evm_pct"] is not None and c_kpi["evm_pct"] is not None:
        evm_delta = c_kpi["evm_pct"] - b_kpi["evm_pct"]
        if evm_delta > float(max_evm_increase_pct):
            regressions.append(f"regression:evm_increase({evm_delta:+.2f}%>{float(max_evm_increase_pct):.2f}%)")

    return {
        "passed": len(regressions) == 0,
        "baseline": b_kpi,
        "current": c_kpi,
        "regressions": regressions,
    }


__all__ = [
    "WCEVerificationDossier",
    "build_wce_dossier",
    "compare_wce_verdicts",
    "verify_wce_dossier_integrity",
]
