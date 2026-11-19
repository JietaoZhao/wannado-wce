"""One-Command Machine Verification Gate (Stage 0 -> Preflight -> Rigor 2.0 -> Link-Level PHY)."""

from __future__ import annotations

import importlib.util
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from wannado.wce.audit.preflight import PreflightReport, preflight_wce
from wannado.wce.audit.rigor_auditor import RigorAuditReport, audit_wce_rigor
from wannado.wce.blocks import BLOCK_ID_TO_RUNNER
from wannado.wce.core.types import ProvisionResult
from wannado.wce.intake.archetypes import SpecAuditReport, audit_wce_spec
from wannado.wce.intake.composer import compose_wce_spec
from wannado.wce.intake.spec import WCESpec
from wannado.wce.workspace.composed_env import ComposedWirelessEnv
from wannado.wce.workspace.link_level_phy import LinkLevelPHYResult, evaluate_link_level_phy


@dataclass
class WCEVerifyReport:
    """Structured machine-verifiable outcome emitted by `verify_wce_target` / `wannado-wce verify`."""

    passed: bool
    exit_code: int
    target_mode: str
    target_name: str
    checks: dict[str, bool] = field(default_factory=dict)
    red_flags: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stage0_audit: dict[str, Any] | None = None
    preflights: dict[str, dict[str, Any]] = field(default_factory=dict)
    rigor_reports: dict[str, dict[str, Any]] = field(default_factory=dict)
    composed_rigor: dict[str, Any] | None = None
    phy_validation: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def format_report(self) -> str:
        verdict = "PASS" if self.passed else "FAIL"
        lines = [
            f"[wannado-wce verify: {verdict}] target={self.target_name} ({self.target_mode})",
            "  Gate Checks:",
        ]
        for k, v in self.checks.items():
            lines.append(f"    - {k}: {'OK' if v else 'FAIL'}")
        if self.phy_validation and self.phy_validation.get("phy_applicable"):
            p = self.phy_validation
            lines.append(
                f"  Link-Level PHY ({p['modulation']}/{p['equalizer'].upper()}/est={p['estimator']} @ {p['snr_db']:.1f} dB): "
                f"uncoded_ber={p['uncoded_ber']:.4f}, coded_ber={p['coded_ber']:.4f}, coded_bler={p['coded_bler']:.4f}, "
                f"csi_nmse={p['csi_nmse_db']:.2f} dB"
            )
        if self.red_flags:
            lines.append("  Red Flags:")
            lines.extend(f"    - {rf}" for rf in self.red_flags)
        if self.warnings:
            lines.append("  Warnings:")
            lines.extend(f"    - {w}" for w in self.warnings)
        return "\n".join(lines)


def _load_interface_spec_and_blocks(interface_path: Path) -> tuple[WCESpec | None, list[str]]:
    p = interface_path.resolve()
    mod_spec = importlib.util.spec_from_file_location("_wannado_wce_verify_iface", p)
    if mod_spec is None or mod_spec.loader is None:
        raise FileNotFoundError(f"Cannot load WCE interface module at {p}")
    mod = importlib.util.module_from_spec(mod_spec)
    mod_spec.loader.exec_module(mod)
    iface = mod.load_wireless_interface()
    return iface.wce_spec, list(iface.blocks)


def verify_wce_target(
    *,
    spec_path: str | Path | None = None,
    wce_spec: WCESpec | None = None,
    interface_path: str | Path | None = None,
    query: str | None = None,
    blocks: Sequence[str] | None = None,
    seeds: Sequence[int] = (0, 1, 2),
    min_seeds: int = 3,
    with_phy_check: bool = True,
    phy_snr_db: float = 12.0,
    strict_warnings: bool = False,
) -> WCEVerifyReport:
    """Execute end-to-end Stage 0 -> Preflight -> Rigor 2.0 -> Link-Level PHY verification gate."""
    from wannado.wce.reporter.env_reporter import build_block_config

    seeds_t = tuple(int(s) for s in seeds)
    if not seeds_t:
        raise ValueError("seeds must contain at least one integer seed.")

    target_mode = "blocks"
    target_name = "custom"
    spec_obj: WCESpec | None = wce_spec
    block_ids: list[str] = list(blocks) if blocks else []

    if interface_path is not None:
        target_mode = "interface"
        target_name = str(Path(interface_path).resolve())
        spec_obj, iface_blocks = _load_interface_spec_and_blocks(Path(interface_path))
        if not block_ids:
            block_ids = iface_blocks
    elif spec_path is not None:
        target_mode = "spec_file"
        target_name = str(Path(spec_path).resolve())
        spec_obj = WCESpec.load(spec_path)
    elif wce_spec is not None:
        target_mode = "wce_spec"
        target_name = wce_spec.problem_name
    elif query is not None:
        target_mode = "query"
        target_name = query
        spec_obj, _ = compose_wce_spec(query)
    elif not block_ids:
        raise ValueError("Provide one of wce_spec, spec_path, interface_path, query, or blocks.")

    if spec_obj is not None and not block_ids:
        block_ids = list(spec_obj.blocks_selected)

    checks: dict[str, bool] = {}
    red_flags: list[str] = []
    warnings: list[str] = []

    # Gate 1: Stage 0 Spec & Table I Provenance Audit
    stage0_dict: dict[str, Any] | None = None
    if spec_obj is not None:
        spec_aud: SpecAuditReport = audit_wce_spec(spec_obj)
        stage0_dict = spec_aud.to_dict()
        warnings.extend(spec_aud.warnings)
        stage0_ok = spec_aud.ok
        if not stage0_ok:
            red_flags.extend(f"stage0_error:{e}" for e in spec_aud.errors)
        # Verify Table I provenance key uniqueness
        seen_keys: set[str] = set()
        for row in spec_obj.parameter_provenance:
            if row.parameter in seen_keys or not str(row.citation_key).strip():
                stage0_ok = False
                red_flags.append("table1_provenance_integrity_violation")
                break
            seen_keys.add(row.parameter)
        checks["stage0_spec_audit_ok"] = stage0_ok
    else:
        checks["stage0_spec_audit_ok"] = True

    # Gate 2: Stage 1 Preflight across active blocks
    preflights_dict: dict[str, dict[str, Any]] = {}
    preflight_ok = True
    for bid in block_ids:
        if bid not in BLOCK_ID_TO_RUNNER:
            preflight_ok = False
            red_flags.append(f"unknown_block:{bid}")
            continue
        cfg = build_block_config(bid, wce_spec=spec_obj) if spec_obj is not None else None
        pf: PreflightReport = preflight_wce(bid, config=cfg, wce_spec=spec_obj, seed=seeds_t[0])
        preflights_dict[bid] = pf.to_dict()
        warnings.extend(f"{bid}:{w}" for w in pf.warnings)
        if not pf.ok:
            preflight_ok = False
            red_flags.extend(f"preflight_{bid}:{e}" for e in pf.errors)
    checks["stage1_preflight_ok"] = preflight_ok

    # Gate 3: Stage 3 Rigor Auditor 2.0 across ComposedWirelessEnv and/or per-block runs
    rigor_reports_dict: dict[str, dict[str, Any]] = {}
    rigor_ok = True
    has_baseline = bool(spec_obj and spec_obj.baseline_blocks) or any(
        bid in {"awgn_link", "wmmse_precoder", "water_filling", "ls_lmmse_csi", "proportional_fair"}
        for bid in block_ids
    )
    composed_rigor_dict: dict[str, Any] | None = None
    phy_source: Any = None

    if spec_obj is not None and checks.get("stage0_spec_audit_ok", False):
        comp_runs = [
            ComposedWirelessEnv(spec_obj, batch_size=8, seed=s).to_rigor_result(num_slots=8, seed=s)
            for s in seeds_t
        ]
        comp_rep: RigorAuditReport = audit_wce_rigor(
            comp_runs,
            wce_spec=spec_obj,
            has_classic_baseline=has_baseline,
            min_seeds=int(min_seeds),
        )
        composed_rigor_dict = comp_rep.to_dict()
        if not comp_rep.passed:
            rigor_ok = False
            red_flags.extend(f"composed:{rf}" for rf in comp_rep.red_flags)
        phy_source = ComposedWirelessEnv(spec_obj, batch_size=8, seed=seeds_t[0])
    else:
        for bid in block_ids:
            if bid not in BLOCK_ID_TO_RUNNER:
                continue
            mod_path, fn_name = BLOCK_ID_TO_RUNNER[bid]
            fn = getattr(__import__(mod_path, fromlist=[fn_name]), fn_name)
            cfg = build_block_config(bid, wce_spec=spec_obj) if spec_obj is not None else None
            runs: list[ProvisionResult] = []
            for s in seeds_t:
                if cfg is not None:
                    cfg_s = type(cfg)(**{**asdict(cfg), "seed": int(s)})
                    runs.append(fn(config=cfg_s))
                else:
                    runs.append(fn(seed=int(s)))
            rep = audit_wce_rigor(
                runs,
                wce_spec=spec_obj,
                has_classic_baseline=has_baseline,
                min_seeds=int(min_seeds),
            )
            rigor_reports_dict[bid] = rep.to_dict()
            if not rep.passed:
                rigor_ok = False
                red_flags.extend(f"{bid}:{rf}" for rf in rep.red_flags)
            if phy_source is None:
                for tkey in ("h_freq", "h_small", "h", "h_near", "h_eff"):
                    if tkey in runs[0].tensors:
                        phy_source = runs[0]
                        break
    checks["stage3_rigor_audit_ok"] = rigor_ok

    # Gate 4: Link-Level OFDM-MIMO PHY Validation
    phy_dict: dict[str, Any] | None = None
    phy_ok = True
    if with_phy_check:
        if phy_source is not None:
            try:
                phy_res: LinkLevelPHYResult = evaluate_link_level_phy(
                    phy_source,
                    snr_db=phy_snr_db,
                    modulation="QPSK",
                    equalizer="mmse",
                    estimator="ls",
                    coded=True,
                    num_blocks=24,
                    seed=seeds_t[0],
                )
                phy_dict = {**phy_res.to_dict(), "phy_applicable": True}
                finite_ok = (
                    0.0 <= phy_res.uncoded_ber <= 1.0
                    and 0.0 <= phy_res.coded_ber <= 1.0
                    and 0.0 <= phy_res.coded_bler <= 1.0
                )
                if not finite_ok:
                    phy_ok = False
                    red_flags.append("link_level_phy_nonfinite")
                elif phy_snr_db >= 0.0 and phy_res.coded_bler >= 0.99 and phy_res.uncoded_ber >= 0.40:
                    phy_ok = False
                    red_flags.append("link_level_phy_complete_outage_at_nominal_snr")
            except Exception as exc:  # noqa: BLE001
                phy_ok = False
                red_flags.append(f"link_level_phy_error:{exc}")
                phy_dict = {"phy_applicable": True, "error": str(exc)}
        else:
            phy_dict = {"phy_applicable": False, "reason": "No complex MIMO/OFDM channel tensor among selected blocks."}
    checks["link_level_phy_ok"] = phy_ok

    if strict_warnings and warnings:
        red_flags.append("strict_warnings_triggered")

    passed = all(checks.values()) and len(red_flags) == 0
    return WCEVerifyReport(
        passed=passed,
        exit_code=0 if passed else 1,
        target_mode=target_mode,
        target_name=target_name,
        checks=checks,
        red_flags=red_flags,
        warnings=warnings,
        stage0_audit=stage0_dict,
        preflights=preflights_dict,
        rigor_reports=rigor_reports_dict,
        composed_rigor=composed_rigor_dict,
        phy_validation=phy_dict,
    )


__all__ = ["WCEVerifyReport", "verify_wce_target"]
