"""O-RAN WG3 E2SM-KPM v3.0 Telemetry Stream & E2SM-RC Control Loop Adapter over ComposedWirelessEnv."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from wannado.wce.phy.nr_coding_and_amc import select_nr_mcs_from_sinr
from wannado.wce.workspace.composed_env import ComposedSlotOutput, ComposedWirelessEnv


@dataclass
class ORANKPMIndication:
    """Standardized O-RAN E2SM-KPM v03.00 cell & per-UE RIC Indication record."""

    format_version: str
    ric_indication_sn: int
    timestamp_ms: float
    granularity_period_ms: float
    cell_id: str
    meas_info_list: tuple[str, ...]
    cell_metrics: dict[str, float]
    ue_measurements: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _snr_to_cqi(sinr_db: float) -> int:
    """Map effective SINR (dB) to 3GPP TS 38.214 4-bit CQI index [1..15]."""
    idx = round((float(sinr_db) + 6.0) / 2.2) + 1
    return max(1, min(15, idx))


def _estimate_bler_from_margin(sinr_db: float, min_sinr_db: float) -> float:
    """Smooth logistic transport-block BLER proxy around MCS SINR threshold (10% target at threshold)."""
    margin = float(sinr_db) - float(min_sinr_db)
    # At margin == 0 -> log1p/logistic gives 0.10 BLER; higher margin pushes BLER toward 0
    return float(1.0 / (1.0 + math.exp(2.2 + 1.4 * margin)))


class ORANKPMTelemetryAdapter:
    """Expose any ``ComposedWirelessEnv`` as an O-RAN WG3 E2SM-KPM v3.0 telemetry + E2SM-RC control testbed."""

    MEAS_INFO_LIST: tuple[str, ...] = (
        "DRB.UEThpDl",
        "DRB.RlcSduDelayDl",
        "RRU.PrbUsedDl",
        "L1.RSRP",
        "L1.SINR",
        "L1.CQI",
        "DL.BLER",
    )

    def __init__(
        self,
        env: ComposedWirelessEnv,
        *,
        cell_id: str = "gNB-DU-001-Cell-01",
        total_prbs: int = 273,
    ) -> None:
        self.env = env
        self.cell_id = cell_id
        self.total_prbs = max(1, int(total_prbs))
        self._sn = 0
        self._history: list[ORANKPMIndication] = []

    def reset(self, *, seed: int | None = None) -> ORANKPMIndication:
        self._sn = 0
        self._history.clear()
        out = self.env.reset(seed=seed)
        return self._build_indication(out, prb_shares=None)

    def step_indication(
        self,
        *,
        rc_control: dict[str, Any] | None = None,
        distances_m: np.ndarray | Sequence[float] | float | None = None,
        tx_pos_3d: np.ndarray | Sequence[float] | None = None,
        rx_pos_3d: np.ndarray | Sequence[float] | None = None,
    ) -> ORANKPMIndication:
        """Advance 1 RIC reporting step applying optional E2SM-RC control action and return E2SM-KPM v3 indication."""
        power_weights = None
        prb_shares = None
        power_scale = None
        ris_scale = None

        if rc_control:
            if "per_user_power_weights" in rc_control:
                power_weights = rc_control["per_user_power_weights"]
            if "prb_allocation" in rc_control:
                raw_prb = np.maximum(np.asarray(rc_control["prb_allocation"], dtype=np.float64), 0.0)
                s = float(np.sum(raw_prb))
                prb_shares = raw_prb / max(s, 1e-12) if raw_prb.size == self.env.batch_size else None
            if "power_scale" in rc_control:
                power_scale = float(rc_control["power_scale"])
            if "ris_scale" in rc_control:
                ris_scale = float(rc_control["ris_scale"])

        out = self.env.step(
            distances_m=distances_m,
            tx_pos_3d=tx_pos_3d,
            rx_pos_3d=rx_pos_3d,
            per_user_power_weights=power_weights,
            power_scale=power_scale,
            ris_scale=ris_scale,
        )
        return self._build_indication(out, prb_shares=prb_shares)

    def _build_indication(
        self,
        out: ComposedSlotOutput,
        *,
        prb_shares: np.ndarray | None,
    ) -> ORANKPMIndication:
        n_ue = self.env.batch_size
        if prb_shares is None:
            prb_shares = np.full(n_ue, 1.0 / float(n_ue), dtype=np.float64)

        snr_lin = (
            self.env._last_eff_snr_lin
            if self.env._last_eff_snr_lin is not None
            else np.ones(n_ue, dtype=np.float64)
        )
        se_arr = (
            self.env._last_se
            if self.env._last_se is not None
            else np.full(n_ue, out.achievable_rate_bps_hz, dtype=np.float64)
        )
        noise_dbm = 10.0 * math.log10(max(self.env.noise_w, 1e-30)) + 30.0
        bw_mbps_per_hz = self.env.spec.bandwidth_hz / 1e6

        ue_rows: list[dict[str, Any]] = []
        prbs_used_total = 0
        bler_vals: list[float] = []
        for u in range(n_ue):
            sinr_db = 10.0 * math.log10(max(float(snr_lin[u]), 1e-12))
            rsrp_dbm = noise_dbm + sinr_db - 10.0 * math.log10(max(float(self.total_prbs), 1.0))
            mcs = select_nr_mcs_from_sinr(sinr_db)
            cqi = _snr_to_cqi(sinr_db)
            bler = _estimate_bler_from_margin(sinr_db, float(mcs["min_sinr_db"]))
            bler_vals.append(bler)
            prb_u = round(float(prb_shares[u]) * self.total_prbs)
            prbs_used_total += prb_u
            thp_mbps = float(se_arr[u]) * bw_mbps_per_hz * float(prb_shares[u]) * max(0.0, 1.0 - bler)
            q_bits = float(self.env._queue_bits[u])
            cap_rate_bps = max(float(se_arr[u]) * self.env.spec.bandwidth_hz * float(prb_shares[u]), 1e3)
            delay_ms = min(1000.0, (q_bits / cap_rate_bps) * 1e3)
            ue_rows.append(
                {
                    "ue_id": f"UE-{u:03d}",
                    "DRB.UEThpDl": round(thp_mbps, 4),
                    "DRB.RlcSduDelayDl": round(delay_ms, 3),
                    "RRU.PrbUsedDl": prb_u,
                    "L1.RSRP": round(rsrp_dbm, 2),
                    "L1.SINR": round(sinr_db, 2),
                    "L1.CQI": cqi,
                    "DL.BLER": round(bler, 4),
                    "mcs_index": int(mcs["mcs_index"]),
                }
            )

        ind = ORANKPMIndication(
            format_version="E2SM-KPM-v03.00",
            ric_indication_sn=self._sn,
            timestamp_ms=round(float(out.slot_index) * self.env.spec.slot_duration_ms, 3),
            granularity_period_ms=float(self.env.spec.slot_duration_ms),
            cell_id=self.cell_id,
            meas_info_list=self.MEAS_INFO_LIST,
            cell_metrics={
                "DRB.CellThpDl_Mbps": round(sum(float(r["DRB.UEThpDl"]) for r in ue_rows), 4),
                "RRU.PrbAvailDl": self.total_prbs,
                "RRU.PrbUsedDl": prbs_used_total,
                "DL.MeanBLER": round(float(np.mean(bler_vals)) if bler_vals else 0.0, 4),
                "L1.MeanSINR_dB": round(float(out.mean_snr_db), 3),
                "XR.PDBSatisfaction": round(float(out.xr_pdb_satisfaction), 4),
            },
            ue_measurements=ue_rows,
        )
        self._sn += 1
        self._history.append(ind)
        return ind

    def export_jsonl(self, path: str | Path) -> Path:
        """Write all recorded E2SM-KPM v3.0 indications as newline-delimited JSON."""
        p = Path(path).resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        lines = [json.dumps(item.to_dict(), ensure_ascii=False) for item in self._history]
        p.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        return p


__all__ = ["ORANKPMIndication", "ORANKPMTelemetryAdapter"]
