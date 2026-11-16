"""Physical-Layer Security, Covert Square-Root Law & JSR Jamming (Wyner 1975; Bash 2013; Sadeghi 2019).

[Lineage & Intuition]
- Family / Era / Tier : domain_6g / frontier / S (IMT-2030 Resilience & Bridge to wannado-adv)
- Parent              : `awgn_link`
- Key Diff            : Evaluates Wyner secrecy rate C_s=[C_B-C_E]^+, Bash covert KL divergence D(P1||P0)
                        over n channel uses, and complex64 JSR-dB adversarial/jamming waveforms.
- Extension Hook [AI-Prompt Hook]:
    1. Hand off complex64 IQ samples + channel directly to `wannado-adv` (`norm='jsr_db'`).
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from typing import Any

import torch

from wannado.wce.core.provenance import MECHANISM_PROVENANCE
from wannado.wce.core.types import ParamProvenance, ProvisionResult, WCEMeta
from wannado.wce.core.units import db_to_linear
from wannado.wce.core.utils import complex_normal, resolve_device, set_global_seed

META = WCEMeta(
    id="pls_covert_antijamming",
    name="Physical-Layer Secrecy, Covert Comm & RF Anti-Jamming Environment",
    family="domain_6g",
    era="frontier",
    tier="S",
    output_kind="complex_channel_tensor",
    supports_torch_complex=True,
    supports_autograd=True,
    standard_ids=("itur_m2160",),
    parent_id="awgn_link",
    diff_from_parent="Adds Bob-vs-Eve wiretap secrecy rate, Bash O(sqrt(n)) covert KL detection, and JSR-dB RF jamming.",
    summary_plain="Security & resilience RF environment covering wiretap secrecy, covert communication limits, and JSR jamming aligned with wannado-adv.",
    when_to_use="PLS beamforming, covert UAV/IoT transmissions, and adversarial/anti-jamming baseband RF benchmarks.",
    when_not_to_use="Benign non-contested propagation links.",
    hyperparam_intuition={
        "jsr_db": "Jamming-to-Signal Ratio in dB (e.g., -10 dB light jammer, 0 dB equal power, +10 dB strong barrage).",
        "bob_snr_db": "Legitimate receiver Bob's average SNR.",
        "eve_snr_db": "Eavesdropper/Warden Eve's average SNR.",
    },
    extension_hooks=("Export tensors['iq_signal'] + tensors['jammed_signal'] to wannado-adv VictimInterface",),
    module_path="wannado.wce.blocks.domain_6g.pls_covert_antijamming",
    default_mechanisms=("wyner_wiretap_secrecy", "bash_covert_sqrt_law", "sadeghi_rf_adversarial"),
    citation=MECHANISM_PROVENANCE["sadeghi_rf_adversarial"].citation,
)


@dataclass
class WirelessSecurityConfig:
    batch_size: int = 256
    blocklength_n: int = 128
    bob_snr_db: float = 15.0
    eve_snr_db: float = 5.0
    jsr_db: float = -10.0
    covert_tx_power_ratio: float = 0.08
    device: str = "auto"
    seed: int = 42


def provision_pls_covert_antijamming(config: WirelessSecurityConfig | None = None, **overrides: Any) -> ProvisionResult:
    """Provision wiretap secrecy fading, covert KL detection check, and calibrated JSR-dB jammed IQ signals."""
    cfg = config or WirelessSecurityConfig()
    if overrides:
        cfg = WirelessSecurityConfig(**{**asdict(cfg), **overrides})
    t0 = time.perf_counter()
    set_global_seed(cfg.seed)
    dev = resolve_device(cfg.device)

    h_bob = complex_normal((cfg.batch_size,), variance=1.0, device=dev)
    h_eve = complex_normal((cfg.batch_size,), variance=1.0, device=dev)
    snr_b = float(db_to_linear(cfg.bob_snr_db)) * h_bob.abs().pow(2)
    snr_e = float(db_to_linear(cfg.eve_snr_db)) * h_eve.abs().pow(2)
    c_bob = torch.log2(1.0 + snr_b)
    c_eve = torch.log2(1.0 + snr_e)
    c_sec = torch.clamp(c_bob - c_eve, min=0.0)

    # Bash et al. (2013) covert communication Pinsker upper bound on warden detection accuracy
    # For complex Gaussian signalling P_f = P / sigma_w^2 over n uses: D(P1||P0) = n * [ln(1+pf) - pf/(1+pf)]
    pf = float(cfg.covert_tx_power_ratio) * float(db_to_linear(cfg.eve_snr_db))
    kl_div = float(cfg.blocklength_n * (math.log(1.0 + pf) - pf / (1.0 + pf)))
    min_warden_error_prob = max(0.0, 1.0 - math.sqrt(0.5 * kl_div))  # Pinsker xi >= 1 - sqrt(D/2)

    # Calibrated JSR-dB jammed baseband waveform compatible with wannado-adv
    iq_clean = torch.complex(
        torch.sign(torch.randn((cfg.batch_size, cfg.blocklength_n), device=dev)) / math.sqrt(2.0),
        torch.sign(torch.randn((cfg.batch_size, cfg.blocklength_n), device=dev)) / math.sqrt(2.0),
    ).to(torch.complex64)
    sig_pow = torch.mean(iq_clean.abs().pow(2)).item()
    jam_var = sig_pow * float(db_to_linear(cfg.jsr_db))
    jammer = complex_normal((cfg.batch_size, cfg.blocklength_n), variance=jam_var, device=dev)
    iq_jammed = (iq_clean + jammer).to(torch.complex64)

    prov = [
        ParamProvenance("jsr_db", cfg.jsr_db, "dB", "Jamming-to-Signal power ratio", "sadeghi_rf_adversarial", "IEEE WCL 2019", "sadeghi2019adversarial", "Matches wannado-adv complex64 norm='jsr_db' threat constraint."),
        ParamProvenance("blocklength_n", cfg.blocklength_n, "uses", "Covert transmission blocklength n", "bash_covert_sqrt_law", "IEEE JSAC 2013", "bash2013limits", "Determines O(sqrt(n)) covert bits under KL budget."),
    ]
    return ProvisionResult(
        block_id=META.id,
        seed=cfg.seed,
        elapsed_seconds=time.perf_counter() - t0,
        config=asdict(cfg),
        tensors={"iq_clean": iq_clean, "iq_jammed": iq_jammed, "secrecy_rate": c_sec},
        metrics={
            "mean_secrecy_rate_bps_hz": float(torch.mean(c_sec).item()),
            "secrecy_outage_prob": float(torch.mean((c_sec <= 1e-6).float()).item()),
            "covert_kl_divergence": kl_div,
            "warden_min_detection_error_pinsker": min_warden_error_prob,
            "empirical_jsr_db": float(10.0 * math.log10(max(torch.mean(jammer.abs().pow(2)).item() / max(sig_pow, 1e-12), 1e-12))),
        },
        diagnostics={"device": str(dev), "normalized_channel_power": 1.0},
        parameter_provenance=prov,
        mechanisms_used=list(META.default_mechanisms),
        standard_ids_used=list(META.standard_ids),
    )
