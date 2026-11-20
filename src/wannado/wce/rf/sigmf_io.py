"""SigMF v1.x Real-World IQ Capture Interop & Sim-to-Real PSD Comparison."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch


@dataclass
class SigMFMetadata:
    """Structured SigMF ``core:*`` metadata parsed from ``.sigmf-meta``."""

    datatype: str = "cf32_le"
    sample_rate_hz: float = 30.72e6
    center_freq_hz: float = 0.0
    description: str = ""
    author: str = "wannado-wce"
    sha512: str = ""
    num_samples: int = 0
    extra_global: dict[str, Any] = field(default_factory=dict)
    captures: list[dict[str, Any]] = field(default_factory=list)
    annotations: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _resolve_sigmf_paths(path: str | Path) -> tuple[Path, Path]:
    p = Path(path).resolve()
    name = p.name
    if name.endswith(".sigmf-meta"):
        stem = name[: -len(".sigmf-meta")]
        return p.parent / f"{stem}.sigmf-meta", p.parent / f"{stem}.sigmf-data"
    if name.endswith(".sigmf-data"):
        stem = name[: -len(".sigmf-data")]
        return p.parent / f"{stem}.sigmf-meta", p.parent / f"{stem}.sigmf-data"
    return p.with_name(f"{p.name}.sigmf-meta"), p.with_name(f"{p.name}.sigmf-data")


def write_sigmf_recording(
    iq: torch.Tensor | np.ndarray,
    base_path: str | Path,
    *,
    sample_rate_hz: float = 30.72e6,
    center_freq_hz: float = 0.0,
    datatype: str = "cf32_le",
    description: str = "",
    author: str = "wannado-wce",
    annotations: list[dict[str, Any]] | None = None,
    extra_global: dict[str, Any] | None = None,
) -> tuple[Path, Path]:
    """Write a pair of SigMF v1.x ``.sigmf-meta`` (with ``core:sha512``) and ``.sigmf-data`` binary files."""
    if datatype not in ("cf32_le", "cf64_le", "ci16_le"):
        raise ValueError(f"Unsupported SigMF datatype '{datatype}'. Supported: cf32_le, cf64_le, ci16_le.")

    meta_path, data_path = _resolve_sigmf_paths(base_path)
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    arr = iq.detach().cpu().numpy() if isinstance(iq, torch.Tensor) else np.asarray(iq)
    flat = np.asarray(arr, dtype=np.complex128 if datatype == "cf64_le" else np.complex64).reshape(-1)
    num_samples = int(flat.size)

    if datatype == "cf32_le":
        interleaved = np.empty(2 * num_samples, dtype="<f4")
        interleaved[0::2] = flat.real.astype(np.float32)
        interleaved[1::2] = flat.imag.astype(np.float32)
        raw_bytes = interleaved.tobytes()
    elif datatype == "cf64_le":
        interleaved64 = np.empty(2 * num_samples, dtype="<f8")
        interleaved64[0::2] = flat.real.astype(np.float64)
        interleaved64[1::2] = flat.imag.astype(np.float64)
        raw_bytes = interleaved64.tobytes()
    else:
        max_abs = max(float(np.max(np.abs([flat.real, flat.imag]))) if num_samples > 0 else 1.0, 1e-12)
        scale = 32767.0 / max_abs if max_abs > 1.0 else 32767.0
        interleaved16 = np.empty(2 * num_samples, dtype="<i2")
        interleaved16[0::2] = np.clip(np.round(flat.real * scale), -32768, 32767).astype("<i2")
        interleaved16[1::2] = np.clip(np.round(flat.imag * scale), -32768, 32767).astype("<i2")
        raw_bytes = interleaved16.tobytes()

    sha512_hex = hashlib.sha512(raw_bytes).hexdigest()
    data_path.write_bytes(raw_bytes)

    glob: dict[str, Any] = {
        "core:datatype": datatype,
        "core:sample_rate": float(sample_rate_hz),
        "core:version": "1.2.0",
        "core:author": author,
        "core:sha512": sha512_hex,
        "core:num_samples": num_samples,
    }
    if description:
        glob["core:description"] = description
    if extra_global:
        for k, v in extra_global.items():
            glob[k] = v

    meta_doc: dict[str, Any] = {
        "global": glob,
        "captures": [
            {
                "core:sample_start": 0,
                "core:frequency": float(center_freq_hz),
            }
        ],
        "annotations": list(annotations) if annotations else [],
    }
    meta_path.write_text(json.dumps(meta_doc, indent=2, ensure_ascii=False), encoding="utf-8")
    return meta_path, data_path


def read_sigmf_recording(
    path: str | Path,
    *,
    verify_sha512: bool = True,
    max_samples: int | None = None,
) -> tuple[torch.Tensor, SigMFMetadata]:
    """Load a SigMF ``.sigmf-meta`` + ``.sigmf-data`` recording into ``torch.complex64`` with integrity validation."""
    meta_path, data_path = _resolve_sigmf_paths(path)
    if not meta_path.is_file():
        raise FileNotFoundError(f"SigMF metadata file not found: {meta_path}")
    if not data_path.is_file():
        raise FileNotFoundError(f"SigMF data file not found: {data_path}")

    doc = json.loads(meta_path.read_text(encoding="utf-8"))
    glob = dict(doc.get("global") or {})
    captures = list(doc.get("captures") or [])
    annotations = list(doc.get("annotations") or [])

    datatype = str(glob.get("core:datatype", "cf32_le"))
    expected_sha = str(glob.get("core:sha512", ""))
    raw_bytes = data_path.read_bytes()

    if verify_sha512 and expected_sha:
        actual_sha = hashlib.sha512(raw_bytes).hexdigest()
        if actual_sha != expected_sha:
            raise ValueError(
                f"SigMF SHA-512 integrity mismatch for {data_path.name}: "
                f"expected={expected_sha[:16]}..., actual={actual_sha[:16]}..."
            )

    if datatype == "cf32_le":
        arr = np.frombuffer(raw_bytes, dtype="<f4")
        iq_np = (arr[0::2] + 1j * arr[1::2]).astype(np.complex64)
    elif datatype == "cf64_le":
        arr64 = np.frombuffer(raw_bytes, dtype="<f8")
        iq_np = (arr64[0::2] + 1j * arr64[1::2]).astype(np.complex64)
    elif datatype == "ci16_le":
        arr16 = np.frombuffer(raw_bytes, dtype="<i2").astype(np.float32) / 32768.0
        iq_np = (arr16[0::2] + 1j * arr16[1::2]).astype(np.complex64)
    else:
        raise ValueError(f"Unsupported SigMF core:datatype '{datatype}'.")

    if max_samples is not None:
        iq_np = iq_np[: max(0, int(max_samples))]

    center_freq = float((captures[0] or {}).get("core:frequency", 0.0)) if captures else 0.0
    known_keys = {"core:datatype", "core:sample_rate", "core:author", "core:description", "core:sha512", "core:num_samples"}
    meta = SigMFMetadata(
        datatype=datatype,
        sample_rate_hz=float(glob.get("core:sample_rate", 30.72e6)),
        center_freq_hz=center_freq,
        description=str(glob.get("core:description", "")),
        author=str(glob.get("core:author", "")),
        sha512=expected_sha,
        num_samples=int(iq_np.size),
        extra_global={k: v for k, v in glob.items() if k not in known_keys},
        captures=captures,
        annotations=annotations,
    )
    return torch.from_numpy(iq_np.copy()), meta


def _welch_normalized_psd(iq: torch.Tensor, nfft: int) -> np.ndarray:
    z = iq.detach().cpu().reshape(-1).to(torch.complex64).numpy()
    if z.size < nfft:
        z = np.pad(z, (0, nfft - z.size))
    step = max(1, nfft // 2)
    win = np.hanning(nfft)
    win = win / max(float(np.linalg.norm(win)), 1e-12)
    acc = np.zeros(nfft, dtype=np.float64)
    count = 0
    for start in range(0, z.size - nfft + 1, step):
        seg = z[start : start + nfft] * win
        spec = np.fft.fftshift(np.fft.fft(seg))
        acc += np.abs(spec) ** 2
        count += 1
    psd = acc / max(count, 1)
    return psd / max(float(np.sum(psd)), 1e-18)


def compare_iq_psd_and_stats(
    iq_ref: torch.Tensor,
    iq_test: torch.Tensor,
    *,
    nfft: int = 256,
) -> dict[str, float]:
    """Compare simulated vs captured IQ recordings via normalized PSD correlation, power delta, PAPR & amplitude quantiles."""
    r = iq_ref.detach().cpu().reshape(-1).to(torch.complex64)
    t = iq_test.detach().cpu().reshape(-1).to(torch.complex64)

    psd_r = _welch_normalized_psd(r, nfft=max(16, int(nfft)))
    psd_t = _welch_normalized_psd(t, nfft=max(16, int(nfft)))
    psd_rc = psd_r - float(np.mean(psd_r))
    psd_tc = psd_t - float(np.mean(psd_t))
    denom = math.sqrt(max(float(np.sum(psd_rc ** 2) * np.sum(psd_tc ** 2)), 1e-24))
    psd_corr = float(np.sum(psd_rc * psd_tc) / denom) if denom > 1e-18 else 1.0

    p_r = max(float(torch.mean(r.abs().pow(2)).item()), 1e-18)
    p_t = max(float(torch.mean(t.abs().pow(2)).item()), 1e-18)
    mean_power_delta_db = 10.0 * math.log10(p_t / p_r)

    papr_r = 10.0 * math.log10(max(float(torch.amax(r.abs().pow(2)).item()), 1e-18) / p_r)
    papr_t = 10.0 * math.log10(max(float(torch.amax(t.abs().pow(2)).item()), 1e-18) / p_t)

    qs = np.linspace(0.05, 0.95, 19)
    amp_r = np.quantile((r.abs() / math.sqrt(p_r)).numpy(), qs)
    amp_t = np.quantile((t.abs() / math.sqrt(p_t)).numpy(), qs)
    amp_quantile_mae = float(np.mean(np.abs(amp_r - amp_t)))

    return {
        "psd_pearson_corr": psd_corr,
        "mean_power_delta_db": mean_power_delta_db,
        "papr_ref_db": papr_r,
        "papr_test_db": papr_t,
        "papr_delta_db": papr_t - papr_r,
        "amplitude_quantile_mae": amp_quantile_mae,
    }


__all__ = [
    "SigMFMetadata",
    "compare_iq_psd_and_stats",
    "read_sigmf_recording",
    "write_sigmf_recording",
]
