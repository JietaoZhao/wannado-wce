"""Pure-PyTorch OFDM-MIMO Link-Level PHY Transceiver & Coded BLER/BER Validator."""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np
import torch

from wannado.wce.core.utils import resolve_device, set_global_seed

ModulationOrder = Literal["QPSK", "16QAM", "64QAM"]
EqualizerType = Literal["mmse", "zf", "mrc"]
EstimatorType = Literal["genie", "ls", "lmmse"]

_MOD_BITS: dict[str, int] = {"QPSK": 2, "16QAM": 4, "64QAM": 6}


def _gray_pam_levels(bits_per_axis: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate Gray-coded 1D PAM amplitude levels and binary bit table (adjacent levels differ by 1 bit)."""
    m_axis = 1 << bits_per_axis
    bit_patterns = np.zeros((m_axis, bits_per_axis), dtype=np.int64)
    levels = np.zeros(m_axis, dtype=np.float64)
    for idx in range(m_axis):
        gray = idx ^ (idx >> 1)
        for b in range(bits_per_axis):
            bit_patterns[idx, bits_per_axis - 1 - b] = (gray >> b) & 1
        # Map Gray-ordered index -> monotonic odd integers +(m-1), +(m-3), ..., -(m-1)
        levels[idx] = float((m_axis - 1) - 2 * idx)
    return levels, bit_patterns


def build_qam_constellation(
    modulation: ModulationOrder,
    *,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return unit-average-power QAM symbols (M,) complex64 and Gray bit table (M, bps) long."""
    bps = _MOD_BITS[modulation]
    b_axis = bps // 2
    lvl, pam_bits = _gray_pam_levels(b_axis)
    # Cartesian outer product I x Q with Gray ordering (I high bits, Q low bits)
    m_axis = 1 << b_axis
    symbols: list[complex] = []
    bits_rows: list[np.ndarray] = []
    for i_idx in range(m_axis):
        for q_idx in range(m_axis):
            symbols.append(complex(lvl[i_idx], lvl[q_idx]))
            bits_rows.append(np.concatenate([pam_bits[i_idx], pam_bits[q_idx]], axis=0))
    sym_t = torch.tensor(symbols, dtype=torch.complex64, device=device)
    # Normalize E[|s|^2] == 1.0 rigorously
    norm = torch.sqrt(torch.mean(sym_t.abs().pow(2))).clamp(min=1e-12)
    sym_t = sym_t / norm
    bits_t = torch.tensor(np.stack(bits_rows, axis=0), dtype=torch.int64, device=device)
    return sym_t, bits_t


def _build_ldpc_parity_matrix(
    k_info: int = 64,
    m_parity: int = 64,
    col_weight: int = 3,
    seed: int = 2025,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], list[np.ndarray]]:
    """Construct girth >= 6 (zero length-4 cycles) systematic dual-diagonal LDPC matrix H = [H_u | H_p]."""
    rng = np.random.default_rng(seed)
    h_p = np.zeros((m_parity, m_parity), dtype=np.int8)
    for r in range(m_parity):
        h_p[r, r] = 1
        if r > 0:
            h_p[r, r - 1] = 1

    # Track all row pairs that already share a column (including adjacent pairs in dual-diagonal H_p)
    used_pairs: set[tuple[int, int]] = {(r - 1, r) for r in range(1, m_parity)}
    row_deg = np.sum(h_p, axis=1).astype(np.int64)
    h_u = np.zeros((m_parity, k_info), dtype=np.int8)

    for col in range(k_info):
        noise = rng.random(m_parity)
        order = sorted(range(m_parity), key=lambda r: (int(row_deg[r]), float(noise[r])))
        chosen: list[int] = []
        for r in order:
            if all((min(r, p), max(r, p)) not in used_pairs for p in chosen):
                chosen.append(r)
                if len(chosen) == col_weight:
                    break
        if len(chosen) < col_weight:
            # Fallback only for tiny parity dimensions where strict girth-6 is infeasible
            for r in order:
                if r not in chosen:
                    chosen.append(r)
                    if len(chosen) == col_weight:
                        break
        for i in range(len(chosen)):
            for j in range(i + 1, len(chosen)):
                a, b = chosen[i], chosen[j]
                used_pairs.add((a, b) if a < b else (b, a))
        h_u[chosen, col] = 1
        row_deg[chosen] += 1

    h_full = np.concatenate([h_u, h_p], axis=1)
    check_neighbors = [np.flatnonzero(h_full[r]) for r in range(m_parity)]
    var_neighbors = [np.flatnonzero(h_full[:, c]) for c in range(k_info + m_parity)]
    return h_u, h_full, check_neighbors, var_neighbors


_LDPC_CACHE: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, list[np.ndarray], list[np.ndarray]]] = {}


def _get_ldpc(k_info: int = 64, m_parity: int = 64) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], list[np.ndarray]]:
    key = (k_info, m_parity)
    if key not in _LDPC_CACHE:
        _LDPC_CACHE[key] = _build_ldpc_parity_matrix(k_info=k_info, m_parity=m_parity)
    return _LDPC_CACHE[key]


def ldpc_encode(info_bits: np.ndarray, *, k_info: int = 64, m_parity: int = 64) -> np.ndarray:
    """Systematically encode info_bits (B, K) into LDPC codewords (B, K+M) over GF(2)."""
    h_u, _, _, _ = _get_ldpc(k_info=k_info, m_parity=m_parity)
    u = np.asarray(info_bits, dtype=np.int8) & 1
    syndrome_u = (u @ h_u.T) & 1  # (B, M)
    # Dual-diagonal accumulator: p_0 = s_0, p_i = p_{i-1} ^ s_i
    parity = np.cumsum(syndrome_u, axis=-1) & 1
    return np.concatenate([u, parity], axis=-1).astype(np.int8)


def ldpc_decode_min_sum(
    channel_llr: np.ndarray,
    *,
    k_info: int = 64,
    m_parity: int = 64,
    num_iters: int = 12,
    alpha: float = 0.8,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode channel LLRs (LLR > 0 favors bit 0) via Normalized Min-Sum Belief Propagation.

    Returns ``(decoded_codeword_bits, info_bits_hat)``.
    """
    _, h_full, check_neighbors, var_neighbors = _get_ldpc(k_info=k_info, m_parity=m_parity)
    llr = np.clip(np.asarray(channel_llr, dtype=np.float64), -30.0, 30.0)
    b_size, n_code = llr.shape

    # Edge-list vectorized BP over sparse check/variable degree graphs
    r_msg = np.zeros((b_size, m_parity, n_code), dtype=np.float64)
    q_msg = np.broadcast_to(llr[:, None, :], (b_size, m_parity, n_code)).copy() * h_full[None, :, :]
    total_llr = llr.copy()

    for _ in range(num_iters):
        # Check-to-variable update
        r_msg.fill(0.0)
        for c in range(m_parity):
            nbrs = check_neighbors[c]
            vals = q_msg[:, c, nbrs]  # (B, deg_c)
            signs = np.where(vals >= 0.0, 1.0, -1.0)
            abs_v = np.abs(vals)
            total_sign = np.prod(signs, axis=-1, keepdims=True)
            # Smallest two magnitudes along deg_c
            part_idx = np.argpartition(abs_v, 1, axis=-1)
            min1_idx = part_idx[:, :1]
            min1 = np.take_along_axis(abs_v, min1_idx, axis=-1)
            min2 = np.take_along_axis(abs_v, part_idx[:, 1:2], axis=-1)
            mag_excl = np.where(np.arange(len(nbrs))[None, :] == min1_idx, min2, min1)
            r_msg[:, c, nbrs] = alpha * (total_sign * signs) * mag_excl

        total_llr = llr + np.sum(r_msg, axis=1)
        hard = (total_llr < 0.0).astype(np.int8)
        synd = (hard @ h_full.T) & 1
        if not np.any(synd):
            return hard, hard[:, :k_info]

        # Variable-to-check update
        for v in range(n_code):
            vnbrs = var_neighbors[v]
            q_msg[:, vnbrs, v] = total_llr[:, v : v + 1] - r_msg[:, vnbrs, v]

    hard = (total_llr < 0.0).astype(np.int8)
    return hard, hard[:, :k_info]


@dataclass
class LinkLevelPHYConfig:
    """Configuration for pure-PyTorch OFDM-MIMO physical transceiver verification."""

    modulation: ModulationOrder = "16QAM"
    equalizer: EqualizerType = "mmse"
    estimator: EstimatorType = "ls"
    coded: bool = True
    snr_db: float = 15.0
    num_subcarriers: int = 32
    pilot_spacing: int = 4
    num_blocks: int = 48
    cp_length: int | None = None
    seed: int = 42
    device: str = "auto"


@dataclass
class LinkLevelPHYResult:
    """Empirical transceiver telemetry emitted by evaluate_link_level_phy()."""

    modulation: ModulationOrder
    equalizer: EqualizerType
    estimator: EstimatorType
    coded: bool
    snr_db: float
    num_streams: int
    num_rx: int
    num_tx: int
    num_bits_evaluated: int
    num_blocks_evaluated: int
    uncoded_ber: float
    coded_ber: float
    coded_bler: float
    csi_nmse_db: float
    evm_pct: float
    mean_post_sinr_db: float
    effective_goodput_bps_hz: float
    elapsed_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _extract_mimo_channel_tensor(
    source: Any,
    *,
    num_blocks: int,
    num_subcarriers: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    """Normalize ProvisionResult / ComposedWirelessEnv / Tensor into (B, Nr, Nt, Nsc) complex64."""
    from wannado.wce.core.types import ProvisionResult
    from wannado.wce.intake.spec import WCESpec
    from wannado.wce.workspace.composed_env import ComposedWirelessEnv

    if isinstance(source, ComposedWirelessEnv):
        h = source.sample_channel_tensor(num_blocks).to(device=device, dtype=torch.complex64)
    elif isinstance(source, WCESpec):
        env = ComposedWirelessEnv(source, batch_size=max(4, min(num_blocks, 32)), seed=seed)
        h = env.sample_channel_tensor(num_blocks).to(device=device, dtype=torch.complex64)
    elif isinstance(source, ProvisionResult):
        t_sel: torch.Tensor | None = None
        for key in ("h_freq", "h_small", "h", "h_near", "h_eff"):
            if key in source.tensors and isinstance(source.tensors[key], torch.Tensor):
                t_sel = source.tensors[key].detach().to(device=device, dtype=torch.complex64)
                break
        if t_sel is None:
            raise ValueError(f"ProvisionResult '{source.block_id}' contains no complex MIMO channel tensor.")
        h = t_sel
    elif isinstance(source, torch.Tensor):
        h = source.detach().to(device=device, dtype=torch.complex64)
    else:
        h = torch.as_tensor(np.asarray(source), device=device, dtype=torch.complex64)

    if h.ndim == 2:
        # (B, Nt) -> (B, 1, Nt)
        h = h.unsqueeze(1)
    elif h.ndim == 1:
        h = h.view(-1, 1, 1)

    if h.ndim == 5:
        # (B, Nr, Nt, Nsc, T) -> pick middle OFDM slot
        h = h[..., h.shape[-1] // 2]

    if h.ndim == 3:
        # Flat MIMO (B, Nr, Nt) -> broadcast over Nsc
        h = h.unsqueeze(-1).expand(-1, -1, -1, num_subcarriers).clone()
    elif h.ndim == 4:
        if h.shape[-1] != num_subcarriers:
            # Interpolate or tile subcarriers smoothly
            idx = torch.linspace(0, h.shape[-1] - 1, num_subcarriers, device=device).round().long()
            h = h[..., idx]
    else:
        raise ValueError(f"Unsupported channel tensor shape {tuple(h.shape)}; expected 2D-5D.")

    # Normalize entry power to E[|H|^2] = 1.0 if unnormalized, and match requested num_blocks
    mean_pow = float(torch.mean(h.abs().pow(2)).item())
    if mean_pow > 1e-18 and (mean_pow < 0.25 or mean_pow > 4.0):
        h = h / math.sqrt(mean_pow)
    if h.shape[0] != num_blocks:
        reps = (num_blocks + h.shape[0] - 1) // h.shape[0]
        h = h.repeat(reps, 1, 1, 1)[:num_blocks]
    return h


def evaluate_link_level_phy(
    source: Any,
    *,
    config: LinkLevelPHYConfig | None = None,
    **overrides: Any,
) -> LinkLevelPHYResult:
    """Transmit Gray QAM + optional systematic LDPC frames over a WCE MIMO-OFDM channel tensor.

    Performs comb-pilot channel estimation (Genie / LS / LMMSE with band-edge clamping),
    unprecoded spatial MIMO equalization (MMSE / ZF / MRC across ``min(Nr, Nt)`` streams),
    effective-SINR LLR soft demapping, and lightweight Rate-1/2 (K=64, N=128) dual-diagonal
    Normalized Min-Sum LDPC decoding (distinct from full 3GPP TS 38.212 BG1/BG2 quasi-cyclic codes).
    """
    t0 = time.perf_counter()
    cfg = config or LinkLevelPHYConfig()
    if overrides:
        cfg = LinkLevelPHYConfig(**{**asdict(cfg), **overrides})

    rng = set_global_seed(cfg.seed)
    dev = resolve_device(cfg.device)
    bps = _MOD_BITS[cfg.modulation]
    sym_table, bit_table = build_qam_constellation(cfg.modulation, device=dev)

    h_grid = _extract_mimo_channel_tensor(
        source,
        num_blocks=cfg.num_blocks,
        num_subcarriers=cfg.num_subcarriers,
        seed=cfg.seed,
        device=dev,
    )
    b_size, n_rx, n_tx, n_sc = h_grid.shape
    n_streams = min(n_rx, n_tx)
    if cfg.equalizer == "mrc":
        n_streams = 1

    # Subcarrier partitioning: comb pilots vs data subcarriers
    pilot_idx = np.arange(0, n_sc, max(2, int(cfg.pilot_spacing)))
    data_idx = np.array([i for i in range(n_sc) if i not in set(pilot_idx.tolist())], dtype=np.int64)
    if data_idx.size == 0:
        data_idx = np.arange(n_sc, dtype=np.int64)

    # Frame sizing for systematic LDPC (K=64, N=128, Rate 1/2)
    k_info, m_parity = 64, 64
    n_code = k_info + m_parity
    codewords_per_block = max(1, (len(data_idx) * n_streams * bps) // n_code)
    total_coded_bits = codewords_per_block * n_code
    total_data_syms = (total_coded_bits + bps - 1) // bps
    # Pad data symbols to multiple of (n_streams)
    n_re_uses = max(1, (total_data_syms + n_streams - 1) // n_streams)
    pad_syms = n_re_uses * n_streams

    # Generate info bits -> optional LDPC encoding -> pad to full symbols
    if cfg.coded:
        info_bits = rng.integers(0, 2, size=(b_size * codewords_per_block, k_info), dtype=np.int8)
        coded_bits = ldpc_encode(info_bits, k_info=k_info, m_parity=m_parity).reshape(b_size, total_coded_bits)
        tx_bits = coded_bits
    else:
        total_coded_bits = pad_syms * bps
        tx_bits = rng.integers(0, 2, size=(b_size, total_coded_bits), dtype=np.int8)

    if pad_syms * bps > tx_bits.shape[1]:
        pad_width = pad_syms * bps - tx_bits.shape[1]
        tx_bits_padded = np.concatenate([tx_bits, np.zeros((b_size, pad_width), dtype=np.int8)], axis=1)
    else:
        tx_bits_padded = tx_bits[:, : pad_syms * bps]

    bit_groups = tx_bits_padded.reshape(b_size, pad_syms, bps)
    powers = (1 << np.arange(bps - 1, -1, -1, dtype=np.int64))
    # Index lookup matches binary integer of Gray table row? Map directly via bit_table match!
    # Since build_qam_constellation constructs binary indices i_idx, q_idx in order via Gray, compute exact index map:
    bit_key = np.sum(bit_table.cpu().numpy() * powers[None, :], axis=1)
    inv_lut = np.zeros(1 << bps, dtype=np.int64)
    inv_lut[bit_key] = np.arange(1 << bps, dtype=np.int64)
    sym_indices = inv_lut[np.sum(bit_groups.astype(np.int64) * powers[None, None, :], axis=-1)]
    x_syms = sym_table[torch.from_numpy(sym_indices).to(dev)]  # (B, n_re_uses * n_streams)
    x_grid = x_syms.view(b_size, n_re_uses, n_streams)  # (B, U, Ns)

    # Wideband unitary eigenbeamforming precoder when Nt > n_streams so ALL TX antennas contribute array gain
    # while preserving subcarrier phase continuity for causal pilot interpolation (V^H V = I_{n_streams}).
    if n_tx > n_streams:
        h_sc = h_grid.permute(0, 3, 1, 2)  # (B, Nsc, Nr, Nt)
        gram_wb = torch.mean(torch.matmul(h_sc.conj().transpose(-1, -2), h_sc), dim=1)  # (B, Nt, Nt)
        _, eigvecs = torch.linalg.eigh(gram_wb.cpu())
        v_wb = eigvecs[..., -n_streams:].to(device=dev, dtype=torch.complex64)  # (B, Nt, Ns)
        h_eff_grid = torch.einsum("bnmk,bms->bnsk", h_grid, v_wb)  # (B, Nr, Ns, Nsc)
    else:
        h_eff_grid = h_grid[:, :, :n_streams, :]

    # Map resource uses sequentially onto data_idx subcarriers
    sc_assign_np = data_idx[np.arange(n_re_uses) % len(data_idx)]
    sc_assign = torch.as_tensor(sc_assign_np, device=dev, dtype=torch.long)
    h_data_sub = h_eff_grid[:, :, :, sc_assign]  # (B, Nr, Ns, U)
    h_u = h_data_sub.permute(0, 3, 1, 2)  # (B, U, Nr, Ns)

    # Transmit power normalized per stream so total E[||V s||^2] = 1 -> per stream power = 1 / n_streams
    scale_tx = 1.0 / math.sqrt(float(n_streams))
    y_clean = scale_tx * torch.einsum("bunm,bum->bun", h_u, x_grid)  # (B, U, Nr)

    # Optional time-domain OFDM IFFT + Cyclic Prefix + tapped-delay convolution (exposes ISI when max_delay > cp_length)
    if cfg.cp_length is not None:
        cp_len = max(0, int(cfg.cp_length))
        # Derive causal discrete-time channel impulse response from h_eff_grid via IFFT across subcarriers
        # Shift center-DC frequency grid back to standard [0..Nsc-1] DFT ordering before IFFT
        h_cir = torch.fft.ifft(torch.fft.ifftshift(h_eff_grid, dim=-1), dim=-1)  # (B, Nr, Ns, Nsc)
        # Retain causal taps up to Nsc // 2
        l_taps = min(n_sc, max(1, n_sc // 2))
        h_taps = h_cir[..., :l_taps]
        if l_taps > cp_len + 1:
            # Build OFDM frequency symbols on all Nsc subcarriers (data + random previous symbol for inter-symbol interference)
            n_ofdm = (n_re_uses + len(data_idx) - 1) // len(data_idx)
            y_td_chunks: list[torch.Tensor] = []
            prev_td = torch.zeros((b_size, n_streams, n_sc), dtype=torch.complex64, device=dev)
            data_t = torch.as_tensor(data_idx, device=dev, dtype=torch.long)
            for sym_i in range(n_ofdm):
                s0 = sym_i * len(data_idx)
                s1 = min(n_re_uses, s0 + len(data_idx))
                n_here = s1 - s0
                x_fd = torch.zeros((b_size, n_streams, n_sc), dtype=torch.complex64, device=dev)
                x_fd[:, :, data_t[:n_here]] = scale_tx * x_grid[:, s0:s1, :].permute(0, 2, 1)
                x_td = torch.fft.ifft(torch.fft.ifftshift(x_fd, dim=-1), dim=-1)
                if sym_i == 0:
                    # Preceding random OFDM symbol creates realistic ISI tail when max tap delay > cp_len
                    prev_fd = torch.zeros_like(x_fd)
                    prev_fd[:, :, data_t] = scale_tx * sym_table[
                        torch.randint(0, len(sym_table), (b_size, n_streams, len(data_idx)), device=dev)
                    ]
                    prev_td = torch.fft.ifft(torch.fft.ifftshift(prev_fd, dim=-1), dim=-1)
                cp_part = x_td[..., n_sc - cp_len :] if cp_len > 0 else x_td[..., :0]
                tail_len = l_taps - 1
                prefix_prev = prev_td[..., n_sc - tail_len :] if tail_len > 0 else prev_td[..., :0]
                tx_stream = torch.cat([prefix_prev, cp_part, x_td], dim=-1)
                # Linear tapped-delay convolution over valid post-CP window of length n_sc
                y_td = torch.zeros((b_size, n_rx, n_sc), dtype=torch.complex64, device=dev)
                start_idx = tail_len + cp_len
                for tap_l in range(l_taps):
                    sl = tx_stream[..., start_idx - tap_l : start_idx - tap_l + n_sc]
                    y_td = y_td + torch.einsum("bnm,bmk->bnk", h_taps[..., tap_l], sl)
                y_fd = torch.fft.fftshift(torch.fft.fft(y_td, dim=-1), dim=-1)
                y_td_chunks.append(y_fd[:, :, data_t[:n_here]].permute(0, 2, 1))
                prev_td = x_td
            y_clean = torch.cat(y_td_chunks, dim=1)

    snr_lin = 10.0 ** (float(cfg.snr_db) / 10.0)
    # Unit-power baseline channel reference (per-RX noise variance = 1 / snr_lin so TX array gain boosts SNR physically)
    noise_var_rx = 1.0 / max(snr_lin, 1e-12)
    std_rx = math.sqrt(noise_var_rx / 2.0)
    noise_rx = torch.complex(
        torch.randn_like(y_clean.real) * std_rx,
        torch.randn_like(y_clean.imag) * std_rx,
    )
    y_rx = y_clean + noise_rx

    # Causal Channel Estimation: Genie vs Comb-Pilot LS vs True Wiener Frequency-Correlation LMMSE
    if cfg.estimator == "genie":
        h_hat = h_u
        csi_nmse_db = -60.0
    else:
        h_pilots_true = h_eff_grid[:, :, :, pilot_idx]  # (B, Nr, Ns, Np)
        pilot_noise_var = noise_var_rx
        std_p = math.sqrt(pilot_noise_var / 2.0)
        p_noise = torch.complex(
            torch.randn_like(h_pilots_true.real) * std_p,
            torch.randn_like(h_pilots_true.imag) * std_p,
        )
        h_pilots_ls = h_pilots_true + p_noise
        if cfg.estimator == "lmmse" and len(pilot_idx) >= 2:
            # Causal exponential-PDP Wiener LMMSE frequency interpolation estimated purely from noisy pilots:
            # R(dk) = p_sig * exp(j * theta_step * dk) / (1 + j * omega_tau * dk)
            dp = float(pilot_idx[1] - pilot_idx[0])
            raw_pow = float(torch.mean(h_pilots_ls.abs().pow(2)).item())
            p_sig = max(raw_pow - pilot_noise_var, 0.05 * max(raw_pow, 1e-6))
            c1 = torch.mean(h_pilots_ls[..., 1:] * h_pilots_ls[..., :-1].conj())
            rho_mag = min(max(float(c1.abs().item()) / max(p_sig, 1e-9), 0.05), 0.999)
            theta_step = float(torch.angle(c1).item()) / max(dp, 1.0)
            omega_tau = math.sqrt(max(1.0 / (rho_mag * rho_mag) - 1.0, 0.0)) / max(dp, 1.0)

            p_pos = torch.as_tensor(pilot_idx, dtype=torch.float64)
            d_pos = torch.as_tensor(sc_assign_np, dtype=torch.float64)
            diff_pp = p_pos.unsqueeze(1) - p_pos.unsqueeze(0)  # (Np, Np)
            diff_dp = d_pos.unsqueeze(1) - p_pos.unsqueeze(0)  # (U, Np)
            r_pp = p_sig * torch.exp(1j * theta_step * diff_pp) / (1.0 + 1j * omega_tau * diff_pp)
            r_dp = p_sig * torch.exp(1j * theta_step * diff_dp) / (1.0 + 1j * omega_tau * diff_dp)
            eye_p = torch.eye(len(pilot_idx), dtype=torch.complex128)
            w_lmmse = torch.linalg.solve(r_pp + pilot_noise_var * eye_p, r_dp.conj().transpose(-1, -2)).conj().transpose(-1, -2)
            h_hat = torch.einsum("uk,brsk->burs", w_lmmse.to(device=dev, dtype=torch.complex64), h_pilots_ls)
        else:
            # Linear interpolation from pilot_idx onto data_idx
            h_pilots_est = (1.0 / (1.0 + pilot_noise_var)) * h_pilots_ls if cfg.estimator == "lmmse" else h_pilots_ls
            xp = torch.as_tensor(pilot_idx, device=dev, dtype=torch.float32)
            xq = torch.as_tensor(sc_assign_np, device=dev, dtype=torch.float32)
            if len(xp) == 1:
                h_hat = h_pilots_est.expand(-1, -1, -1, n_re_uses).permute(0, 3, 1, 2)
            else:
                idx_hi = torch.searchsorted(xp, xq).clamp(1, len(xp) - 1)
                idx_lo = idx_hi - 1
                alpha_f = (
                    ((xq - xp[idx_lo]) / (xp[idx_hi] - xp[idx_lo]).clamp(min=1e-6))
                    .clamp(0.0, 1.0)
                    .view(1, 1, 1, -1)
                )
                interp = h_pilots_est[..., idx_lo] * (1.0 - alpha_f) + h_pilots_est[..., idx_hi] * alpha_f
                h_hat = interp.permute(0, 3, 1, 2)
        err_pow = float(torch.mean((h_hat - h_u).abs().pow(2)).item())
        true_pow = max(float(torch.mean(h_u.abs().pow(2)).item()), 1e-12)
        csi_nmse_db = 10.0 * math.log10(max(err_pow / true_pow, 1e-10))

    # Spatial MIMO Equalization on scaled effective channel H_eff = scale_tx * H_hat
    h_eff = scale_tx * h_hat  # (B, U, Nr, Ns)
    # Effective channel noise variance per RX entry = noise_var_rx
    if cfg.equalizer == "mrc":
        # (B, U, Nr, 1) -> normalized matched filter w = h^H / ||h||^2
        h_vec = h_eff[..., 0]
        h_norm2 = torch.sum(h_vec.abs().pow(2), dim=-1, keepdim=True).clamp(min=1e-12)
        x_hat = torch.sum(h_vec.conj() * y_rx, dim=-1, keepdim=True) / h_norm2
        # Unbiased post-MRC SINR (including estimation residual if any)
        post_sinr = (h_norm2 / max(noise_var_rx, 1e-12)).clamp(min=1e-6, max=1e6)
    else:
        gram = torch.matmul(h_eff.conj().transpose(-1, -2), h_eff)  # (B, U, Ns, Ns)
        reg = noise_var_rx if cfg.equalizer == "mmse" else 1e-6
        eye = torch.eye(n_streams, device=dev, dtype=torch.complex64).view(1, 1, n_streams, n_streams)
        inv_gram = torch.linalg.inv(gram + reg * eye)  # (B, U, Ns, Ns)
        w_eq = torch.matmul(inv_gram, h_eff.conj().transpose(-1, -2))  # (B, U, Ns, Nr)
        x_raw = torch.matmul(w_eq, y_rx.unsqueeze(-1)).squeeze(-1)  # (B, U, Ns)

        # Bias compensation for LMMSE unbiased symbol estimate: diag(W @ H_eff)
        wh = torch.matmul(w_eq, h_eff)  # (B, U, Ns, Ns)
        diag_gain = torch.diagonal(wh, dim1=-2, dim2=-1).real.clamp(min=1e-5)  # (B, U, Ns)
        x_hat = x_raw / diag_gain

        # True post-equalizer signal vs noise + residual interference power
        interf_pow = torch.sum((wh - torch.diag_embed(torch.diagonal(wh, dim1=-2, dim2=-1))).abs().pow(2), dim=-1)
        noise_enh = noise_var_rx * torch.sum(w_eq.abs().pow(2), dim=-1)
        post_sinr = (diag_gain.pow(2) / (interf_pow + noise_enh).clamp(min=1e-12)).clamp(min=1e-6, max=1e6)

    # Error Vector Magnitude (EVM) against true transmitted constellation points
    evm_num = float(torch.mean((x_hat - x_grid).abs().pow(2)).item())
    evm_pct = 100.0 * math.sqrt(max(evm_num, 0.0))

    # Flatten symbols -> Exact Max-Log Soft-LLR Demapping
    x_flat = x_hat.reshape(b_size, pad_syms)
    sinr_flat = post_sinr.reshape(b_size, pad_syms)
    dist2 = (x_flat.unsqueeze(-1) - sym_table.view(1, 1, -1)).abs().pow(2)  # (B, pad_syms, M)

    llrs = torch.empty((b_size, pad_syms, bps), dtype=torch.float32, device=dev)
    for b_idx in range(bps):
        mask_1 = bit_table[:, b_idx] == 1
        d0 = dist2[..., ~mask_1].min(dim=-1).values
        d1 = dist2[..., mask_1].min(dim=-1).values
        llrs[..., b_idx] = sinr_flat.float() * (d1 - d0)

    llr_np = llrs.reshape(b_size, -1)[:, :total_coded_bits].detach().cpu().numpy()
    hard_rx = (llr_np < 0.0).astype(np.int8)
    uncoded_ber = float(np.mean(hard_rx != tx_bits))

    if cfg.coded:
        llr_blocks = llr_np.reshape(b_size * codewords_per_block, n_code)
        _, info_hat = ldpc_decode_min_sum(llr_blocks, k_info=k_info, m_parity=m_parity)
        bit_errs = info_hat != info_bits
        coded_ber = float(np.mean(bit_errs))
        coded_bler = float(np.mean(np.any(bit_errs, axis=-1)))
        code_rate = float(k_info) / float(n_code)
        num_eval_blocks = b_size * codewords_per_block
        num_eval_bits = info_bits.size
    else:
        coded_ber = uncoded_ber
        coded_bler = float(np.mean(np.any(hard_rx != tx_bits, axis=-1)))
        code_rate = 1.0
        num_eval_blocks = b_size
        num_eval_bits = tx_bits.size

    goodput = float(n_streams * bps * code_rate * max(0.0, 1.0 - coded_bler) * (len(data_idx) / float(n_sc)))
    return LinkLevelPHYResult(
        modulation=cfg.modulation,
        equalizer=cfg.equalizer,
        estimator=cfg.estimator,
        coded=cfg.coded,
        snr_db=float(cfg.snr_db),
        num_streams=n_streams,
        num_rx=n_rx,
        num_tx=n_tx,
        num_bits_evaluated=int(num_eval_bits),
        num_blocks_evaluated=int(num_eval_blocks),
        uncoded_ber=uncoded_ber,
        coded_ber=coded_ber,
        coded_bler=coded_bler,
        csi_nmse_db=float(csi_nmse_db),
        evm_pct=float(evm_pct),
        mean_post_sinr_db=float(10.0 * math.log10(max(float(post_sinr.mean().item()), 1e-12))),
        effective_goodput_bps_hz=goodput,
        elapsed_seconds=time.perf_counter() - t0,
    )


def sweep_link_level_bler(
    source: Any,
    *,
    snr_db_list: Sequence[float] = (0.0, 4.0, 8.0, 12.0, 16.0, 20.0),
    modulation: ModulationOrder = "16QAM",
    equalizer: EqualizerType = "mmse",
    estimator: EstimatorType = "ls",
    coded: bool = True,
    num_blocks: int = 48,
    seed: int = 42,
) -> list[LinkLevelPHYResult]:
    """Run an SNR sweep returning full OFDM-MIMO PHY BER/BLER waterfall telemetry."""
    return [
        evaluate_link_level_phy(
            source,
            snr_db=float(snr),
            modulation=modulation,
            equalizer=equalizer,
            estimator=estimator,
            coded=coded,
            num_blocks=num_blocks,
            seed=seed,
        )
        for snr in snr_db_list
    ]


__all__ = [
    "EqualizerType",
    "EstimatorType",
    "LinkLevelPHYConfig",
    "LinkLevelPHYResult",
    "ModulationOrder",
    "build_qam_constellation",
    "evaluate_link_level_phy",
    "ldpc_decode_min_sum",
    "ldpc_encode",
    "sweep_link_level_bler",
]
