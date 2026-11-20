"""Vectorized 3GPP-Style QC-LDPC Coding, 256QAM/2D-DMRS Link Simulation & TS 38.214 AMC + HARQ (Chase/IR)."""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np
import torch

from wannado.wce.core.utils import resolve_device, set_global_seed
from wannado.wce.workspace.link_level_phy import (
    EqualizerType,
    EstimatorType,
    ModulationOrder,
    _extract_mimo_channel_tensor,
    build_qam_constellation,
)

# 3GPP TS 38.214 Table 5.1.3.1-2 (256QAM MCS Index Table 2 — representative subset covering QPSK..256QAM)
NR_MCS_TABLE_256QAM: tuple[dict[str, Any], ...] = (
    {"mcs_index": 0, "modulation": "QPSK", "mod_order": 2, "target_code_rate": 1 / 3, "se": 0.67, "min_sinr_db": -1.5},
    {"mcs_index": 2, "modulation": "QPSK", "mod_order": 2, "target_code_rate": 1 / 2, "se": 1.00, "min_sinr_db": 1.5},
    {"mcs_index": 4, "modulation": "QPSK", "mod_order": 2, "target_code_rate": 2 / 3, "se": 1.33, "min_sinr_db": 4.2},
    {"mcs_index": 7, "modulation": "16QAM", "mod_order": 4, "target_code_rate": 1 / 2, "se": 2.00, "min_sinr_db": 7.2},
    {"mcs_index": 9, "modulation": "16QAM", "mod_order": 4, "target_code_rate": 2 / 3, "se": 2.67, "min_sinr_db": 10.2},
    {"mcs_index": 12, "modulation": "64QAM", "mod_order": 6, "target_code_rate": 1 / 2, "se": 3.00, "min_sinr_db": 12.8},
    {"mcs_index": 15, "modulation": "64QAM", "mod_order": 6, "target_code_rate": 2 / 3, "se": 4.00, "min_sinr_db": 15.8},
    {"mcs_index": 18, "modulation": "64QAM", "mod_order": 6, "target_code_rate": 3 / 4, "se": 4.50, "min_sinr_db": 18.2},
    {"mcs_index": 21, "modulation": "256QAM", "mod_order": 8, "target_code_rate": 2 / 3, "se": 5.33, "min_sinr_db": 21.0},
    {"mcs_index": 24, "modulation": "256QAM", "mod_order": 8, "target_code_rate": 3 / 4, "se": 6.00, "min_sinr_db": 23.8},
    {"mcs_index": 27, "modulation": "256QAM", "mod_order": 8, "target_code_rate": 5 / 6, "se": 6.67, "min_sinr_db": 26.8},
)


def select_nr_mcs_from_sinr(effective_sinr_db: float, *, target_bler: float = 0.1) -> dict[str, Any]:
    """Select the highest 3GPP TS 38.214 Table 5.1.3.1-2 MCS meeting the SINR margin."""
    # Stricter BLER target (<10%) shifts required SINR margin slightly higher
    margin_db = max(0.0, 1.5 * math.log10(0.1 / max(min(float(target_bler), 0.5), 1e-4)))
    eff = float(effective_sinr_db) - margin_db
    chosen = dict(NR_MCS_TABLE_256QAM[0])
    for row in NR_MCS_TABLE_256QAM:
        if eff >= float(row["min_sinr_db"]):
            chosen = dict(row)
    return chosen


def _build_qc_mother_graph(
    k_blocks: int = 4,
    m_blocks: int = 8,
    z_lift: int = 16,
    col_weight: int = 3,
    seed: int = 2026,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct girth >= 6 Quasi-Cyclic lifted dual-diagonal mother LDPC H = [H_u | H_p] (R_mother = 1/3)."""
    k_info = k_blocks * z_lift
    m_parity = m_blocks * z_lift
    rng = np.random.default_rng(seed)

    # Shift-0 block dual-diagonal H_p enables exact O(N) PyTorch cumulative-XOR systematic encoding
    h_p = np.zeros((m_parity, m_parity), dtype=np.int8)
    for rb in range(m_blocks):
        r0 = rb * z_lift
        h_p[r0 : r0 + z_lift, r0 : r0 + z_lift] = np.eye(z_lift, dtype=np.int8)
        if rb > 0:
            c0 = (rb - 1) * z_lift
            h_p[r0 : r0 + z_lift, c0 : c0 + z_lift] = np.eye(z_lift, dtype=np.int8)

    # Track row pairs sharing an active parity or info column to enforce zero length-4 cycles globally
    used_pairs: set[tuple[int, int]] = set()
    for col in range(m_parity):
        rows = np.flatnonzero(h_p[:, col])
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                a, b = int(rows[i]), int(rows[j])
                used_pairs.add((a, b) if a < b else (b, a))

    row_deg = np.sum(h_p, axis=1).astype(np.int64)
    h_u = np.zeros((m_parity, k_info), dtype=np.int8)

    # Assign girth-safe circulant blocks across (m_blocks x k_blocks)
    col_targets = min(col_weight, m_blocks)
    for cb in range(k_blocks):
        noise = rng.random(m_blocks)
        block_order = sorted(range(m_blocks), key=lambda r: (int(np.mean(row_deg[r * z_lift : (r + 1) * z_lift])), float(noise[r])))
        chosen_rbs: list[int] = []
        chosen_shifts: list[int] = []

        for rb in block_order:
            shifts = rng.permutation(z_lift).tolist()
            selected_shift: int | None = None
            for sh in shifts:
                # Check if adding circulant I^{sh} at block (rb, cb) creates any 4-cycle with existing pairs
                ok = True
                new_edges: list[tuple[int, int]] = []
                for zz in range(z_lift):
                    r_idx = rb * z_lift + zz
                    c_local = (zz + sh) % z_lift
                    new_edges.append((r_idx, c_local))
                # Row conflicts within the same local column? Only 1 row per local column for 1 circulant,
                # so check against previously chosen rbs in this SAME column block cb!
                for prev_rb, prev_sh in zip(chosen_rbs, chosen_shifts, strict=True):
                    for zz in range(z_lift):
                        r1 = rb * z_lift + zz
                        c_local = (zz + sh) % z_lift
                        z_prev = (c_local - prev_sh) % z_lift
                        r2 = prev_rb * z_lift + z_prev
                        pair = (r1, r2) if r1 < r2 else (r2, r1)
                        if pair in used_pairs:
                            ok = False
                            break
                    if not ok:
                        break
                if ok:
                    selected_shift = int(sh)
                    break

            if selected_shift is not None:
                chosen_rbs.append(rb)
                chosen_shifts.append(selected_shift)
                if len(chosen_rbs) >= col_targets:
                    break

        # Commit selected circulant blocks to H_u and update forbidden row pairs
        for i, (rb1, sh1) in enumerate(zip(chosen_rbs, chosen_shifts, strict=True)):
            c0 = cb * z_lift
            for zz in range(z_lift):
                r1 = rb1 * z_lift + zz
                c_loc = (zz + sh1) % z_lift
                h_u[r1, c0 + c_loc] = 1
                row_deg[r1] += 1
            for rb2, sh2 in zip(chosen_rbs[i + 1 :], chosen_shifts[i + 1 :], strict=True):
                for zz in range(z_lift):
                    r1 = rb1 * z_lift + zz
                    c_loc = (zz + sh1) % z_lift
                    z2 = (c_loc - sh2) % z_lift
                    r2 = rb2 * z_lift + z2
                    used_pairs.add((r1, r2) if r1 < r2 else (r2, r1))

    h_full = np.concatenate([h_u, h_p], axis=1)
    return h_u, h_p, h_full


_QC_CACHE: dict[tuple[int, int, int], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}


class QCLDPCCodec:
    """3GPP TS 38.212-style Quasi-Cyclic LDPC encoder/decoder with vectorized PyTorch Min-Sum BP & HARQ IR."""

    def __init__(
        self,
        *,
        z_lift: int = 16,
        k_blocks: int = 4,
        m_blocks: int = 8,
        seed: int = 2026,
    ) -> None:
        self.z_lift = max(8, int(z_lift))
        self.k_blocks = max(2, int(k_blocks))
        self.m_blocks = max(4, int(m_blocks))
        self.k_info = self.k_blocks * self.z_lift
        self.m_parity = self.m_blocks * self.z_lift
        self.n_mother = self.k_info + self.m_parity

        key = (self.k_blocks, self.m_blocks, self.z_lift)
        if key not in _QC_CACHE:
            _QC_CACHE[key] = _build_qc_mother_graph(
                k_blocks=self.k_blocks,
                m_blocks=self.m_blocks,
                z_lift=self.z_lift,
                seed=seed,
            )
        h_u_np, _, h_full_np = _QC_CACHE[key]
        self._h_u_np = h_u_np
        self._h_full_np = h_full_np

        # Build vectorized edge-list tensors once per device cache
        rows, cols = np.nonzero(h_full_np)
        self._num_edges = int(rows.size)
        self._edge_c_np = rows.astype(np.int64)
        self._edge_v_np = cols.astype(np.int64)

        row_deg = np.sum(h_full_np, axis=1)
        self._d_max = int(np.max(row_deg))
        # Padded (M, d_max) table mapping each check row slot -> global edge index (or _num_edges sentinel)
        chk_edge_idx = np.full((self.m_parity, self._d_max), self._num_edges, dtype=np.int64)
        chk_valid = np.zeros((self.m_parity, self._d_max), dtype=bool)
        cursor = np.zeros(self.m_parity, dtype=np.int64)
        for e_id, r_id in enumerate(rows):
            slot = int(cursor[r_id])
            chk_edge_idx[r_id, slot] = e_id
            chk_valid[r_id, slot] = True
            cursor[r_id] += 1
        self._chk_edge_idx_np = chk_edge_idx
        self._chk_valid_np = chk_valid
        self._device_tensors: dict[str, dict[str, torch.Tensor]] = {}

    def _get_tensors(self, device: torch.device) -> dict[str, torch.Tensor]:
        dk = str(device)
        if dk not in self._device_tensors:
            self._device_tensors[dk] = {
                "h_u": torch.from_numpy(self._h_u_np.astype(np.float32)).to(device),
                "h_full": torch.from_numpy(self._h_full_np.astype(np.int64)).to(device),
                "edge_c": torch.from_numpy(self._edge_c_np).to(device),
                "edge_v": torch.from_numpy(self._edge_v_np).to(device),
                "chk_edge_idx": torch.from_numpy(self._chk_edge_idx_np).to(device),
                "chk_valid": torch.from_numpy(self._chk_valid_np).to(device),
                "d_arange": torch.arange(self._d_max, device=device).view(1, 1, self._d_max),
            }
        return self._device_tensors[dk]

    def encode(self, info_bits: torch.Tensor) -> torch.Tensor:
        """Systematically encode ``(B, K)`` bits into ``(B, N_mother)`` codeword in pure PyTorch."""
        t = self._get_tensors(info_bits.device)
        u = (info_bits.to(torch.float32).reshape(-1, self.k_info) > 0.5).to(torch.float32)
        syndrome = torch.matmul(u, t["h_u"].t()).to(torch.int64) & 1
        s_blk = syndrome.view(u.shape[0], self.m_blocks, self.z_lift)
        p_blk = torch.cumsum(s_blk, dim=1) & 1
        parity = p_blk.reshape(u.shape[0], self.m_parity)
        return torch.cat([u.to(torch.int8), parity.to(torch.int8)], dim=-1)

    def select_rate_matched_positions(
        self,
        code_rate: float,
        *,
        rv_index: int = 0,
    ) -> np.ndarray:
        """Select bit indices inside mother codeword ``[0..N_mother)`` for rate matching & HARQ IR.

        Natural sequential parity ordering ensures every active dual-diagonal check row ``0..b``
        is 100% closed without hanging punctured nodes, while HARQ IR RV1..RV3 unlock deeper parity blocks.
        """
        r = min(max(float(code_rate), 1.0 / 3.0), 0.90)
        n_target = max(self.k_info + max(8, self.z_lift // 2), min(self.n_mother, round(self.k_info / r)))
        n_par_target = n_target - self.k_info

        if rv_index == 0:
            return np.arange(n_target, dtype=np.int64)
        # Subsequent HARQ Incremental Redundancy transmissions send fresh parity segments first, then chase-wrap
        par_all = np.arange(self.k_info, self.n_mother, dtype=np.int64)
        start_par = (n_par_target + (rv_index - 1) * max(self.z_lift, n_par_target)) % self.m_parity
        rolled = np.roll(par_all, -start_par)
        return rolled[: max(self.z_lift, n_par_target)]

    @torch.no_grad()
    def decode_min_sum(
        self,
        channel_llr: torch.Tensor,
        *,
        num_iters: int = 16,
        alpha: float = 0.8,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Vectorized PyTorch Normalized Min-Sum Belief Propagation decoder (LLR > 0 favors bit 0)."""
        dev = channel_llr.device
        t = self._get_tensors(dev)
        llr = torch.clamp(channel_llr.to(torch.float32).reshape(-1, self.n_mother), -30.0, 30.0)
        b_size = llr.shape[0]
        edge_v = t["edge_v"]
        chk_edge_idx = t["chk_edge_idx"]
        chk_valid = t["chk_valid"]
        h_full = t["h_full"]
        d_arange = t["d_arange"]

        # Active check rows are those whose diagonal parity variable was observed (handles rate-matched puncturing)
        observed_par = torch.any(llr[:, self.k_info :].abs() > 1e-9, dim=0)
        active_rows = torch.nonzero(observed_par, as_tuple=False).reshape(-1)
        if active_rows.numel() == 0:
            active_rows = torch.arange(self.m_parity, device=dev, dtype=torch.long)

        q_edge = llr[:, edge_v].clone()  # (B, E)
        r_edge = torch.zeros_like(q_edge)
        total_llr = llr.clone()
        zeros_col = torch.zeros((b_size, 1), dtype=torch.float32, device=dev)

        for _ in range(max(1, int(num_iters))):
            q_with_pad = torch.cat([q_edge, zeros_col], dim=1)  # (B, E+1)
            q_mat = q_with_pad[:, chk_edge_idx]  # (B, M, d_max)
            signs = torch.where(q_mat >= 0.0, 1.0, -1.0)
            signs = torch.where(chk_valid.unsqueeze(0), signs, 1.0)
            total_sign = torch.prod(signs, dim=-1, keepdim=True)
            out_signs = total_sign * signs

            abs_q = torch.where(chk_valid.unsqueeze(0), q_mat.abs(), 1e6)
            top2_vals, top2_idx = torch.topk(abs_q, k=2, dim=-1, largest=False)
            min1 = top2_vals[..., :1]
            min2 = top2_vals[..., 1:2]
            min1_idx = top2_idx[..., :1]
            mag_excl = torch.where(d_arange == min1_idx, min2, min1)

            r_mat = float(alpha) * out_signs * mag_excl
            # Scatter valid check-to-variable messages back to flat edge buffer (B, E)
            flat_slots = chk_edge_idx[chk_valid]  # length E
            r_vals_flat = r_mat[:, chk_valid]  # (B, E)
            r_edge[:, flat_slots] = r_vals_flat

            r_sum = torch.zeros_like(llr)
            r_sum.index_add_(1, edge_v, r_edge)
            total_llr = llr + r_sum

            hard = (total_llr < 0.0).to(torch.int64)
            synd = (
                torch.matmul(hard.to(torch.float32), h_full[active_rows].to(torch.float32).t()).to(torch.int64)
                & 1
            )
            if not bool(synd.any().item()):
                return hard.to(torch.int8), hard[:, : self.k_info].to(torch.int8)

            q_edge = total_llr[:, edge_v] - r_edge

        hard = (total_llr < 0.0).to(torch.int64)
        return hard.to(torch.int8), hard[:, : self.k_info].to(torch.int8)


@dataclass
class HARQAMCResult:
    """Telemetry returned by 2D-DMRS + QC-LDPC + TS 38.214 AMC + HARQ link simulation."""

    snr_db: float
    mcs_index: int
    modulation: ModulationOrder
    code_rate: float
    harq_mode: str
    max_harq_tx: int
    equalizer: EqualizerType
    estimator: EstimatorType
    num_codewords: int
    initial_bler: float
    residual_bler: float
    residual_ber: float
    mean_transmissions: float
    effective_throughput_bps_hz: float
    csi_nmse_db: float
    rf_evm_pct: float
    elapsed_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _transmit_qam_2d_slot(
    tx_bits_np: np.ndarray,
    h_slot: torch.Tensor,
    *,
    modulation: ModulationOrder,
    snr_db: float,
    equalizer: EqualizerType,
    estimator: EstimatorType,
    dmrs_symbol_indices: tuple[int, int] = (2, 11),
    pilot_spacing: int = 4,
    rf_impairments: Any = None,
) -> tuple[torch.Tensor, float, float]:
    """Map bits onto a 2D (Nsc, T_sym) MIMO-OFDM slot with 2D-DMRS interpolation and return soft LLRs."""
    dev = h_slot.device
    b_size, n_rx, n_tx, n_sc, n_sym = h_slot.shape
    n_streams = 1 if equalizer == "mrc" else min(n_rx, n_tx)
    sym_table, bit_table = build_qam_constellation(modulation, device=dev)
    bps = int(bit_table.shape[1])

    # Identify DMRS grid positions vs data resource elements in (Nsc, n_sym)
    pilot_sc = np.arange(0, n_sc, max(2, int(pilot_spacing)), dtype=np.int64)
    dmrs_syms = tuple(sorted({min(max(0, int(s)), n_sym - 1) for s in dmrs_symbol_indices}))
    if n_sym == 1:
        dmrs_syms = (0,)

    pilot_mask = np.zeros((n_sc, n_sym), dtype=bool)
    for sym_s in dmrs_syms:
        pilot_mask[pilot_sc, sym_s] = True
    data_rc = np.argwhere(~pilot_mask)  # (N_data_re, 2)
    if data_rc.shape[0] == 0:
        data_rc = np.argwhere(np.ones((n_sc, n_sym), dtype=bool))

    n_re_slot = int(data_rc.shape[0])
    n_bits_tx = int(tx_bits_np.shape[1])
    total_syms_needed = (n_bits_tx + bps - 1) // bps
    re_uses_needed = (total_syms_needed + n_streams - 1) // n_streams
    # Tile multi-slot channel realizations smoothly if bit payload exceeds 1 slot capacity
    n_slots_rep = max(1, (re_uses_needed + n_re_slot - 1) // n_re_slot)
    total_data_re = n_slots_rep * n_re_slot
    pad_syms = total_data_re * n_streams

    pad_bits = pad_syms * bps - n_bits_tx
    if pad_bits > 0:
        tx_padded = np.concatenate([tx_bits_np, np.zeros((b_size, pad_bits), dtype=np.int8)], axis=1)
    else:
        tx_padded = tx_bits_np[:, : pad_syms * bps]

    powers = (1 << np.arange(bps - 1, -1, -1, dtype=np.int64))
    bit_key = np.sum(bit_table.cpu().numpy() * powers[None, :], axis=1)
    inv_lut = np.zeros(1 << bps, dtype=np.int64)
    inv_lut[bit_key] = np.arange(1 << bps, dtype=np.int64)
    sym_idx = inv_lut[np.sum(tx_padded.reshape(b_size, pad_syms, bps).astype(np.int64) * powers[None, None, :], axis=-1)]
    x_grid = sym_table[torch.from_numpy(sym_idx).to(dev)].view(b_size, total_data_re, n_streams)

    # Wideband unitary eigenbeamforming precoder when Nt > n_streams
    if n_tx > n_streams:
        h_mat = h_slot.permute(0, 3, 4, 1, 2)  # (B, Nsc, T, Nr, Nt)
        gram_wb = torch.mean(torch.matmul(h_mat.conj().transpose(-1, -2), h_mat), dim=(1, 2))
        _, eigvecs = torch.linalg.eigh(gram_wb.cpu())
        v_wb = eigvecs[..., -n_streams:].to(device=dev, dtype=torch.complex64)
        h_eff_grid = torch.einsum("bnmft,bms->bnsft", h_slot, v_wb)  # (B, Nr, Ns, Nsc, T)
    else:
        h_eff_grid = h_slot[:, :, :n_streams, :, :]

    sc_idx_slot = torch.as_tensor(data_rc[:, 0], device=dev, dtype=torch.long)
    tm_idx_slot = torch.as_tensor(data_rc[:, 1], device=dev, dtype=torch.long)
    h_data_one = h_eff_grid[:, :, :, sc_idx_slot, tm_idx_slot].permute(0, 3, 1, 2)  # (B, N_re, Nr, Ns)
    h_u = h_data_one.repeat(1, n_slots_rep, 1, 1)  # (B, total_data_re, Nr, Ns)

    scale_tx = 1.0 / math.sqrt(float(n_streams))
    y_clean = scale_tx * torch.einsum("bunm,bum->bun", h_u, x_grid)
    if rf_impairments is not None:
        if hasattr(rf_impairments, "forward") or callable(rf_impairments):
            y_rx_sig = rf_impairments(y_clean.transpose(1, 2)).transpose(1, 2)
        else:
            from wannado.wce.rf.impairments import RFImpairmentChain

            y_rx_sig = RFImpairmentChain(rf_impairments)(y_clean.transpose(1, 2)).transpose(1, 2)
    else:
        y_rx_sig = y_clean

    snr_lin = 10.0 ** (float(snr_db) / 10.0)
    noise_var_rx = 1.0 / max(snr_lin, 1e-12)
    std_rx = math.sqrt(noise_var_rx / 2.0)
    y_rx = y_rx_sig + torch.complex(
        torch.randn_like(y_rx_sig.real) * std_rx,
        torch.randn_like(y_rx_sig.imag) * std_rx,
    )

    # 2D DMRS Channel Estimation (frequency comb interpolation + temporal symbol interpolation across T_sym)
    if estimator == "genie":
        h_hat = h_u
        csi_nmse_db = -60.0
    else:
        # Build estimated full 2D grid h_hat_2d (B, Nr, Ns, Nsc, T_sym) from DMRS locations
        sc_q = torch.arange(n_sc, device=dev, dtype=torch.float32)
        p_sc_t = torch.as_tensor(pilot_sc, device=dev, dtype=torch.float32)
        h_at_dmrs_syms: list[torch.Tensor] = []
        for s_dmrs in dmrs_syms:
            h_p_true = h_eff_grid[:, :, :, pilot_sc, s_dmrs]  # (B, Nr, Ns, Np)
            h_p_ls = h_p_true + torch.complex(
                torch.randn_like(h_p_true.real) * math.sqrt(noise_var_rx / 2.0),
                torch.randn_like(h_p_true.imag) * math.sqrt(noise_var_rx / 2.0),
            )
            if estimator == "lmmse":
                h_p_ls = (1.0 / (1.0 + noise_var_rx)) * h_p_ls
            if len(p_sc_t) == 1:
                h_f = h_p_ls.expand(-1, -1, -1, n_sc)
            else:
                hi = torch.searchsorted(p_sc_t, sc_q).clamp(1, len(p_sc_t) - 1)
                lo = hi - 1
                af = ((sc_q - p_sc_t[lo]) / (p_sc_t[hi] - p_sc_t[lo]).clamp(min=1e-6)).clamp(0.0, 1.0).view(1, 1, 1, -1)
                h_f = h_p_ls[..., lo] * (1.0 - af) + h_p_ls[..., hi] * af
            h_at_dmrs_syms.append(h_f)

        if len(dmrs_syms) == 1:
            h_grid_est = h_at_dmrs_syms[0].unsqueeze(-1).expand(-1, -1, -1, -1, n_sym)
        else:
            h_stack = torch.stack(h_at_dmrs_syms, dim=-1)  # (B, Nr, Ns, Nsc, N_dmrs)
            t_p = torch.as_tensor(dmrs_syms, device=dev, dtype=torch.float32)
            t_q = torch.arange(n_sym, device=dev, dtype=torch.float32)
            hi_t = torch.searchsorted(t_p, t_q).clamp(1, len(t_p) - 1)
            lo_t = hi_t - 1
            at = ((t_q - t_p[lo_t]) / (t_p[hi_t] - t_p[lo_t]).clamp(min=1e-6)).clamp(0.0, 1.0).view(1, 1, 1, 1, -1)
            h_grid_est = h_stack[..., lo_t] * (1.0 - at) + h_stack[..., hi_t] * at

        h_hat_one = h_grid_est[:, :, :, sc_idx_slot, tm_idx_slot].permute(0, 3, 1, 2)
        h_hat = h_hat_one.repeat(1, n_slots_rep, 1, 1)
        err_p = float(torch.mean((h_hat - h_u).abs().pow(2)).item())
        true_p = max(float(torch.mean(h_u.abs().pow(2)).item()), 1e-12)
        csi_nmse_db = 10.0 * math.log10(max(err_p / true_p, 1e-10))

    h_eff = scale_tx * h_hat
    if equalizer == "mrc":
        hv = h_eff[..., 0]
        hn2 = torch.sum(hv.abs().pow(2), dim=-1, keepdim=True).clamp(min=1e-12)
        x_hat = torch.sum(hv.conj() * y_rx, dim=-1, keepdim=True) / hn2
        post_sinr = (hn2 / max(noise_var_rx, 1e-12)).clamp(min=1e-6, max=1e6)
    else:
        gram = torch.matmul(h_eff.conj().transpose(-1, -2), h_eff)
        reg = noise_var_rx if equalizer == "mmse" else 1e-6
        eye = torch.eye(n_streams, device=dev, dtype=torch.complex64).view(1, 1, n_streams, n_streams)
        w_eq = torch.matmul(torch.linalg.inv(gram + reg * eye), h_eff.conj().transpose(-1, -2))
        x_raw = torch.matmul(w_eq, y_rx.unsqueeze(-1)).squeeze(-1)
        wh = torch.matmul(w_eq, h_eff)
        diag_gain = torch.diagonal(wh, dim1=-2, dim2=-1).real.clamp(min=1e-5)
        x_hat = x_raw / diag_gain
        interf_p = torch.sum((wh - torch.diag_embed(torch.diagonal(wh, dim1=-2, dim2=-1))).abs().pow(2), dim=-1)
        noise_enh = noise_var_rx * torch.sum(w_eq.abs().pow(2), dim=-1)
        post_sinr = (diag_gain.pow(2) / (interf_p + noise_enh).clamp(min=1e-12)).clamp(min=1e-6, max=1e6)

    evm_pct = 100.0 * math.sqrt(max(float(torch.mean((x_hat - x_grid).abs().pow(2)).item()), 0.0))
    x_flat = x_hat.reshape(b_size, pad_syms)
    sinr_flat = post_sinr.reshape(b_size, pad_syms)
    dist2 = (x_flat.unsqueeze(-1) - sym_table.view(1, 1, -1)).abs().pow(2)
    llrs = torch.empty((b_size, pad_syms, bps), dtype=torch.float32, device=dev)
    for b_idx in range(bps):
        m1 = bit_table[:, b_idx] == 1
        d0 = dist2[..., ~m1].min(dim=-1).values
        d1 = dist2[..., m1].min(dim=-1).values
        llrs[..., b_idx] = sinr_flat.float() * (d1 - d0)

    return llrs.reshape(b_size, -1)[:, :n_bits_tx], float(csi_nmse_db), float(evm_pct)


def evaluate_harq_amc_link(
    source: Any,
    *,
    snr_db: float = 12.0,
    mcs_index: int | None = None,
    max_harq_tx: int = 4,
    harq_mode: Literal["chase", "ir"] = "chase",
    target_bler: float = 0.1,
    num_blocks: int = 32,
    num_subcarriers: int = 32,
    num_symbols: int = 14,
    dmrs_symbol_indices: tuple[int, int] = (2, 11),
    pilot_spacing: int = 4,
    equalizer: EqualizerType = "mmse",
    estimator: EstimatorType = "lmmse",
    z_lift: int = 16,
    rf_impairments: Any = None,
    seed: int = 42,
    device: str = "auto",
) -> HARQAMCResult:
    """Evaluate 3GPP-style 2D-DMRS OFDM-MIMO link with TS 38.214 AMC, QC-LDPC, and HARQ Chase/IR combining."""
    t0 = time.perf_counter()
    set_global_seed(seed)
    dev = resolve_device(device)

    # Extract 5D time-frequency channel (B, Nr, Nt, Nsc, T).
    # Note: `tr38901_cdl_tdl` outputs slot-spaced axis T (slot_duration_ms per index), so map each
    # HARQ transmission across consecutive slots interpolated over `num_symbols` intra-slot OFDM symbols.
    h5_slots: torch.Tensor | None = None
    if isinstance(source, torch.Tensor) and source.ndim == 5:
        h5 = source.detach().to(device=dev, dtype=torch.complex64)
        if h5.shape[3] != num_subcarriers:
            idx_f = torch.linspace(0, h5.shape[3] - 1, num_subcarriers, device=dev).round().long()
            h5 = h5[:, :, :, idx_f, :]
        if h5.shape[4] != num_symbols:
            idx_t = torch.linspace(0, h5.shape[4] - 1, num_symbols, device=dev).round().long()
            h5 = h5[:, :, :, :, idx_t]
        if h5.shape[0] != num_blocks:
            reps = (num_blocks + h5.shape[0] - 1) // h5.shape[0]
            h5 = h5.repeat(reps, 1, 1, 1, 1)[:num_blocks]
    elif hasattr(source, "tensors") and isinstance(source.tensors.get("h_freq"), torch.Tensor) and source.tensors["h_freq"].ndim == 5:
        h5_raw = source.tensors["h_freq"].detach().to(device=dev, dtype=torch.complex64)
        idx_f = torch.linspace(0, h5_raw.shape[3] - 1, num_subcarriers, device=dev).round().long()
        h5_f = h5_raw[:, :, :, idx_f, :]
        if h5_f.shape[0] != num_blocks:
            reps = (num_blocks + h5_f.shape[0] - 1) // h5_f.shape[0]
            h5_f = h5_f.repeat(reps, 1, 1, 1, 1)[:num_blocks]
        if hasattr(source, "metrics") and "slot_duration_ms" in source.metrics:
            h5_slots = h5_f
            h5 = h5_f[..., :1].expand(-1, -1, -1, -1, num_symbols)
        else:
            idx_t = torch.linspace(0, h5_f.shape[4] - 1, num_symbols, device=dev).round().long()
            h5 = h5_f[:, :, :, :, idx_t]
    else:
        h4 = _extract_mimo_channel_tensor(
            source,
            num_blocks=num_blocks,
            num_subcarriers=num_subcarriers,
            seed=seed,
            device=dev,
        )
        # Synthesize gentle intra-slot temporal phase evolution across the 14 OFDM symbols when static 4D provided
        sym_phase = torch.linspace(0.0, 0.15 * math.pi, num_symbols, device=dev, dtype=torch.float32).view(1, 1, 1, 1, -1)
        rot = torch.complex(torch.cos(sym_phase), torch.sin(sym_phase))
        h5 = (h4.unsqueeze(-1) * rot).to(torch.complex64)

    n_streams = 1 if equalizer == "mrc" else min(int(h5.shape[1]), int(h5.shape[2]))
    # Array gain boosts effective post-combining SINR for TS 38.214 outer-loop MCS selection
    eff_sinr_est_db = float(snr_db) + 10.0 * math.log10(max(float(h5.shape[1] * h5.shape[2]) / max(n_streams, 1), 1.0))
    if mcs_index is not None:
        by_idx = {int(r["mcs_index"]): dict(r) for r in NR_MCS_TABLE_256QAM}
        mcs = by_idx.get(int(mcs_index), select_nr_mcs_from_sinr(eff_sinr_est_db, target_bler=target_bler))
    else:
        mcs = select_nr_mcs_from_sinr(eff_sinr_est_db, target_bler=target_bler)

    modulation: ModulationOrder = mcs["modulation"]
    code_rate = float(mcs["target_code_rate"])

    codec = QCLDPCCodec(z_lift=z_lift, k_blocks=4, m_blocks=8, seed=seed)
    info_bits = torch.randint(0, 2, (num_blocks, codec.k_info), device=dev, dtype=torch.int8)
    mother_cw = codec.encode(info_bits)  # (B, N_mother)

    llr_buffer = torch.zeros((num_blocks, codec.n_mother), dtype=torch.float32, device=dev)
    unresolved = torch.ones(num_blocks, dtype=torch.bool, device=dev)
    tx_counts = torch.zeros(num_blocks, dtype=torch.int64, device=dev)
    initial_bler = 1.0
    csi_nmse_list: list[float] = []
    evm_list: list[float] = []

    max_tx = max(1, int(max_harq_tx))
    last_info_hat = torch.zeros_like(info_bits)

    for tx_idx in range(max_tx):
        active_idx = torch.nonzero(unresolved, as_tuple=False).reshape(-1)
        if active_idx.numel() == 0:
            break
        rv = 0 if harq_mode == "chase" else tx_idx
        pos_np = codec.select_rate_matched_positions(code_rate, rv_index=rv)
        pos_t = torch.as_tensor(pos_np, device=dev, dtype=torch.long)

        tx_bits = mother_cw[active_idx][:, pos_t].detach().cpu().numpy()
        if h5_slots is not None:
            n_tot_slots = int(h5_slots.shape[4])
            s0 = (tx_idx * 2) % n_tot_slots
            s1 = (s0 + 1) % n_tot_slots
            w_sym = torch.linspace(0.0, 1.0, num_symbols, device=dev, dtype=torch.float32).view(1, 1, 1, 1, -1)
            h_slot_tx = (
                h5_slots[active_idx, :, :, :, s0 : s0 + 1] * (1.0 - w_sym)
                + h5_slots[active_idx, :, :, :, s1 : s1 + 1] * w_sym
            ).to(torch.complex64)
            pow_tx = float(torch.mean(h_slot_tx.abs().pow(2)).item())
            if pow_tx > 1e-12:
                h_slot_tx = (h_slot_tx / math.sqrt(pow_tx)).to(torch.complex64)
        else:
            h_slot_tx = h5[active_idx]
        llrs_rx, csi_nmse_db, evm_pct = _transmit_qam_2d_slot(
            tx_bits,
            h_slot_tx,
            modulation=modulation,
            snr_db=snr_db,
            equalizer=equalizer,
            estimator=estimator,
            dmrs_symbol_indices=dmrs_symbol_indices,
            pilot_spacing=pilot_spacing,
            rf_impairments=rf_impairments,
        )
        csi_nmse_list.append(csi_nmse_db)
        evm_list.append(evm_pct)

        # Accumulate soft LLRs across HARQ transmissions (Chase combining OR Incremental Redundancy)
        llr_buffer[active_idx[:, None], pos_t[None, :]] += llrs_rx
        tx_counts[active_idx] += 1

        _, info_hat_sub = codec.decode_min_sum(llr_buffer[active_idx])
        last_info_hat[active_idx] = info_hat_sub
        block_err = torch.any(info_hat_sub != info_bits[active_idx], dim=-1)
        if tx_idx == 0:
            initial_bler = float(block_err.float().mean().item())
        unresolved[active_idx[~block_err]] = False

    residual_err_bits = last_info_hat != info_bits
    residual_ber = float(residual_err_bits.float().mean().item())
    residual_bler = float(torch.any(residual_err_bits, dim=-1).float().mean().item())
    mean_tx = float(tx_counts.float().mean().item())

    bps = {"QPSK": 2, "16QAM": 4, "64QAM": 6, "256QAM": 8}[modulation]
    overhead_ok = (max(1, num_symbols - len(dmrs_symbol_indices)) / float(max(1, num_symbols)))
    eff_tp = float(n_streams * bps * code_rate * max(0.0, 1.0 - residual_bler) * overhead_ok / max(mean_tx, 1.0))

    return HARQAMCResult(
        snr_db=float(snr_db),
        mcs_index=int(mcs["mcs_index"]),
        modulation=modulation,
        code_rate=code_rate,
        harq_mode=harq_mode,
        max_harq_tx=max_tx,
        equalizer=equalizer,
        estimator=estimator,
        num_codewords=num_blocks,
        initial_bler=initial_bler,
        residual_bler=residual_bler,
        residual_ber=residual_ber,
        mean_transmissions=mean_tx,
        effective_throughput_bps_hz=eff_tp,
        csi_nmse_db=float(np.mean(csi_nmse_list)) if csi_nmse_list else -60.0,
        rf_evm_pct=float(np.mean(evm_list)) if evm_list else 0.0,
        elapsed_seconds=time.perf_counter() - t0,
    )


__all__ = [
    "NR_MCS_TABLE_256QAM",
    "HARQAMCResult",
    "QCLDPCCodec",
    "evaluate_harq_amc_link",
    "select_nr_mcs_from_sinr",
]
