"""3D node geometry, PPP/hexagonal cellular layouts, and far-field / near-field steering vectors."""

from __future__ import annotations

import math

import numpy as np
import torch

from wannado.wce.core.units import wavelength_m


def generate_ppp_positions(
    num_nodes: int | None,
    *,
    area_radius_m: float = 500.0,
    density_per_m2: float | None = None,
    z_height_m: float = 1.5,
    min_distance_m: float = 10.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Sample 3D node positions uniformly in an annular disk [min_distance_m, area_radius_m].

    If ``num_nodes is None`` and ``density_per_m2`` is given, draws N ~ Poisson(lambda * area)
    matching homogeneous Poisson Point Process (PPP) cellular modeling (Andrews et al., 2011).
    """
    gen = rng if rng is not None else np.random.default_rng()
    if num_nodes is None:
        if density_per_m2 is None:
            raise ValueError("Provide either num_nodes or density_per_m2.")
        area = math.pi * (area_radius_m ** 2)
        n = int(gen.poisson(max(density_per_m2 * area, 0.0)))
        n = max(1, n)
    else:
        n = int(num_nodes)

    r_min2 = min(min_distance_m, 0.9 * area_radius_m) ** 2
    r_max2 = area_radius_m ** 2
    r = np.sqrt(gen.uniform(r_min2, r_max2, size=n))
    phi = gen.uniform(0.0, 2.0 * math.pi, size=n)
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    z = np.full(n, float(z_height_m), dtype=np.float64)
    return np.stack([x, y, z], axis=-1)


def hex_multicell_layout(
    num_rings: int = 1,
    *,
    isd_m: float = 500.0,
    bs_height_m: float = 25.0,
) -> np.ndarray:
    """Generate hexagonal cellular BS 3D coordinates (1 ring = 7 cells, 2 rings = 19 cells)."""
    coords: list[tuple[float, float, float]] = [(0.0, 0.0, float(bs_height_m))]
    for q in range(-num_rings, num_rings + 1):
        for r in range(-num_rings, num_rings + 1):
            s = -q - r
            if (q == 0 and r == 0) or max(abs(q), abs(r), abs(s)) > num_rings:
                continue
            x = isd_m * (math.sqrt(3.0) / 2.0 * q + math.sqrt(3.0) * 0.0 * r)
            # Axial hex coordinates -> Cartesian with inter-site distance = isd_m
            x = isd_m * (q + 0.5 * r)
            y = isd_m * (math.sqrt(3.0) / 2.0 * r)
            coords.append((x, y, float(bs_height_m)))
    return np.asarray(coords, dtype=np.float64)


def wrap_around_distances_2d(
    tx_xy: np.ndarray,
    rx_xy: np.ndarray,
    *,
    box_size_m: float,
) -> np.ndarray:
    """Compute minimum-image periodic wrap-around 2D distances on a torus [0, box_size_m)^2."""
    diff = tx_xy[:, None, :2] - rx_xy[None, :, :2]
    diff = diff - box_size_m * np.round(diff / box_size_m)
    return np.linalg.norm(diff, axis=-1)


def compute_elevation_azimuth(
    tx_xyz: np.ndarray | torch.Tensor,
    rx_xyz: np.ndarray | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute (d_2d, d_3d, elevation_rad, azimuth_rad) from Tx to Rx 3D coordinates."""
    tx = torch.as_tensor(tx_xyz, dtype=torch.float32)
    rx = torch.as_tensor(rx_xyz, dtype=torch.float32)
    delta = rx - tx
    dx = delta[..., 0]
    dy = delta[..., 1]
    dz = delta[..., 2]
    d_2d = torch.sqrt(torch.clamp(dx * dx + dy * dy, min=1e-12))
    d_3d = torch.sqrt(torch.clamp(dx * dx + dy * dy + dz * dz, min=1e-12))
    elev = torch.atan2(torch.abs(dz), d_2d)
    azim = torch.atan2(dy, dx)
    return d_2d, d_3d, elev, azim


def ula_farfield_steering(
    num_antennas: int,
    azimuth_rad: torch.Tensor,
    *,
    spacing_lambda: float = 0.5,
) -> torch.Tensor:
    """Far-field ULA unit-norm steering vector a(theta) with shape (..., num_antennas)."""
    idx = torch.arange(num_antennas, device=azimuth_rad.device, dtype=torch.float32)
    phase = -2.0 * math.pi * spacing_lambda * torch.sin(azimuth_rad).unsqueeze(-1) * idx
    vec = torch.complex(torch.cos(phase), torch.sin(phase)) / math.sqrt(max(1, num_antennas))
    return vec.to(torch.complex64)


def upa_farfield_steering(
    nx: int,
    ny: int,
    azimuth_rad: torch.Tensor,
    elevation_rad: torch.Tensor,
    *,
    spacing_lambda: float = 0.5,
) -> torch.Tensor:
    """Far-field 2D UPA unit-norm steering vector a(phi, theta) = a_x kron a_y, shape (..., nx*ny)."""
    sin_theta = torch.sin(elevation_rad)
    u_x = torch.cos(azimuth_rad) * sin_theta
    u_y = torch.sin(azimuth_rad) * sin_theta

    ix = torch.arange(nx, device=azimuth_rad.device, dtype=torch.float32)
    iy = torch.arange(ny, device=azimuth_rad.device, dtype=torch.float32)
    px = -2.0 * math.pi * spacing_lambda * u_x.unsqueeze(-1) * ix
    py = -2.0 * math.pi * spacing_lambda * u_y.unsqueeze(-1) * iy
    ax = torch.complex(torch.cos(px), torch.sin(px))
    ay = torch.complex(torch.cos(py), torch.sin(py))
    kron = (ax.unsqueeze(-1) * ay.unsqueeze(-2)).reshape(*azimuth_rad.shape, nx * ny)
    return (kron / math.sqrt(max(1, nx * ny))).to(torch.complex64)


def upa_element_coordinates(
    nx: int,
    ny: int,
    fc_hz: float,
    *,
    spacing_lambda: float = 0.5,
    center_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> torch.Tensor:
    """Generate 3D physical coordinates (nx*ny, 3) of a centered UPA on the x-y or y-z plane (x-z vertical array)."""
    d = spacing_lambda * wavelength_m(fc_hz)
    gx = (torch.arange(nx, dtype=torch.float32) - 0.5 * (nx - 1)) * d + center_xyz[0]
    gz = (torch.arange(ny, dtype=torch.float32) - 0.5 * (ny - 1)) * d + center_xyz[2]
    xx, zz = torch.meshgrid(gx, gz, indexing="ij")
    yy = torch.full_like(xx, float(center_xyz[1]))
    return torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)


def nearfield_spherical_steering(
    element_coords: torch.Tensor,
    target_coords: torch.Tensor,
    fc_hz: float,
    *,
    include_amplitude_variation: bool = True,
) -> torch.Tensor:
    """Exact spherical-wave near-field steering vector b(r, theta) for XL-MIMO / RIS (Cui & Dai 2022).

    Args:
        element_coords: Array element 3D coordinates, shape ``(N, 3)``.
        target_coords: Scatterer/user 3D coordinates, shape ``(..., 3)``.
        fc_hz: Carrier frequency in Hz.
        include_amplitude_variation: Weight by ``r_center / r_n`` across the array aperture.
    """
    lam = wavelength_m(fc_hz)
    k0 = 2.0 * math.pi / lam
    center = element_coords.mean(dim=0)
    r_ref = torch.norm(target_coords - center, dim=-1, keepdim=True).clamp(min=1e-4)
    # Broadcast distance from every array element n to every target position
    diff = target_coords.unsqueeze(-2) - element_coords
    r_n = torch.norm(diff, dim=-1).clamp(min=1e-4)
    phase = -k0 * (r_n - r_ref)
    amp = (r_ref / r_n) if include_amplitude_variation else torch.ones_like(r_n)
    vec = amp * torch.complex(torch.cos(phase), torch.sin(phase))
    norm_factor = torch.sqrt(torch.sum(amp.pow(2), dim=-1, keepdim=True)).clamp(min=1e-8)
    return (vec / norm_factor).to(torch.complex64)
