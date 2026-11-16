"""Authoritative open-source wireless ecosystem registry & Just-in-Time (JIT) environment planner."""

from __future__ import annotations

import importlib.util
import platform
import shutil
from dataclasses import asdict, dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class EcosystemTool:
    """Metadata and JIT installation/fusion blueprint for an open-source wireless simulator/library."""

    id: str
    name: str
    tier: str
    language_backend: str
    category: str
    python_import_name: str | None
    cli_binary_name: str | None
    install_command: str
    repo_or_url: str
    license_name: str
    os_gpu_constraints: str
    disk_footprint_hint: str
    when_to_use: str
    when_not_to_use: str
    fusion_with_wce_lab: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


ECOSYSTEM_REGISTRY: dict[str, EcosystemTool] = {
    "wannado_wce_builtin": EcosystemTool(
        id="wannado_wce_builtin",
        name="wannado-wce Built-in Standards Physics & Traffic Kernel",
        tier="Tier 0 (Built-in)",
        language_backend="Python + PyTorch (complex64) + NumPy + SciPy",
        category="channel_phy_mac_6g",
        python_import_name="wannado.wce",
        cli_binary_name="wannado-wce",
        install_command="Already installed with wannado-wce",
        repo_or_url="https://github.com/JietaoZhao/wannado-wce",
        license_name="MIT",
        os_gpu_constraints="Cross-platform (CPU / CUDA / macOS); zero external build.",
        disk_footprint_hint="< 5 MB",
        when_to_use="Instant 3GPP/ITU/6G physics, differentiable baseband channels, traffic/MAC queues, and DRL/Adv handoff.",
        when_not_to_use="Site-specific 3D CAD building ray tracing or real-time USRP C++ L1/L2/L3 protocol stacks.",
        fusion_with_wce_lab="Default execution core directly inside ./wce_lab/interface.py.",
    ),
    "sionna": EcosystemTool(
        id="sionna",
        name="NVIDIA Sionna (PHY & SYS)",
        tier="Tier 1 (PyPI JIT)",
        language_backend="Python + PyTorch (Sionna 1.0+/2.x; legacy 0.x was TensorFlow)",
        category="link_phy_mimo_coding",
        python_import_name="sionna",
        cli_binary_name=None,
        install_command="pip install sionna-no-rt  # or `pip install sionna` with Ray Tracer",
        repo_or_url="https://github.com/NVlabs/sionna",
        license_name="Apache-2.0",
        os_gpu_constraints="Python >= 3.11 + PyTorch >= 2.9 recommended; Linux CUDA best for large batches.",
        disk_footprint_hint="~300 MB - 1.5 GB (with dependencies)",
        when_to_use="Full LDPC/Polar FEC chains, bit-level OFDM-MIMO BLER curves, and GPU-differentiable link simulation.",
        when_not_to_use="Lightweight analytic/statistical baselines or TCP/IP packet-level network routing.",
        fusion_with_wce_lab="Wrap Sionna PHY blocks inside WirelessEnvInterface.channel_callable for bit-exact BLER evaluation.",
    ),
    "sionna_rt": EcosystemTool(
        id="sionna_rt",
        name="NVIDIA Sionna RT (3D Differentiable Ray Tracer)",
        tier="Tier 1 (PyPI JIT)",
        language_backend="Python + Mitsuba 3 / Dr.Jit + PyTorch/NumPy/JAX",
        category="ray_tracing",
        python_import_name="sionna.rt",
        cli_binary_name=None,
        install_command="pip install sionna-rt",
        repo_or_url="https://github.com/NVlabs/sionna-rt",
        license_name="Apache-2.0",
        os_gpu_constraints="GPU OptiX ray tracing requires Linux + NVIDIA GPU; Windows/WSL2/macOS falls back to CPU.",
        disk_footprint_hint="~500 MB + 3D scene XML/PLY meshes",
        when_to_use="Site-specific 3D city/indoor building reflections, diffraction, and RIS electromagnetic ray tracing.",
        when_not_to_use="Quick statistical 3GPP TR 38.901 Monte Carlo sweeps without 3D scene geometry.",
        fusion_with_wce_lab="Export ray paths/CIR from sionna.rt into ./wce_lab/cache/ and expose as WirelessEnvInterface tensors.",
    ),
    "deepmimo": EcosystemTool(
        id="deepmimo",
        name="DeepMIMO (v4+)",
        tier="Tier 1 (PyPI JIT)",
        language_backend="Python (PyTorch + Sionna RT integration)",
        category="ray_tracing_dataset",
        python_import_name="deepmimo",
        cli_binary_name=None,
        install_command="pip install deepmimo  # or `pip install 'deepmimo[sionna]'`",
        repo_or_url="https://github.com/DeepMIMO/DeepMIMO",
        license_name="Apache-2.0",
        os_gpu_constraints="Cross-platform Python; datasets download on demand (e.g., asu_campus_3p5).",
        disk_footprint_hint="~50 MB - several GB per scenario",
        when_to_use="Standardized ray-traced MIMO, beam prediction, blockage, and localization benchmarks.",
        when_not_to_use="Pure synthetic stochastic fading without geographic maps.",
        fusion_with_wce_lab="Load scenario channels via deepmimo.load() inside ./wce_lab/interface.py.",
    ),
    "quadriga_lib": EcosystemTool(
        id="quadriga_lib",
        name="QuaDRiGa (`quadriga-lib` C++/Python)",
        tier="Tier 1 (PyPI JIT)",
        language_backend="C++ core with native Python bindings",
        category="gscm_channel_model",
        python_import_name="quadriga_lib",
        cli_binary_name=None,
        install_command="pip install quadriga-lib",
        repo_or_url="https://github.com/fraunhoferhhi/quadriga-lib",
        license_name="Open Source (Fraunhofer HHI)",
        os_gpu_constraints="Cross-platform Python wheels; no MATLAB license required.",
        disk_footprint_hint="~30 MB",
        when_to_use="Continuous trajectory spatial consistency, calibrated antenna patterns, and QuaDRiGa ray-mesh generation.",
        when_not_to_use="Pure PyTorch autograd backpropagation through channel synthesis.",
        fusion_with_wce_lab="Generate trajectory CIR arrays via quadriga_lib and convert to torch.complex64 inside ./wce_lab/.",
    ),
    "hermespy": EcosystemTool(
        id="hermespy",
        name="HermesPy (Barkhausen Institut)",
        tier="Tier 1 (PyPI JIT)",
        language_backend="Python + NumPy/SciPy",
        category="isac_and_multi_rat",
        python_import_name="hermespy",
        cli_binary_name=None,
        install_command="pip install hermespy",
        repo_or_url="https://github.com/Barkhausen-Institut/hermespy",
        license_name="Open Source (Barkhausen Institut)",
        os_gpu_constraints="Cross-platform Python.",
        disk_footprint_hint="~50 MB",
        when_to_use="Multi-RAT link-level simulation, radar/ISAC sensing waveforms, and SDR hardware playback.",
        when_not_to_use="Large multi-cell packet queueing.",
        fusion_with_wce_lab="Bridge HermesPy waveform/radar evaluators into ./wce_lab/interface.py.",
    ),
    "torchsig": EcosystemTool(
        id="torchsig",
        name="TorchSig (PyTorch RF ML Toolkit)",
        tier="Tier 1 (PyPI JIT)",
        language_backend="Python + PyTorch",
        category="rf_ml_modulation",
        python_import_name="torchsig",
        cli_binary_name=None,
        install_command="pip install torchsig",
        repo_or_url="https://github.com/TorchSig/torchsig",
        license_name="MIT",
        os_gpu_constraints="Cross-platform PyTorch.",
        disk_footprint_hint="~100 MB (datasets optional)",
        when_to_use="AMC modulation classification, RF hardware impairments (IQ imbalance, PA nonlinearity), and RF datasets.",
        when_not_to_use="System-level multi-cell scheduling.",
        fusion_with_wce_lab="Pair TorchSig RF waveforms with wannado-wce channels and wannado-adv JSR attacks.",
    ),
    "py3gppchannels": EcosystemTool(
        id="py3gppchannels",
        name="py3gppchannels",
        tier="Tier 1 (PyPI JIT)",
        language_backend="Python + NumPy",
        category="gscm_channel_model",
        python_import_name="py3gppchannels",
        cli_binary_name=None,
        install_command="pip install py3gppchannels",
        repo_or_url="https://github.com/zilligm/py3gppchannels",
        license_name="MIT",
        os_gpu_constraints="Cross-platform Python.",
        disk_footprint_hint="< 10 MB",
        when_to_use="Reference full TR 38.901 SSP/LSP cluster generation cross-check.",
        when_not_to_use="GPU-differentiable PyTorch channels.",
        fusion_with_wce_lab="Optional reference cross-check inside ./wce_lab/.",
    ),
    "scikit_commpy": EcosystemTool(
        id="scikit_commpy",
        name="Scikit-CommPy",
        tier="Tier 1 (PyPI JIT)",
        language_backend="Python + NumPy + SciPy",
        category="classic_phy_coding",
        python_import_name="commpy",
        cli_binary_name=None,
        install_command="pip install scikit-commpy",
        repo_or_url="https://github.com/veeresht/CommPy",
        license_name="BSD-3-Clause",
        os_gpu_constraints="Cross-platform Python.",
        disk_footprint_hint="< 10 MB",
        when_to_use="Classic LDPC/Turbo/Convolutional coding, PSK/QAM modems, and textbook link benchmarks.",
        when_not_to_use="High-throughput GPU tensor simulation.",
        fusion_with_wce_lab="Use as classic FEC encoder/decoder front-end before wannado-wce channel layers.",
    ),
    "mobile_env": EcosystemTool(
        id="mobile_env",
        name="mobile-env (Gymnasium Mobile Network RL)",
        tier="Tier 1 (PyPI JIT)",
        language_backend="Python + Gymnasium",
        category="marl_system_env",
        python_import_name="mobile_env",
        cli_binary_name=None,
        install_command="pip install mobile-env",
        repo_or_url="https://github.com/stefanbschneider/mobile-env",
        license_name="MIT",
        os_gpu_constraints="Cross-platform Python + Gymnasium.",
        disk_footprint_hint="< 10 MB",
        when_to_use="Fast multi-BS cell selection / handover MARL prototyping.",
        when_not_to_use="Detailed 3GPP TR 38.901 MIMO/OFDM physical layers (pair with wannado-wce blocks instead).",
        fusion_with_wce_lab="Expose via WirelessGymEnvAdapter -> wannado-drl EnvInterface.",
    ),
    "nyusim_python": EcosystemTool(
        id="nyusim_python",
        name="NYUSIM Python (NYU WIRELESS)",
        tier="Tier 1 (Git JIT)",
        language_backend="Python",
        category="mmwave_subthz_channel",
        python_import_name="nyusim",
        cli_binary_name=None,
        install_command="git clone https://github.com/nyu-wireless/NYUSIM_Python  # install per NYU WIRELESS release",
        repo_or_url="https://wireless.engineering.nyu.edu/nyusim/",
        license_name="NYU WIRELESS Open Source",
        os_gpu_constraints="Cross-platform Python; specialized for FR1(C), FR3 (7-24 GHz), mmWave & sub-THz.",
        disk_footprint_hint="~50 MB",
        when_to_use="Empirical NYC/outdoor measurement-validated FR3/mmWave/sub-THz spatial lobe channels.",
        when_not_to_use="Sub-6 GHz standard 3GPP TR 38.901 compliance tests.",
        fusion_with_wce_lab="Pull into ./wce_lab/external/ and convert generated CIR snapshots into ChannelBatch.",
    ),
    "pyaerial": EcosystemTool(
        id="pyaerial",
        name="NVIDIA AI Aerial / pyAerial (cuPHY)",
        tier="Tier 3 (GPU Docker JIT)",
        language_backend="C++/CUDA (cuPHY) + Python",
        category="gpu_5g_nr_l1",
        python_import_name="aerial",
        cli_binary_name=None,
        install_command="docker pull nvcr.io/nvidia/aerial/aerial-cuda-accelerated-ran:latest",
        repo_or_url="https://github.com/NVIDIA/aerial-cuda-accelerated-ran",
        license_name="Apache-2.0 / NVIDIA Aerial",
        os_gpu_constraints="Requires Linux + NVIDIA datacenter/RTX GPU + CUDA + Docker.",
        disk_footprint_hint="~10+ GB container",
        when_to_use="Commercial-grade GPU 5G NR L1 PUSCH/PDSCH pipeline with PyTorch neural receiver plugins.",
        when_not_to_use="macOS laptops or CPU-only servers.",
        fusion_with_wce_lab="Generate ./wce_lab/jit_setup.sh container recipe when Linux+CUDA is detected.",
    ),
    "ns3_5glena_ns3gym": EcosystemTool(
        id="ns3_5glena_ns3gym",
        name="ns-3 + 5G-LENA (nr) + ns3-gym",
        tier="Tier 2 (C++ / Docker JIT)",
        language_backend="C++17/20 + ZeroMQ/Protobuf Python Bridge",
        category="packet_network_simulator",
        python_import_name="ns3gym",
        cli_binary_name="ns3",
        install_command="git clone https://gitlab.com/cttc-lena/nr.git && git clone https://github.com/tkn-tub/ns3-gym.git",
        repo_or_url="https://github.com/tkn-tub/ns3-gym",
        license_name="GPLv2",
        os_gpu_constraints="Linux / macOS C++ build toolchain (CMake, g++/clang, ZMQ, Protobuf); 15-40 min compile.",
        disk_footprint_hint="~2 - 5 GB build tree",
        when_to_use="Full TCP/UDP, PDCP, RLC, MAC scheduling, HARQ, and multi-hop packet delay evaluation.",
        when_not_to_use="Differentiable PHY optimization or quick tensor experiments.",
        fusion_with_wce_lab="Emit isolated build recipe + ZMQ Gym bridge inside ./wce_lab/external/ns3/.",
    ),
    "openran_gym_nsoran": EcosystemTool(
        id="openran_gym_nsoran",
        name="OpenRAN Gym & ns-O-RAN",
        tier="Tier 2 (C++ / Docker JIT)",
        language_backend="ns-3 + O-RAN E2SM-KPM/RC + Python xApp",
        category="oran_ric_sim",
        python_import_name=None,
        cli_binary_name=None,
        install_command="git clone https://github.com/wineslab/ns-o-ran-gym.git",
        repo_or_url="https://openrangym.com/",
        license_name="Open Source (WiNES Lab)",
        os_gpu_constraints="Linux Docker heavily recommended.",
        disk_footprint_hint="~5 - 12 GB container",
        when_to_use="O-RAN Near-RT RIC xApp/dApp closed-loop KPM telemetry and E2 control evaluation.",
        when_not_to_use="Pure PHY channel estimation or RIS beamforming math.",
        fusion_with_wce_lab="Generate O-RAN Docker compose + E2 bridge configuration inside ./wce_lab/external/.",
    ),
    "simu5g": EcosystemTool(
        id="simu5g",
        name="Simu5G (OMNeT++ / INET)",
        tier="Tier 2 (C++ JIT)",
        language_backend="C++ (OMNeT++ + INET Framework)",
        category="packet_network_simulator",
        python_import_name=None,
        cli_binary_name="opp_run",
        install_command="git clone https://github.com/Unipisa/Simu5G.git  # inside OMNeT++ + INET workspace",
        repo_or_url="https://github.com/Unipisa/Simu5G",
        license_name="LGPL",
        os_gpu_constraints="Requires OMNeT++ IDE/runtime and INET framework.",
        disk_footprint_hint="~3 - 6 GB",
        when_to_use="5G NR user-plane, UPF/SMF 5GC, dual-connectivity, and V2X network-level scenarios.",
        when_not_to_use="PyTorch-native gradient training.",
        fusion_with_wce_lab="Export .ini scenario & TR 38.901 parameters to Simu5G configs.",
    ),
    "openairinterface": EcosystemTool(
        id="openairinterface",
        name="OpenAirInterface (OAI 5G RAN + 5GC)",
        tier="Tier 3 (Real-Time SDR Stack)",
        language_backend="C / C++",
        category="real_time_ran_sdr",
        python_import_name=None,
        cli_binary_name="nr-softmodem",
        install_command="git clone https://gitlab.eurecom.fr/oai/openairinterface5g.git",
        repo_or_url="https://openairinterface.org/",
        license_name="OAI Public License V1.1",
        os_gpu_constraints="Requires Ubuntu Linux low-latency kernel (+ USRP/RFsimulator); high RAM/CPU.",
        disk_footprint_hint="~5 - 15 GB",
        when_to_use="Over-the-air 3GPP 5G NR hardware tests, RFsimulator digital twins, or FlexRIC xApps.",
        when_not_to_use="macOS laptops or analytical Monte Carlo simulation.",
        fusion_with_wce_lab="Emit FlexRIC / RFsimulator parameter profiles from wce_spec.json.",
    ),
    "srsran_project": EcosystemTool(
        id="srsran_project",
        name="srsRAN Project (5G CU/DU/gNB) & FlexRIC",
        tier="Tier 3 (Real-Time SDR Stack)",
        language_backend="C++",
        category="real_time_ran_sdr",
        python_import_name=None,
        cli_binary_name="gnb",
        install_command="git clone https://github.com/srsRAN/srsRAN_Project.git",
        repo_or_url="https://github.com/srsRAN/srsRAN_Project",
        license_name="AGPLv3",
        os_gpu_constraints="Linux + UHD/ZMQ/USRP hardware recommended.",
        disk_footprint_hint="~2 - 6 GB",
        when_to_use="Modular O-RAN split 7.2 / ZMQ virtual RF 5G gNB testbeds.",
        when_not_to_use="Pure link/channel math studies.",
        fusion_with_wce_lab="Emit YAML gNB cell numerology & bandwidth config synced with TS 38.211.",
    ),
    "vienna_5g": EcosystemTool(
        id="vienna_5g",
        name="Vienna 5G Link & System Level Simulators",
        tier="Tier 3 (MATLAB)",
        language_backend="MATLAB",
        category="matlab_benchmark",
        python_import_name=None,
        cli_binary_name="matlab",
        install_command="Download academic release from https://www.nt.tuwien.ac.at/",
        repo_or_url="https://www.nt.tuwien.ac.at/",
        license_name="TU Wien Academic License",
        os_gpu_constraints="Requires licensed MATLAB installation.",
        disk_footprint_hint="~500 MB (plus MATLAB)",
        when_to_use="Cross-validating existing MATLAB 5G system-level research pipelines.",
        when_not_to_use="Open Python-only AI/ML stacks.",
        fusion_with_wce_lab="Export parameter table matching Vienna 5G scenario structs.",
    ),
}


def _safe_find_spec(module_name: str) -> bool:
    """Safely check whether a Python module or dotted submodule is importable without crashing on missing parents."""
    top_pkg = module_name.split(".")[0]
    try:
        if importlib.util.find_spec(top_pkg) is None:
            return False
        return importlib.util.find_spec(module_name) is not None
    except (ModuleNotFoundError, ImportError, ValueError):
        return False


def probe_local_ecosystem() -> dict[str, Any]:
    """Inspect current OS, GPU, installed Python wireless packages, and C++/Docker binaries."""
    installed_python: dict[str, bool] = {}
    for tid, tool in ECOSYSTEM_REGISTRY.items():
        if tool.python_import_name:
            installed_python[tid] = _safe_find_spec(tool.python_import_name)

    binaries: dict[str, bool] = {}
    for bin_name in ("git", "cmake", "docker", "matlab", "ns3", "gnb", "nr-softmodem"):
        binaries[bin_name] = shutil.which(bin_name) is not None

    return {
        "os": platform.system(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
        "cuda_available": torch.cuda.is_available(),
        "mps_available": bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
        "installed_python_tools": installed_python,
        "available_binaries": binaries,
    }


def recommend_ecosystem_tools(
    *,
    needs_ray_tracing: bool = False,
    needs_full_protocol_stack: bool = False,
    needs_real_sdr_oran: bool = False,
    needs_rf_iq_ml: bool = False,
    needs_fec_coding: bool = False,
    needs_isac_waveform: bool = False,
    needs_spatial_gscm_traj: bool = False,
) -> list[EcosystemTool]:
    """Select the minimal Just-in-Time (JIT) external tools needed beyond Tier 0 built-in physics."""
    picks: list[EcosystemTool] = [ECOSYSTEM_REGISTRY["wannado_wce_builtin"]]
    if needs_ray_tracing:
        picks.extend([ECOSYSTEM_REGISTRY["sionna_rt"], ECOSYSTEM_REGISTRY["deepmimo"]])
    if needs_fec_coding:
        picks.extend([ECOSYSTEM_REGISTRY["sionna"], ECOSYSTEM_REGISTRY["scikit_commpy"]])
    if needs_spatial_gscm_traj:
        picks.append(ECOSYSTEM_REGISTRY["quadriga_lib"])
    if needs_isac_waveform:
        picks.append(ECOSYSTEM_REGISTRY["hermespy"])
    if needs_rf_iq_ml:
        picks.append(ECOSYSTEM_REGISTRY["torchsig"])
    if needs_full_protocol_stack:
        picks.extend([ECOSYSTEM_REGISTRY["ns3_5glena_ns3gym"], ECOSYSTEM_REGISTRY["simu5g"]])
    if needs_real_sdr_oran:
        picks.extend([ECOSYSTEM_REGISTRY["openran_gym_nsoran"], ECOSYSTEM_REGISTRY["srsran_project"], ECOSYSTEM_REGISTRY["openairinterface"]])

    # Deduplicate preserving order
    seen: set[str] = set()
    out: list[EcosystemTool] = []
    for t in picks:
        if t.id not in seen:
            seen.add(t.id)
            out.append(t)
    return out


def format_ecosystem_table() -> str:
    """Format human/AI-readable Markdown inventory of the open-source wireless ecosystem."""
    lines = [
        "| Tool ID | Tier | Backend | When to Pull Just-in-Time | Install / Pull Recipe |",
        "| :--- | :--- | :--- | :--- | :--- |",
    ]
    for tid, t in ECOSYSTEM_REGISTRY.items():
        lines.append(f"| `{tid}` | {t.tier} | {t.language_backend} | {t.when_to_use} | `{t.install_command}` |")
    return "\n".join(lines)
