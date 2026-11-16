"""Project traversal and wireless domain/simulator detection for Stage 0 WCE provisioning."""

from __future__ import annotations

import ast
import os
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

_IGNORE_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".agent",
    ".agents",
    ".claude",
    ".cursor",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "node_modules",
    ".venv",
    "venv",
    "env",
    "dist",
    "build",
    "egg-info",
    ".ipynb_checkpoints",
}

_ARCHETYPE_SIGNALS: dict[str, tuple[str, ...]] = {
    "imt2030_isac_monostatic": ("isac", "jsac", "radar", "rcs", "crb", "sensing", "monostatic", "bistatic", "802.11bf"),
    "uav_ris_relay": ("uav", "drone", "aerial", "ris", "irs", "star-ris", "star_ris", "reconfigurable intelligent"),
    "xlmimo_nearfield_thz": ("near-field", "nearfield", "xl-mimo", "xlmimo", "spherical", "rayleigh distance", "thz", "terahertz"),
    "imt2030_ubiquitous_leo_ntn": ("ntn", "leo", "satellite", "orbit", "elevation angle", "tr 38.811", "otfs"),
    "imt2030_hrllc_factory": ("urllc", "hrllc", "finite blocklength", "fbl", "polyanskiy", "factory", "inf-dl", "short packet"),
    "imt2030_immersive_xr": ("xr", "extended reality", "pdb", "fps", "38.838", "immersive", "holographic"),
    "imt2030_massive_ambient_iot": ("ambient iot", "a-iot", "backscatter", "energy harvesting", "mmtc", "38.848"),
    "imt2030_aiac_semantic_aircomp": ("semantic", "deepsc", "aircomp", "over-the-air", "federated", "ckm", "radiomap", "38.843"),
    "v2x_highway_urllc": ("v2x", "v2v", "v2i", "sidelink", "37.885", "highway", "platooning"),
    "multicell_ofdma_mac": ("scheduler", "scheduling", "resource block", "prb", "proportional fair", "ofdma", "queue", "cqi", "sinr"),
    "jamming_contested_rf": ("jamming", "jsr", "wiretap", "secrecy", "covert", "eavesdropper", "iq", "modulation"),
}

_ARCHETYPE_DESCRIPTIONS: dict[str, tuple[str, str]] = {
    "imt2030_isac_monostatic": (
        "This looks like an Integrated Sensing and Communication (ISAC / JSAC) research project.",
        "Provision a 3GPP Rel-19 ISAC wireless environment modeling background multipath, target RCS reflection echoes, and sensing CRB vs communication rate.",
    ),
    "uav_ris_relay": (
        "This looks like a UAV and/or Reconfigurable Intelligent Surface (RIS) wireless relay project.",
        "Provision an elevation-dependent UAV A2G (TR 36.777 / Al-Hourani) + energy-conserving RIS cascaded environment (ETSI GR RIS 003).",
    ),
    "xlmimo_nearfield_thz": (
        "This looks like a Near-Field XL-MIMO or Sub-THz wireless propagation project.",
        "Provision a spherical-wavefront XL-MIMO channel honoring the Rayleigh distance 2D^2/lambda and ITU-R P.676 molecular absorption.",
    ),
    "imt2030_ubiquitous_leo_ntn": (
        "This looks like a Non-Terrestrial Network (NTN / LEO satellite / OTFS) communications project.",
        "Provision a 3GPP TR 38.811/38.821 LEO orbital pass environment with elevation-indexed clutter/K-factor and orbital Doppler.",
    ),
    "imt2030_hrllc_factory": (
        "This looks like a Hyper-Reliable Low-Latency (HRLLC / URLLC / Factory) communication project.",
        "Provision a TR 38.901 InF industrial environment coupled with Polyanskiy finite-blocklength capacity & short-packet BLER.",
    ),
    "imt2030_immersive_xr": (
        "This looks like an Immersive XR / high-throughput 5G-6G cellular communication project.",
        "Provision a 3GPP TR 38.901 cellular MIMO-OFDM channel paired with TR 38.838 periodic XR frame traffic and PDB satisfaction checks.",
    ),
    "imt2030_massive_ambient_iot": (
        "This looks like a Massive / Ambient IoT backscatter or ultra-dense sensor network project.",
        "Provision a TR 38.848 RF energy-harvesting backscatter link and PPP spatial density environment.",
    ),
    "imt2030_aiac_semantic_aircomp": (
        "This looks like an AI-Native Wireless (Semantic Comm / AirComp / CKM / CSI ML) project.",
        "Provision a differentiable PyTorch baseband semantic/AirComp Cell-Free environment with LS/LMMSE & CKM baselines.",
    ),
    "v2x_highway_urllc": (
        "This looks like a Vehicular V2X / Sidelink communication project.",
        "Provision a 3GPP TR 37.885 V2V Urban Grid / Highway environment with NLOSv vehicle blockage and dual-mobility Doppler.",
    ),
    "multicell_ofdma_mac": (
        "This looks like a Multi-Cell Radio Resource Scheduling / MAC Queue project.",
        "Provision a wrap-around 3GPP TR 38.901 multi-cell environment with non-full-buffer FTP/Poisson queues and Gym interface.",
    ),
    "jamming_contested_rf": (
        "This looks like a Physical-Layer Security, Covert, or RF Anti-Jamming project.",
        "Provision a complex64 contested RF environment with Wyner secrecy, Bash covert limits, and JSR-dB jamming aligned with wannado-adv.",
    ),
}


def _extract_readme_summary(root: Path) -> tuple[str, str]:
    for name in ("README.md", "readme.md", "README.rst", "README.txt"):
        fpath = root / name
        if not fpath.is_file():
            continue
        try:
            raw = fpath.read_text(encoding="utf-8", errors="ignore")[:4096]
        except OSError:
            continue
        title = ""
        para: list[str] = []
        for line in raw.splitlines():
            s = line.strip()
            if not s or s.startswith("<") or s.startswith("![") or s.startswith("[!["):
                continue
            if not title and s.startswith("#"):
                title = s.lstrip("#").strip()
                continue
            if s.startswith("#") or s.startswith("---"):
                if para:
                    break
                continue
            para.append(s)
            if sum(len(p) for p in para) >= 220:
                break
        return title, " ".join(para)[:240].strip()
    return "", ""


@dataclass
class ProjectScan:
    """Fresh structural snapshot of a user repository before wireless environment provisioning."""

    root: str
    scanned_files_count: int
    total_python_files: int
    frameworks: list[str] = field(default_factory=list)
    sibling_labs_present: list[str] = field(default_factory=list)
    existing_wce_lab: bool = False
    likely_archetype: str = "imt2030_immersive_xr"
    detected_symbols: list[str] = field(default_factory=list)
    readme_title: str = ""
    readme_excerpt: str = ""
    understanding_confidence: str = "low"
    project_summary_guess: str = ""
    inferred_wce_goal: str = ""
    evidence_clues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def format_summary(self) -> str:
        lines = [
            f"[wannado-wce Project Scan] root={self.root}",
            f"  Python files scanned : {self.scanned_files_count} (total={self.total_python_files})",
            f"  Detected frameworks  : {', '.join(self.frameworks) if self.frameworks else 'none yet'}",
            f"  Sibling WannaDo labs : {', '.join(self.sibling_labs_present) if self.sibling_labs_present else 'none'} | wce_lab={self.existing_wce_lab}",
            f"  AI Understanding     : [{self.understanding_confidence.upper()}] {self.project_summary_guess}",
            f"  Inferred WCE Goal    : {self.inferred_wce_goal}",
        ]
        if self.evidence_clues:
            lines.append(f"  Observed clues       : {'; '.join(self.evidence_clues[:6])}")
        return "\n".join(lines)


def scan_project(project_dir: str | Path = ".", max_files: int = 200) -> ProjectScan:
    """Traverse workspace files and detect wireless physics clues, external libs, and WannaDo siblings."""
    root = Path(project_dir).resolve()
    py_files: list[Path] = []
    total_py = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in _IGNORE_DIRS and not d.endswith(".egg-info"))
        for fname in sorted(filenames):
            if fname.endswith(".py"):
                total_py += 1
                if len(py_files) < max_files:
                    py_files.append(Path(dirpath) / fname)

    fw_map = {
        "torch": "PyTorch",
        "numpy": "NumPy",
        "scipy": "SciPy",
        "sionna": "NVIDIA Sionna",
        "deepmimo": "DeepMIMO",
        "quadriga_lib": "QuaDRiGa",
        "hermespy": "HermesPy",
        "torchsig": "TorchSig",
        "gymnasium": "Gymnasium",
        "gym": "Legacy Gym",
        "commpy": "CommPy",
    }
    frameworks: set[str] = set()
    symbols: list[str] = []
    doc_snippets: list[str] = []
    scores: dict[str, int] = {k: 0 for k in _ARCHETYPE_SIGNALS}
    matched_kws: dict[str, list[str]] = {k: [] for k in _ARCHETYPE_SIGNALS}

    for fpath in py_files:
        rel = fpath.relative_to(root).as_posix()
        is_lab = "wce_lab/" in rel.lower()
        try:
            text = fpath.read_text(encoding="utf-8", errors="ignore")[:131072]
        except OSError:
            continue
        low_txt = text.lower()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                tree = ast.parse(text)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for a in node.names:
                        top = a.name.split(".")[0]
                        if a.name.startswith("wannado."):
                            frameworks.add(a.name.split(".")[0] + "." + a.name.split(".")[1] if "." in a.name else a.name)
                        elif top in fw_map:
                            frameworks.add(fw_map[top])
                elif isinstance(node, ast.ImportFrom) and node.module:
                    top = node.module.split(".")[0]
                    if node.module.startswith("wannado."):
                        parts = node.module.split(".")
                        frameworks.add(".".join(parts[:2]))
                    elif top in fw_map:
                        frameworks.add(fw_map[top])
                elif not is_lab and isinstance(node, ast.ClassDef) and not node.name.startswith("_") and len(symbols) < 12:
                    symbols.append(f"class {node.name} ({rel})")
            if not is_lab:
                doc = ast.get_docstring(tree)
                if doc and len(doc_snippets) < 4:
                    first = doc.strip().splitlines()[0].strip()
                    if first:
                        doc_snippets.append(f"{rel}: {first[:140]}")
        except SyntaxError:
            pass

        if not is_lab:
            for arch, kws in _ARCHETYPE_SIGNALS.items():
                hits = [kw for kw in kws if kw in low_txt]
                if hits:
                    scores[arch] += len(hits)
                    for kw in hits:
                        if kw not in matched_kws[arch]:
                            matched_kws[arch].append(kw)

    readme_title, readme_excerpt = _extract_readme_summary(root)
    readme_low = f"{readme_title} {readme_excerpt}".lower()
    for arch, kws in _ARCHETYPE_SIGNALS.items():
        hits = [kw for kw in kws if kw in readme_low]
        if hits:
            scores[arch] += 2 * len(hits)
            for kw in hits:
                if kw not in matched_kws[arch]:
                    matched_kws[arch].append(kw)

    best_arch = max(scores.items(), key=lambda kv: kv[1])[0]
    best_score = scores[best_arch]
    if best_score == 0:
        best_arch = "imt2030_immersive_xr"

    sibling_labs = [d for d in ("drl_lab", "adv_eval") if (root / d).is_dir()]
    existing_wce = (root / "wce_lab").is_dir()

    clues: list[str] = []
    if readme_title or readme_excerpt:
        clues.append(f"README: {readme_title or readme_excerpt}")
    if best_score > 0:
        clues.append(f"Wireless keywords ({best_arch}): {', '.join(matched_kws[best_arch][:6])}")
    if doc_snippets:
        clues.append(f"Module doc: {doc_snippets[0]}")
    if symbols and len(clues) < 4:
        clues.append(f"Symbols: {', '.join(symbols[:3])}")
    if sibling_labs:
        clues.append(f"Sibling WannaDo workspaces: {', '.join(sibling_labs)}")

    base_summary, base_goal = _ARCHETYPE_DESCRIPTIONS[best_arch]
    if best_score >= 3:
        conf = "high"
        summary = base_summary + (f" Clues: {clues[0]}." if clues else "")
        goal = base_goal
    elif best_score >= 1 or readme_excerpt or doc_snippets or frameworks:
        conf = "medium"
        summary = f"{base_summary} Partial cues detected ({clues[0] if clues else sorted(frameworks)}); please verify exact research scope."
        goal = base_goal
    else:
        conf = "low"
        summary = (
            f"Scanned {total_py} Python file(s); no explicit wireless scenario clues or README detected yet. "
            "Share what you are researching in one sentence so we can custom-provision the exact 3GPP/ITU environment."
        )
        goal = "Clarify the target wireless research scenario (channel, topology, traffic, and downstream consumer) in one sentence."

    return ProjectScan(
        root=str(root),
        scanned_files_count=len(py_files),
        total_python_files=total_py,
        frameworks=sorted(frameworks),
        sibling_labs_present=sibling_labs,
        existing_wce_lab=existing_wce,
        likely_archetype=best_arch,
        detected_symbols=symbols,
        readme_title=readme_title,
        readme_excerpt=readme_excerpt,
        understanding_confidence=conf,
        project_summary_guess=summary,
        inferred_wce_goal=goal,
        evidence_clues=clues,
    )
