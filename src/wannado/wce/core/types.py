"""Shared metadata schemas, standard references, and provisioning contracts for wannado.wce."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

FamilyType = Literal[
    "channel_classic",
    "channel_3gpp",
    "arrays_and_ris",
    "traffic_and_mac",
    "domain_6g",
    "baselines",
]
EraType = Literal["classic", "standard", "frontier"]
TierType = Literal["S", "A", "B", "C"]
OutputKindType = Literal[
    "complex_channel_tensor",
    "traffic_queue_trace",
    "mac_gym_environment",
    "link_capacity_metric",
    "baseline_policy_solution",
    "spatial_channel_map",
]


@dataclass(frozen=True)
class LitCitation:
    """Verified academic or technical-standard citation record (zero hallucination)."""

    key: str
    title: str
    authors: str
    venue: str
    year: int
    url: str
    bibtex: str
    ieee: str
    apa: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StandardRef:
    """Structured authoritative standard reference (ITU-R, 3GPP, ETSI, IEEE)."""

    id: str
    org: str
    doc_id: str
    section: str
    title: str
    release: str
    year: int
    url: str
    summary_plain: str
    key_parameters: tuple[str, ...]
    citation: LitCitation

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["key_parameters"] = list(self.key_parameters)
        return data


@dataclass(frozen=True)
class ParamProvenance:
    """Explicit traceability entry linking one configured simulation parameter to its authority."""

    parameter: str
    value: Any
    unit: str
    plain_meaning: str
    authority_id: str
    authority_section: str
    citation_key: str
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class WCEMeta:
    """Single-source-of-truth metadata card embedded in every wannado-wce single-file module."""

    id: str
    name: str
    family: FamilyType
    era: EraType
    tier: TierType
    output_kind: OutputKindType
    supports_torch_complex: bool
    supports_autograd: bool
    standard_ids: tuple[str, ...]
    parent_id: str | None
    diff_from_parent: str
    summary_plain: str
    when_to_use: str
    when_not_to_use: str
    hyperparam_intuition: dict[str, str]
    extension_hooks: tuple[str, ...]
    citation: LitCitation
    module_path: str = ""
    default_mechanisms: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Serialize metadata card into a JSON-compatible dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "family": self.family,
            "era": self.era,
            "tier": self.tier,
            "output_kind": self.output_kind,
            "supports_torch_complex": self.supports_torch_complex,
            "supports_autograd": self.supports_autograd,
            "standard_ids": list(self.standard_ids),
            "parent_id": self.parent_id,
            "diff_from_parent": self.diff_from_parent,
            "summary_plain": self.summary_plain,
            "when_to_use": self.when_to_use,
            "when_not_to_use": self.when_not_to_use,
            "hyperparam_intuition": dict(self.hyperparam_intuition),
            "extension_hooks": list(self.extension_hooks),
            "module_path": self.module_path,
            "default_mechanisms": list(self.default_mechanisms),
            "citation": self.citation.to_dict(),
        }


@dataclass
class ProvisionResult:
    """Standardized output returned by every wannado.wce single-file provisioning block."""

    block_id: str
    seed: int
    elapsed_seconds: float
    config: dict[str, Any]
    tensors: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    parameter_provenance: list[ParamProvenance] = field(default_factory=list)
    mechanisms_used: list[str] = field(default_factory=list)
    standard_ids_used: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_summary_dict(self) -> dict[str, Any]:
        """Serialize JSON-safe summary excluding raw tensor payloads."""
        tensor_shapes = {
            k: {
                "shape": list(v.shape) if hasattr(v, "shape") else None,
                "dtype": str(v.dtype) if hasattr(v, "dtype") else type(v).__name__,
            }
            for k, v in self.tensors.items()
        }
        return {
            "block_id": self.block_id,
            "seed": self.seed,
            "elapsed_seconds": round(float(self.elapsed_seconds), 4),
            "config": dict(self.config),
            "tensor_shapes": tensor_shapes,
            "metrics": {k: float(v) for k, v in self.metrics.items()},
            "diagnostics": dict(self.diagnostics),
            "parameter_provenance": [p.to_dict() for p in self.parameter_provenance],
            "mechanisms_used": list(self.mechanisms_used),
            "standard_ids_used": list(self.standard_ids_used),
        }
