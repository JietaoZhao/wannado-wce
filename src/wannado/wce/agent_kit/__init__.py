"""Universal AI IDE skills & agent rule installers (AGENTS.md / Agent Skills standard + IDE adapters)."""

from __future__ import annotations

from pathlib import Path

AGENT_KIT_DIR = Path(__file__).resolve().parent
SKILL_PATH = AGENT_KIT_DIR / "SKILL.md"
TEMPLATES_DIR = AGENT_KIT_DIR / "templates"

_START_MARKER = "<!-- wannado-wce:start -->"
_END_MARKER = "<!-- wannado-wce:end -->"


def load_skill_text() -> str:
    """Return the canonical wannado-wce AI co-pilot SKILL.md contents."""
    return SKILL_PATH.read_text(encoding="utf-8")


def _upsert_marked_block(target_file: Path, section_body: str) -> Path:
    """Insert or cleanly update a marked wannado-wce instruction block without overwriting user or sibling blocks."""
    block = f"{_START_MARKER}\n{section_body.strip()}\n{_END_MARKER}\n"
    if target_file.exists():
        existing = target_file.read_text(encoding="utf-8")
        if _START_MARKER in existing and _END_MARKER in existing:
            pre, rest = existing.split(_START_MARKER, 1)
            _, post = rest.split(_END_MARKER, 1)
            new_text = pre + block + post.lstrip("\n")
        else:
            sep = "\n\n" if existing and not existing.endswith("\n\n") else ("\n" if existing and not existing.endswith("\n") else "")
            new_text = existing + sep + block
    else:
        new_text = block
    target_file.write_text(new_text, encoding="utf-8")
    return target_file


def install_agent_kit(
    project_dir: str | Path = ".",
    *,
    ide: str = "all",
) -> dict[str, Path]:
    """Install portable AI co-pilot skills and rules into a downstream project directory."""
    valid_ides = ("all", "universal", "cursor", "antigravity", "claude", "codex")
    if ide not in valid_ides:
        raise ValueError(f"Unsupported ide '{ide}'. Expected one of {valid_ides}.")

    root = Path(project_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    skill_text = load_skill_text()
    written: dict[str, Path] = {}

    if ide in ("all", "universal", "cursor", "codex"):
        uni_skill = root / ".agents" / "skills" / "wannado-wce" / "SKILL.md"
        uni_skill.parent.mkdir(parents=True, exist_ok=True)
        uni_skill.write_text(skill_text, encoding="utf-8")
        written["universal_skill"] = uni_skill

    if ide in ("all", "antigravity"):
        ag_skill = root / ".agent" / "skills" / "wannado-wce" / "SKILL.md"
        ag_skill.parent.mkdir(parents=True, exist_ok=True)
        ag_skill.write_text(skill_text, encoding="utf-8")
        written["antigravity_skill"] = ag_skill

    if ide in ("all", "cursor"):
        cursor_rule = root / ".cursor" / "rules" / "wannado-wce.mdc"
        cursor_rule.parent.mkdir(parents=True, exist_ok=True)
        cursor_rule.write_text(
            (TEMPLATES_DIR / "cursor_wannado_wce.mdc").read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        written["cursor_rule"] = cursor_rule

    if ide in ("all", "claude"):
        cl_skill = root / ".claude" / "skills" / "wannado-wce" / "SKILL.md"
        cl_skill.parent.mkdir(parents=True, exist_ok=True)
        cl_skill.write_text(skill_text, encoding="utf-8")
        written["claude_skill"] = cl_skill

        claude_md = _upsert_marked_block(
            root / "CLAUDE.md",
            (TEMPLATES_DIR / "CLAUDE.md").read_text(encoding="utf-8"),
        )
        written["claude_md"] = claude_md

    if ide in ("all", "universal", "cursor", "codex", "antigravity"):
        agents_md = _upsert_marked_block(
            root / "AGENTS.md",
            (TEMPLATES_DIR / "AGENTS.md").read_text(encoding="utf-8"),
        )
        written["agents_md"] = agents_md

    return written


__all__ = [
    "AGENT_KIT_DIR",
    "install_agent_kit",
    "load_skill_text",
]
