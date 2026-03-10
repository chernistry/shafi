from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

_ENV_PROMPTS_DIR = "RAG_CHALLENGE_PROMPTS_DIR"
_PROMPTS_ROOT = Path(__file__).resolve().parent


def _candidate_roots() -> list[Path]:
    candidates: list[Path] = []
    env_dir = os.environ.get(_ENV_PROMPTS_DIR)
    if env_dir:
        candidates.append(Path(env_dir).expanduser().resolve())
    candidates.append(_PROMPTS_ROOT)
    return candidates


def _resolve_prompt_path(prompt_path: str) -> Path:
    normalized = prompt_path.strip().lstrip("/").replace("\\", "/")
    if not normalized.endswith(".md"):
        normalized = f"{normalized}.md"

    for root in _candidate_roots():
        candidate = (root / normalized).resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            continue
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(f"Prompt file not found: {prompt_path}")


@lru_cache(maxsize=256)
def load_prompt(prompt_path: str) -> str:
    """Load prompt text from markdown file with in-process caching."""
    path = _resolve_prompt_path(prompt_path)
    return path.read_text(encoding="utf-8").strip()


def preload_prompts(*prompt_paths: str) -> None:
    for prompt_path in prompt_paths:
        load_prompt(prompt_path)
