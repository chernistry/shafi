from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

JsonDict = dict[str, Any]


def _normalize_paths(changed_files: list[str]) -> list[str]:
    normalized = sorted({path.strip().replace("\\", "/") for path in changed_files if path.strip()})
    return normalized


def _matches_prefix(path: str, prefixes: tuple[str, ...]) -> bool:
    return any(path == prefix or path.startswith(f"{prefix}/") for prefix in prefixes)


def route_changed_files(changed_files: list[str], *, completed_packs: list[str] | None = None) -> JsonDict:
    normalized_files = _normalize_paths(changed_files)
    completed = sorted({pack.strip() for pack in (completed_packs or []) if pack.strip()})

    required_packs: list[str] = []
    reasons: list[JsonDict] = []

    def add(pack: str, reason: str) -> None:
        if pack not in required_packs:
            required_packs.append(pack)
        reasons.append({"pack": pack, "reason": reason})

    if not normalized_files:
        add("full_regression_pack", "no changed files declared; require the broadest safe pack")
    for path in normalized_files:
        if path == "src/rag_challenge/config/settings.py" or path in {"pyproject.toml", "uv.lock"}:
            add("full_regression_pack", f"config-wide change in {path}")
            continue
        if path == "src/rag_challenge/core/strict_answerer.py":
            add("strict_answerer_pack", f"strict answerer touched: {path}")
            continue
        if path in {
            "src/rag_challenge/core/retriever.py",
            "src/rag_challenge/core/pipeline.py",
            "src/rag_challenge/core/reranker.py",
            "src/rag_challenge/core/local_late_interaction_reranker.py",
        }:
            add("page_localization_pack", f"retrieval or reranker path touched: {path}")
            continue
        if path in {
            "src/rag_challenge/ingestion/parser.py",
            "src/rag_challenge/ingestion/chunker.py",
            "src/rag_challenge/ingestion/pipeline.py",
        }:
            add("reingest_ocr_pack", f"parser or chunker path touched: {path}")
            continue
        if path == "src/rag_challenge/llm/generator.py" or _matches_prefix(path, ("src/rag_challenge/prompts",)):
            add("free_text_pack", f"generator or prompt path touched: {path}")
            continue
        add("full_regression_pack", f"unmapped path requires full regression pack: {path}")

    required_packs = sorted(required_packs)
    missing_packs = [pack for pack in required_packs if pack not in completed]
    payload: JsonDict = {
        "changed_files": normalized_files,
        "required_packs": required_packs,
        "completed_packs": completed,
        "missing_packs": missing_packs,
        "routing_reasons": reasons,
        "should_block_promotion": bool(missing_packs),
    }
    return payload


def _render_markdown(payload: JsonDict) -> str:
    lines = [
        "# Impact Router",
        "",
        f"- required_packs: `{', '.join(cast('list[str]', payload.get('required_packs') or [])) or 'none'}`",
        f"- completed_packs: `{', '.join(cast('list[str]', payload.get('completed_packs') or [])) or 'none'}`",
        f"- missing_packs: `{', '.join(cast('list[str]', payload.get('missing_packs') or [])) or 'none'}`",
        f"- should_block_promotion: `{payload.get('should_block_promotion')}`",
        "",
        "## Routing Reasons",
        "",
    ]
    reasons = cast("list[JsonDict]", payload.get("routing_reasons") or [])
    if reasons:
        for row in reasons:
            lines.append(f"- `{row.get('pack')}`: {row.get('reason')}")
    else:
        lines.append("- none")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Route changed files to the smallest valid eval pack.")
    parser.add_argument("--changed-file", action="append", default=[])
    parser.add_argument("--completed-pack", action="append", default=[])
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, default=None)
    args = parser.parse_args()

    payload = {
        "impact_router": route_changed_files(
            [str(item) for item in cast("list[object]", args.changed_file)],
            completed_packs=[str(item) for item in cast("list[object]", args.completed_pack)],
        )
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.out_md is not None:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(_render_markdown(payload["impact_router"]) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
