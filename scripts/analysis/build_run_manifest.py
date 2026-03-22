from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

JsonDict = dict[str, Any]
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORE_FILES = (
    "src/shafi/ingestion/parser.py",
    "src/shafi/ingestion/chunker.py",
    "src/shafi/core/retriever.py",
    "src/shafi/core/pipeline.py",
    "src/shafi/config/settings.py",
)


def _resolve_path(*, root: Path, raw: str | Path | None) -> Path | None:
    if raw is None:
        return None
    value = Path(raw).expanduser()
    if value.is_absolute():
        return value.resolve()
    return (root / value).resolve()


def _load_json(path: Path | None) -> JsonDict:
    if path is None or not path.exists():
        return {}
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return cast("JsonDict", obj)


def _iter_raw_results(path: Path | None) -> list[JsonDict]:
    if path is None or not path.exists():
        return []
    obj = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(obj, list):
        return [cast("JsonDict", item) for item in cast("list[object]", obj) if isinstance(item, dict)]
    if isinstance(obj, dict):
        results_obj = cast("JsonDict", obj).get("results")
        if isinstance(results_obj, list):
            return [cast("JsonDict", item) for item in cast("list[object]", results_obj) if isinstance(item, dict)]
    raise ValueError(f"Expected raw-results list or object with results[] in {path}")


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _relativize(path: Path, *, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _git_value(repo_root: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return ""
    return proc.stdout.strip()


def _build_git_payload(*, repo_root: Path, explicit_sha: str | None) -> JsonDict:
    git_sha = (explicit_sha or "").strip() or _git_value(repo_root, "rev-parse", "HEAD")
    branch = _git_value(repo_root, "rev-parse", "--abbrev-ref", "HEAD")
    porcelain = _git_value(repo_root, "status", "--porcelain=v1", "--untracked-files=all")
    dirty_entries = [line for line in porcelain.splitlines() if line.strip()]
    payload: JsonDict = {
        "sha": git_sha,
        "branch": branch,
        "dirty": bool(dirty_entries),
        "dirty_entry_count": len(dirty_entries),
        "porcelain_sha256": _sha256_text(porcelain) if porcelain else "",
    }
    payload["fingerprint"] = _sha256_text(_canonical_json(payload))
    return payload


def _build_file_entries(*, repo_root: Path, raw_paths: list[str]) -> list[JsonDict]:
    entries: list[JsonDict] = []
    seen: set[str] = set()
    for raw in raw_paths:
        path = _resolve_path(root=repo_root, raw=raw)
        if path is None:
            continue
        rel_path = _relativize(path, root=repo_root)
        if rel_path in seen:
            continue
        seen.add(rel_path)
        entry: JsonDict = {
            "path": rel_path,
            "exists": path.exists(),
            "sha256": _sha256_file(path) if path.exists() and path.is_file() else "",
            "size_bytes": path.stat().st_size if path.exists() and path.is_file() else 0,
        }
        entries.append(entry)
    entries.sort(key=lambda item: str(item["path"]))
    return entries


def _parse_key_value_pairs(raw_pairs: list[str]) -> JsonDict:
    payload: JsonDict = {}
    for raw in raw_pairs:
        text = raw.strip()
        if not text:
            continue
        if "=" not in text:
            raise ValueError(f"Expected KEY=VALUE pair, got: {raw}")
        key, value = text.split("=", 1)
        normalized_key = key.strip()
        if not normalized_key:
            raise ValueError(f"Missing key in pair: {raw}")
        payload[normalized_key] = value.strip()
    return payload


def _build_models_payload(*, raw_results_path: Path | None, declared_versions: JsonDict) -> JsonDict:
    model_counts: dict[str, int] = {}
    route_model_counts: dict[tuple[str, str], int] = {}
    for item in _iter_raw_results(raw_results_path):
        telemetry_obj = item.get("telemetry")
        telemetry = cast("JsonDict", telemetry_obj) if isinstance(telemetry_obj, dict) else {}
        route = str(telemetry.get("route") or telemetry.get("route_id") or telemetry.get("answer_type") or "").strip()
        model_name = str(telemetry.get("model_name") or telemetry.get("model_llm") or "").strip()
        if not model_name:
            continue
        model_counts[model_name] = model_counts.get(model_name, 0) + 1
        route_key = route or "unknown"
        route_model_counts[(route_key, model_name)] = route_model_counts.get((route_key, model_name), 0) + 1

    observed_models: list[JsonDict] = [
        {"name": name, "count": count}
        for name, count in sorted(model_counts.items(), key=lambda item: (-item[1], item[0]))
    ]
    observed_routes: list[JsonDict] = [
        {"route": route, "model": model, "count": count}
        for (route, model), count in sorted(route_model_counts.items(), key=lambda item: (-item[1], item[0][0], item[0][1]))
    ]
    payload: JsonDict = {
        "declared_versions": declared_versions,
        "observed_models": observed_models,
        "observed_route_model_counts": observed_routes,
    }
    payload["fingerprint"] = _sha256_text(_canonical_json(payload))
    return payload


def _coerce_int(value: object) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return int(float(text))
        except ValueError:
            return None
    return None


def _build_qdrant_payload(preflight_payload: JsonDict) -> JsonDict:
    collection_name = str(preflight_payload.get("phase_collection_name") or preflight_payload.get("collection_name") or "").strip()
    point_count = _coerce_int(preflight_payload.get("qdrant_point_count"))
    payload: JsonDict = {
        "collection_name": collection_name,
        "point_count": point_count,
        "submission_sha256": str(preflight_payload.get("submission_sha256") or "").strip(),
        "code_archive_sha256": str(preflight_payload.get("code_archive_sha256") or "").strip(),
        "questions_sha256": str(preflight_payload.get("questions_sha256") or "").strip(),
        "documents_zip_sha256": str(preflight_payload.get("documents_zip_sha256") or "").strip(),
        "pdf_count": _coerce_int(preflight_payload.get("pdf_count")),
    }
    payload["fingerprint"] = _sha256_text(_canonical_json(payload))
    return payload


def _build_environment_payload(env_keys: list[str]) -> JsonDict:
    env_entries: list[JsonDict] = []
    for key in sorted({item.strip() for item in env_keys if item.strip()}):
        value = os.environ.get(key)
        env_entries.append(
            {
                "name": key,
                "is_set": value is not None,
                "value_sha256": _sha256_text(value) if value is not None else "",
            }
        )

    payload: JsonDict = {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "executable_name": Path(sys.executable).name,
        "selected_env": env_entries,
    }
    payload["fingerprint"] = _sha256_text(_canonical_json(payload))
    return payload


def _render_markdown(manifest: JsonDict) -> str:
    git_payload = cast("JsonDict", manifest.get("git") or {})
    qdrant_payload = cast("JsonDict", manifest.get("qdrant") or {})
    models_payload = cast("JsonDict", manifest.get("models") or {})
    touched_files = cast("list[JsonDict]", manifest.get("touched_file_hashes") or [])
    core_files = cast("list[JsonDict]", manifest.get("core_pipeline_hashes") or [])
    lines = [
        "# Run Manifest",
        "",
        f"- candidate_label: `{manifest.get('candidate_label')}`",
        f"- fingerprint: `{manifest.get('fingerprint')}`",
        f"- git_sha: `{git_payload.get('sha') or 'unknown'}`",
        f"- git_branch: `{git_payload.get('branch') or 'unknown'}`",
        f"- git_dirty: `{git_payload.get('dirty')}`",
        f"- qdrant_collection: `{qdrant_payload.get('collection_name') or 'unknown'}`",
        f"- qdrant_point_count: `{qdrant_payload.get('point_count')}`",
        "",
        "## Models",
        "",
    ]
    observed_models = cast("list[JsonDict]", models_payload.get("observed_models") or [])
    if observed_models:
        for item in observed_models:
            lines.append(f"- `{item.get('name')}`: `{item.get('count')}`")
    else:
        lines.append("- none")
    lines.extend(["", "## Touched Files", ""])
    if touched_files:
        for item in touched_files:
            lines.append(f"- `{item.get('path')}` sha256=`{item.get('sha256')}`")
    else:
        lines.append("- none")
    lines.extend(["", "## Core Pipeline Files", ""])
    if core_files:
        for item in core_files:
            lines.append(f"- `{item.get('path')}` sha256=`{item.get('sha256')}`")
    else:
        lines.append("- none")
    return "\n".join(lines) + "\n"


def build_run_manifest(
    *,
    candidate_label: str,
    repo_root: Path,
    git_sha: str | None,
    preflight_json: Path | None,
    raw_results_json: Path | None,
    touched_files: list[str],
    core_files: list[str],
    model_versions: JsonDict,
    env_keys: list[str],
) -> JsonDict:
    effective_core_files = core_files or [path for path in DEFAULT_CORE_FILES if (repo_root / path).exists()]
    preflight_payload = _load_json(preflight_json)
    manifest: JsonDict = {
        "schema_version": 1,
        "candidate_label": candidate_label,
        "repo_root": str(repo_root),
        "git": _build_git_payload(repo_root=repo_root, explicit_sha=git_sha),
        "touched_file_hashes": _build_file_entries(repo_root=repo_root, raw_paths=touched_files),
        "core_pipeline_hashes": _build_file_entries(repo_root=repo_root, raw_paths=effective_core_files),
        "models": _build_models_payload(raw_results_path=raw_results_json, declared_versions=model_versions),
        "qdrant": _build_qdrant_payload(preflight_payload),
        "environment": _build_environment_payload(env_keys),
    }
    manifest["fingerprint"] = _sha256_text(_canonical_json(manifest))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a deterministic run manifest fingerprint for a candidate or eval artifact.")
    parser.add_argument("--candidate-label", required=True)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--git-sha", default=None)
    parser.add_argument("--preflight-json", type=Path, default=None)
    parser.add_argument("--raw-results-json", type=Path, default=None)
    parser.add_argument("--touched-file", action="append", default=[])
    parser.add_argument("--core-file", action="append", default=[])
    parser.add_argument("--model-version", action="append", default=[])
    parser.add_argument("--env-key", action="append", default=[])
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, default=None)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    manifest = build_run_manifest(
        candidate_label=str(args.candidate_label),
        repo_root=repo_root,
        git_sha=str(args.git_sha) if args.git_sha is not None else None,
        preflight_json=_resolve_path(root=repo_root, raw=args.preflight_json),
        raw_results_json=_resolve_path(root=repo_root, raw=args.raw_results_json),
        touched_files=[str(item) for item in cast("list[object]", args.touched_file)],
        core_files=[str(item) for item in cast("list[object]", args.core_file)],
        model_versions=_parse_key_value_pairs([str(item) for item in cast("list[object]", args.model_version)]),
        env_keys=[str(item) for item in cast("list[object]", args.env_key)],
    )

    payload = {"run_manifest": manifest}
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.out_md is not None:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(_render_markdown(manifest) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
