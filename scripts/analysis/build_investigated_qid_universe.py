from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

JsonDict = dict[str, Any]


def _load_json(path: Path) -> JsonDict:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return cast("JsonDict", obj)


def _load_qids(path: Path | None) -> set[str]:
    if path is None or not path.exists():
        return set()
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def build_investigated_qid_universe(manifest_paths: list[Path], *, extra_qids: set[str] | None = None) -> list[str]:
    qids: set[str] = set()
    for manifest_path in manifest_paths:
        payload = _load_json(manifest_path)
        candidates = payload.get("candidates")
        if not isinstance(candidates, list):
            continue
        for raw in cast("list[object]", candidates):
            if not isinstance(raw, dict):
                continue
            row = cast("JsonDict", raw)
            for key in ("allowed_answer_qids", "allowed_page_qids"):
                values = row.get(key)
                if not isinstance(values, list):
                    continue
                for item in cast("list[object]", values):
                    qid = str(item).strip()
                    if qid:
                        qids.add(qid)
    if extra_qids:
        qids.update(extra_qids)
    return sorted(qids)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a union of investigated QIDs from one or more candidate manifests.")
    parser.add_argument("--manifest-json", type=Path, action="append", required=True)
    parser.add_argument("--extra-qids-file", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    extra_qids = _load_qids(args.extra_qids_file)
    qids = build_investigated_qid_universe(args.manifest_json, extra_qids=extra_qids)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(f"{qid}\n" for qid in qids), encoding="utf-8")
    json_out = args.json_out or args.out.with_suffix(".json")
    json_out.write_text(
        json.dumps(
            {
                "count": len(qids),
                "question_ids": qids,
                "manifest_paths": [str(path) for path in args.manifest_json],
                "extra_qids_file": None if args.extra_qids_file is None else str(args.extra_qids_file),
                "submission_policy": "NO_SUBMIT_WITHOUT_USER_APPROVAL",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
