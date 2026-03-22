from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import TypedDict

from rag_challenge.ingestion.document_interrogation import (
    DocumentInterrogationRecord,
    build_document_interrogation_system_prompt,
    build_document_interrogation_user_prompt,
    load_document_interrogation_inputs,
    parse_document_interrogation_json,
)
from rag_challenge.llm.provider import LLMProvider


class PromptRow(TypedDict):
    doc_id: str
    doc_title: str
    system_prompt: str
    user_prompt: str


async def _run_async(args: argparse.Namespace) -> int:
    inputs = load_document_interrogation_inputs(Path(args.input))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prompt_rows: list[PromptRow] = []
    for doc in inputs:
        prompt_rows.append(
            {
                "doc_id": doc.doc_id,
                "doc_title": doc.doc_title,
                "system_prompt": build_document_interrogation_system_prompt(),
                "user_prompt": build_document_interrogation_user_prompt(doc),
            }
        )

    prompt_path = output_dir / "document_interrogation_prompts.json"
    prompt_path.write_text(json.dumps(prompt_rows, ensure_ascii=True, indent=2), encoding="utf-8")

    if args.dry_run:
        print(f"Wrote dry-run prompt pack to {prompt_path}")
        return 0

    provider = LLMProvider()
    records: list[dict[str, object]] = []
    try:
        for prompt_row in prompt_rows:
            result = await provider.generate(
                system_prompt=prompt_row["system_prompt"],
                user_prompt=prompt_row["user_prompt"],
                model=args.model,
                max_tokens=args.max_tokens,
                temperature=0.0,
            )
            record: DocumentInterrogationRecord = parse_document_interrogation_json(result.text)
            records.append(record.model_dump(mode="json"))
    finally:
        await provider.close()

    records_path = output_dir / "document_interrogation_records.json"
    records_path.write_text(json.dumps(records, ensure_ascii=True, indent=2), encoding="utf-8")
    print(f"Wrote {len(records)} interrogation records to {records_path}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate offline LLM document interrogation enrichments.")
    parser.add_argument("--input", required=True, help="JSON or JSONL document interrogation input file.")
    parser.add_argument("--output-dir", required=True, help="Directory for prompts and output records.")
    parser.add_argument("--model", default="", help="Override model name for the offline interrogation call.")
    parser.add_argument("--max-tokens", type=int, default=2000, help="Maximum completion tokens.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write prompt pack only and skip live provider calls.",
    )
    return parser.parse_args()


def main() -> int:
    return asyncio.run(_run_async(parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
