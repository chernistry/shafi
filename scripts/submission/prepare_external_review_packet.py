from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_UPLOAD_DIR = PROJECT_ROOT / ".sdd" / "upload"


@dataclass(frozen=True)
class PacketFile:
    source: Path
    target_name: str


PACKET_FILES: tuple[PacketFile, ...] = (
    PacketFile(PROJECT_ROOT / ".sdd" / "upload" / "00_external_review_prompt_2026-03-12.md", "00_external_review_prompt_2026-03-12.md"),
    PacketFile(PROJECT_ROOT / ".sdd" / "archive" / "scraped_challenge_terms.md", "scraped_challenge_terms.md"),
    PacketFile(PROJECT_ROOT / ".sdd" / "researches" / "competition_supervisor_2026-03-12.md", "competition_supervisor_2026-03-12.md"),
    PacketFile(PROJECT_ROOT / ".sdd" / "researches" / "leaderboard_geometry_2026-03-12_18-45.md", "leaderboard_geometry_2026-03-12_18-45.md"),
    PacketFile(PROJECT_ROOT / ".sdd" / "researches" / "experiment_gate_v_anchor_rebuild_iter9_moneyseed_nopage1blanket_vs_v6_context_seed.md", "experiment_gate_v_anchor_rebuild_iter9_moneyseed_nopage1blanket_vs_v6_context_seed.md"),
    PacketFile(PROJECT_ROOT / ".sdd" / "researches" / "anchor_slice_v_anchor_rebuild_iter9_moneyseed_nopage1blanket_vs_v6_context_seed.md", "anchor_slice_v_anchor_rebuild_iter9_moneyseed_nopage1blanket_vs_v6_context_seed.md"),
    PacketFile(PROJECT_ROOT / ".sdd" / "researches" / "platform_scoring_v_anchor_rebuild_iter9_moneyseed_nopage1blanket.md", "platform_scoring_v_anchor_rebuild_iter9_moneyseed_nopage1blanket.md"),
    PacketFile(PROJECT_ROOT / ".sdd" / "researches" / "exactness_review_queue_2026-03-12.md", "exactness_review_queue_2026-03-12.md"),
    PacketFile(PROJECT_ROOT / ".sdd" / "researches" / "local_ollama_embedding_feasibility_2026-03-12.md", "local_ollama_embedding_feasibility_2026-03-12.md"),
    PacketFile(PROJECT_ROOT / ".sdd" / "researches" / "Embedding_leaderboard.csv", "Embedding_leaderboard.csv"),
    PacketFile(PROJECT_ROOT / "platform_runs" / "warmup" / "exactness_only_candidate_report_v10.json", "exactness_only_candidate_report_v10.json"),
    PacketFile(PROJECT_ROOT / "src" / "shafi" / "core" / "pipeline.py", "pipeline.py"),
    PacketFile(PROJECT_ROOT / "src" / "shafi" / "core" / "retriever.py", "retriever.py"),
    PacketFile(PROJECT_ROOT / "src" / "shafi" / "ingestion" / "parser.py", "parser.py"),
    PacketFile(PROJECT_ROOT / "src" / "shafi" / "ingestion" / "chunker.py", "chunker.py"),
    PacketFile(PROJECT_ROOT / "src" / "shafi" / "core" / "strict_answerer.py", "strict_answerer.py"),
)


def rebuild_packet(upload_dir: Path = DEFAULT_UPLOAD_DIR) -> list[Path]:
    upload_dir.mkdir(parents=True, exist_ok=True)
    inline_sources: dict[Path, bytes] = {}
    for item in PACKET_FILES:
        target = upload_dir / item.target_name
        if item.source.resolve() == target.resolve():
            if not item.source.exists():
                raise FileNotFoundError(f"Missing packet source: {item.source}")
            inline_sources[target.resolve()] = item.source.read_bytes()
    for child in upload_dir.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()

    written: list[Path] = []
    for item in PACKET_FILES:
        target = upload_dir / item.target_name
        resolved_target = target.resolve()
        if item.source.resolve() == resolved_target:
            data = inline_sources.get(resolved_target)
            if data is None:
                raise FileNotFoundError(f"Missing packet inline source: {item.source}")
            target.write_bytes(data)
        else:
            if not item.source.exists():
                raise FileNotFoundError(f"Missing packet source: {item.source}")
            shutil.copy2(item.source, target)
        written.append(target)
    return written


def main() -> int:
    written = rebuild_packet()
    print(f"prepared {len(written)} files in {DEFAULT_UPLOAD_DIR}")
    for path in written:
        print(path.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
