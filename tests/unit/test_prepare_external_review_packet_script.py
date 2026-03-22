from __future__ import annotations

from typing import TYPE_CHECKING

from scripts.prepare_external_review_packet import PacketFile, rebuild_packet

if TYPE_CHECKING:
    from pathlib import Path


def test_rebuild_packet_clears_and_copies_files(tmp_path: Path) -> None:
    src_a = tmp_path / "a.md"
    src_b = tmp_path / "b.json"
    src_a.write_text("alpha", encoding="utf-8")
    src_b.write_text('{"ok": true}', encoding="utf-8")

    upload = tmp_path / "upload"
    upload.mkdir()
    (upload / "old.txt").write_text("stale", encoding="utf-8")

    # Use same-source/target file for one entry to validate in-place rewrite path.
    prompt = upload / "prompt.md"
    prompt.write_text("prompt", encoding="utf-8")

    original = (
        PacketFile(prompt, "prompt.md"),
        PacketFile(src_a, "summary.md"),
        PacketFile(src_b, "data.json"),
    )

    # Local helper instead of mutating module-level constants globally.
    def local_rebuild() -> list[Path]:
        from scripts import prepare_external_review_packet as module

        old = module.PACKET_FILES
        try:
            module.PACKET_FILES = original
            return rebuild_packet(upload)
        finally:
            module.PACKET_FILES = old

    written = local_rebuild()

    assert [path.name for path in written] == ["prompt.md", "summary.md", "data.json"]
    assert not (upload / "old.txt").exists()
    assert (upload / "prompt.md").read_text(encoding="utf-8") == "prompt"
    assert (upload / "summary.md").read_text(encoding="utf-8") == "alpha"
    assert (upload / "data.json").read_text(encoding="utf-8") == '{"ok": true}'
