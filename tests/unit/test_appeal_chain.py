"""Unit tests for DIFCAppealChainDetector and helpers."""

from __future__ import annotations

from shafi.core.pipeline.appeal_chain import (
    DIFCAppealChainDetector,
    _extract_case_refs,
    is_appeal_query,
)

# ---------------------------------------------------------------------------
# Helper fixtures
# ---------------------------------------------------------------------------


def _chunks(*triples: tuple[str, str, str]) -> list[tuple[str, str, str]]:
    """Convenience wrapper so test bodies stay compact."""
    return list(triples)


# ---------------------------------------------------------------------------
# _extract_case_refs
# ---------------------------------------------------------------------------


class TestExtractCaseRefs:
    def test_extracts_canonical_refs(self) -> None:
        text = "This appeal arises from SCT 133/2025 and CFI 070/2025."
        assert _extract_case_refs(text) == ["SCT 133/2025", "CFI 070/2025"]

    def test_deduplicates(self) -> None:
        text = "SCT 133/2025 was appealed; see SCT 133/2025 judgment."
        assert _extract_case_refs(text) == ["SCT 133/2025"]

    def test_case_insensitive_dedup(self) -> None:
        text = "SCT 133/2025 and sct 133/2025"
        assert len(_extract_case_refs(text)) == 1

    def test_empty_text(self) -> None:
        assert _extract_case_refs("") == []

    def test_no_refs(self) -> None:
        assert _extract_case_refs("There are no case refs here.") == []


# ---------------------------------------------------------------------------
# is_appeal_query
# ---------------------------------------------------------------------------


class TestIsAppealQuery:
    def test_detected_with_appealed(self) -> None:
        assert is_appeal_query("Was SCT 133/2025 appealed to the CFI?")

    def test_detected_with_appeal(self) -> None:
        assert is_appeal_query("Is there an appeal from SCT 459/2024?")

    def test_detected_with_affirmed(self) -> None:
        assert is_appeal_query("Was the decision affirmed on appeal?")

    def test_not_detected_plain(self) -> None:
        assert not is_appeal_query("Who is the registrar of CFI 070/2025?")

    def test_empty_query(self) -> None:
        assert not is_appeal_query("")


# ---------------------------------------------------------------------------
# DIFCAppealChainDetector.build_from_chunks
# ---------------------------------------------------------------------------


class TestBuildFromChunks:
    def test_not_built_before_build(self) -> None:
        d = DIFCAppealChainDetector()
        assert not d.built

    def test_built_after_build(self) -> None:
        d = DIFCAppealChainDetector()
        d.build_from_chunks([])
        assert d.built

    def test_builds_sct_to_cfi_chain(self) -> None:
        """CFI judgment that cites SCT 133/2025 → SCT 133 was appealed."""
        chunks = _chunks(
            ("doc-sct133", "SCT 133/2025 Smith v Jones", "Judgment on merits."),
            (
                "doc-cfi070",
                "CFI 070/2025 Smith v Jones",
                "This is an appeal from SCT 133/2025. Permission was granted.",
            ),
        )
        d = DIFCAppealChainDetector()
        d.build_from_chunks(chunks)
        assert d.get_appellate_case_ref("SCT 133/2025") == "CFI 070/2025"

    def test_builds_cfi_to_ca_chain(self) -> None:
        """CA judgment that cites CFI case → CFI was appealed."""
        chunks = _chunks(
            ("doc-cfi", "CFI 069/2024 Alpha v Beta", "Judgment."),
            (
                "doc-ca",
                "CA 006/2024 Alpha v Beta",
                "The Court of Appeal in CFI 069/2024 held that... appeal allowed.",
            ),
        )
        d = DIFCAppealChainDetector()
        d.build_from_chunks(chunks)
        assert d.get_appellate_case_ref("CFI 069/2024") == "CA 006/2024"

    def test_no_chain_for_same_level(self) -> None:
        """CFI docs citing other CFI cases are not appeal chains."""
        chunks = _chunks(
            ("doc-a", "CFI 001/2024 X v Y", "This follows the reasoning in CFI 002/2024."),
            ("doc-b", "CFI 002/2024 A v B", "Judgment."),
        )
        d = DIFCAppealChainDetector()
        d.build_from_chunks(chunks)
        assert d.get_appellate_case_ref("CFI 002/2024") is None

    def test_lower_court_doc_not_appellate(self) -> None:
        """SCT doc citing CFI doesn't create a reverse chain."""
        chunks = _chunks(
            ("doc-sct", "SCT 459/2024 X v Y", "Parties relied on CFI 010/2023."),
        )
        d = DIFCAppealChainDetector()
        d.build_from_chunks(chunks)
        assert d.get_appellate_case_ref("CFI 010/2023") is None

    def test_ignores_empty_doc_id(self) -> None:
        chunks = _chunks(("", "SCT 100/2024 X v Y", "Judgment."))
        d = DIFCAppealChainDetector()
        d.build_from_chunks(chunks)  # should not raise
        assert d.built


# ---------------------------------------------------------------------------
# DIFCAppealChainDetector.expand_doc_refs
# ---------------------------------------------------------------------------


class TestExpandDocRefs:
    def _detector_with_sct133(self) -> DIFCAppealChainDetector:
        d = DIFCAppealChainDetector()
        d.build_from_chunks(
            _chunks(
                ("doc-sct133", "SCT 133/2025 Smith v Jones", "Judgment."),
                (
                    "doc-cfi070",
                    "CFI 070/2025 Smith v Jones",
                    "Appeal from SCT 133/2025 — affirmed.",
                ),
            )
        )
        return d

    def test_expands_for_appeal_query(self) -> None:
        d = self._detector_with_sct133()
        refs = d.expand_doc_refs(["SCT 133/2025"], query="Was SCT 133/2025 appealed?")
        assert "CFI 070/2025" in refs

    def test_preserves_original_first(self) -> None:
        d = self._detector_with_sct133()
        refs = d.expand_doc_refs(["SCT 133/2025"], query="Was SCT 133/2025 appealed?")
        assert refs[0] == "SCT 133/2025"
        assert refs[1] == "CFI 070/2025"

    def test_no_expansion_non_appeal_query(self) -> None:
        d = self._detector_with_sct133()
        refs = d.expand_doc_refs(["SCT 133/2025"], query="Who is the judge in SCT 133/2025?")
        assert refs == ["SCT 133/2025"]

    def test_no_expansion_not_built(self) -> None:
        d = DIFCAppealChainDetector()
        refs = d.expand_doc_refs(["SCT 133/2025"], query="Was SCT 133/2025 appealed?")
        assert refs == ["SCT 133/2025"]

    def test_no_expansion_no_chain(self) -> None:
        d = self._detector_with_sct133()
        refs = d.expand_doc_refs(["SCT 999/2025"], query="Was SCT 999/2025 appealed?")
        assert refs == ["SCT 999/2025"]

    def test_no_duplicate_expansion(self) -> None:
        d = self._detector_with_sct133()
        refs = d.expand_doc_refs(["SCT 133/2025", "CFI 070/2025"], query="Was SCT 133/2025 appealed?")
        # CFI 070/2025 was already present — no duplication
        assert refs.count("CFI 070/2025") == 1
