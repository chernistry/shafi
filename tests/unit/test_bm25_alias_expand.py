"""Tests for BM25 query alias expansion (shai-gilad-5b)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from rag_challenge.core.bm25_retriever import expand_query_with_aliases, load_alias_map


@pytest.fixture()
def alias_json(tmp_path: Path) -> Path:
    """Write a minimal alias JSON file."""
    data = {
        "clusters": [
            {
                "primary": "Employment Law DIFC Law No. 2 of 2019",
                "variants": [
                    "Employment Law",
                    "EMPLOYMENT LAW",
                    "DIFC Law No. 2 of 2019",
                    "Law No. 2 of 2019",
                    "Employment Law DIFC Law No. 2 of 2019",
                ],
            },
            {
                "primary": "General Partnership Law DIFC Law No. 11 of 2004",
                "variants": [
                    "General Partnership Law",
                    "GENERAL PARTNERSHIP LAW",
                    "DIFC Law No. 11 of 2004",
                    "Law No. 11 of 2004",
                    "General Partnership Law DIFC Law No. 11 of 2004",
                ],
            },
        ]
    }
    p = tmp_path / "legal_aliases.json"
    p.write_text(json.dumps(data))
    return p


class TestLoadAliasMap:
    def test_missing_file_returns_empty(self, tmp_path: Path) -> None:
        result = load_alias_map(tmp_path / "no_such_file.json")
        assert result == {}

    def test_loads_bidirectional_lookup(self, alias_json: Path) -> None:
        alias_map = load_alias_map(alias_json)
        assert len(alias_map) > 0
        # Lowercased variant should map to siblings
        assert "employment law" in alias_map
        siblings = alias_map["employment law"]
        # Should NOT contain itself
        assert "employment law" not in [s.casefold() for s in siblings]
        # Should contain the law number form
        assert any("2 of 2019" in s for s in siblings)

    def test_single_variant_cluster_skipped(self, tmp_path: Path) -> None:
        data = {"clusters": [{"primary": "Lone Law", "variants": ["Lone Law"]}]}
        p = tmp_path / "lone.json"
        p.write_text(json.dumps(data))
        alias_map = load_alias_map(p)
        assert alias_map == {}

    def test_malformed_json_returns_empty(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.json"
        p.write_text("{not valid json")
        assert load_alias_map(p) == {}


class TestExpandQueryWithAliases:
    def test_no_alias_map_returns_original(self) -> None:
        q = "What does Law No. 2 of 2019 say about leave?"
        assert expand_query_with_aliases(q, {}) == q

    def test_expands_law_number_to_name(self, alias_json: Path) -> None:
        alias_map = load_alias_map(alias_json)
        q = "What does DIFC Law No. 2 of 2019 require for annual leave?"
        expanded = expand_query_with_aliases(q, alias_map)
        assert "employment law" in expanded.casefold()

    def test_expands_name_to_law_number(self, alias_json: Path) -> None:
        alias_map = load_alias_map(alias_json)
        q = "Does the Employment Law allow fixed-term contracts?"
        expanded = expand_query_with_aliases(q, alias_map)
        assert "2 of 2019" in expanded or "law no. 2" in expanded.casefold()

    def test_no_match_returns_original(self, alias_json: Path) -> None:
        alias_map = load_alias_map(alias_json)
        q = "What is the capital of France?"
        assert expand_query_with_aliases(q, alias_map) == q

    def test_does_not_duplicate_existing_variants(self, alias_json: Path) -> None:
        alias_map = load_alias_map(alias_json)
        # Query already mentions both forms — no duplication
        q = "What does Employment Law DIFC Law No. 2 of 2019 require?"
        expanded = expand_query_with_aliases(q, alias_map)
        # Should not add terms that are already present
        lower_q = q.casefold()
        lower_exp = expanded.casefold()
        # Extra portion should not repeat what's already there
        extra = lower_exp[len(lower_q) :]
        assert "employment law" not in extra

    def test_multiple_clusters_expanded(self, alias_json: Path) -> None:
        alias_map = load_alias_map(alias_json)
        q = "Compare Employment Law and General Partnership Law"
        expanded = expand_query_with_aliases(q, alias_map)
        assert "2 of 2019" in expanded or "law no. 2" in expanded.casefold()
        assert "11 of 2004" in expanded or "law no. 11" in expanded.casefold()
