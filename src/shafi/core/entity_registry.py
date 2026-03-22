"""Runtime wrapper around the compiled canonical entity alias registry."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from shafi.ingestion.canonical_entities import EntityAliasResolver


@dataclass(frozen=True, slots=True)
class EnrichedQuery:
    """Resolved canonical entity metadata for one query."""

    query_text: str
    canonical_entity_ids: tuple[str, ...]


class EntityRegistry:
    """Runtime view over a compiler-produced canonical entity registry."""

    def __init__(self, resolver: EntityAliasResolver | None = None) -> None:
        """Initialize the registry wrapper.

        Args:
            resolver: Optional preloaded alias resolver.
        """

        self._resolver = resolver

    @classmethod
    def load(cls, path: str | Path) -> EntityRegistry:
        """Load a runtime registry from disk.

        Args:
            path: Alias-registry JSON path.

        Returns:
            EntityRegistry: Loaded runtime registry.
        """

        resolved_path = Path(path)
        if not resolved_path.exists():
            return cls()
        return cls(EntityAliasResolver.load(resolved_path))

    @classmethod
    def from_resolver(cls, resolver: EntityAliasResolver) -> EntityRegistry:
        """Wrap an in-memory alias resolver.

        Args:
            resolver: Prebuilt alias resolver.

        Returns:
            EntityRegistry: Wrapped registry.
        """

        return cls(resolver)

    def is_loaded(self) -> bool:
        """Report whether a resolver is available.

        Returns:
            bool: True when the registry has a resolver.
        """

        return self._resolver is not None

    def enrich_query(self, query_text: str) -> EnrichedQuery:
        """Resolve canonical entity IDs mentioned in a query.

        Args:
            query_text: Raw query text.

        Returns:
            EnrichedQuery: Canonical query metadata.
        """

        if self._resolver is None:
            return EnrichedQuery(query_text=query_text, canonical_entity_ids=())
        return EnrichedQuery(
            query_text=query_text,
            canonical_entity_ids=tuple(self._resolver.resolve_query_ids(query_text)),
        )

    def resolve_known_values(
        self,
        *,
        law_titles: list[str] | tuple[str, ...] = (),
        case_numbers: list[str] | tuple[str, ...] = (),
        party_names: list[str] | tuple[str, ...] = (),
        authority_names: list[str] | tuple[str, ...] = (),
        judge_names: list[str] | tuple[str, ...] = (),
    ) -> list[str]:
        """Resolve structured metadata values into canonical IDs.

        Args:
            law_titles: Candidate law-title values.
            case_numbers: Candidate case-number values.
            party_names: Candidate party values.
            authority_names: Candidate authority or court values.
            judge_names: Candidate judge values.

        Returns:
            list[str]: Sorted canonical IDs.
        """

        if self._resolver is None:
            return []
        return self._resolver.resolve_known_values(
            law_titles=law_titles,
            case_numbers=case_numbers,
            party_names=party_names,
            authority_names=authority_names,
            judge_names=judge_names,
        )
