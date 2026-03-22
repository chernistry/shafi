"""Database-first field answering over the compiled closed-world registry."""

from __future__ import annotations

import json
import re
from pathlib import Path

from shafi.core.entity_registry import EntityRegistry
from shafi.core.field_lookup import FieldAnswer, FieldLookupTable, FieldType
from shafi.core.query_contract import PredicateType, QueryContract
from shafi.models.legal_objects import CorpusRegistry

_SLUG_CLEAN_RE = re.compile(r"[^a-z0-9]+")


class DatabaseAnswerer:
    """Resolve field-like questions directly from compiled structured data."""

    def __init__(
        self,
        *,
        lookup_table: FieldLookupTable | None = None,
        entity_registry: EntityRegistry | None = None,
        confidence_threshold: float = 0.85,
    ) -> None:
        """Initialize the database answerer.

        Args:
            lookup_table: Preloaded field lookup table.
            entity_registry: Optional runtime entity registry for fallback entity
                resolution.
            confidence_threshold: Minimum confidence required to bypass RAG.
        """

        self._lookup_table = lookup_table or FieldLookupTable()
        self._entity_registry = entity_registry or EntityRegistry()
        self._confidence_threshold = float(confidence_threshold)
        self._slug_to_canonical: dict[str, str] = self._build_slug_index(self._lookup_table)

    @classmethod
    def from_paths(
        cls,
        *,
        lookup_table_path: str | Path,
        alias_registry_path: str | Path = "",
        confidence_threshold: float = 0.85,
    ) -> DatabaseAnswerer:
        """Load a database answerer from persisted runtime artifacts.

        Args:
            lookup_table_path: Persisted corpus-registry or field-lookup JSON path.
            alias_registry_path: Optional alias-registry JSON path.
            confidence_threshold: Minimum confidence required to bypass RAG.

        Returns:
            DatabaseAnswerer: Runtime-ready database answerer.
        """

        resolved_lookup_path = Path(lookup_table_path)
        lookup_table = cls._load_lookup_table(resolved_lookup_path)
        entity_reg = (
            EntityRegistry.load(alias_registry_path)
            if alias_registry_path and str(alias_registry_path).strip() and Path(alias_registry_path).is_file()
            else EntityRegistry()
        )
        return cls(
            lookup_table=lookup_table,
            entity_registry=entity_reg,
            confidence_threshold=confidence_threshold,
        )

    def is_loaded(self) -> bool:
        """Report whether the answerer has any structured field data.

        Returns:
            bool: True when the lookup table is populated.
        """

        return bool(self._lookup_table.table)

    def answer(self, contract: QueryContract) -> FieldAnswer | None:
        """Attempt a deterministic field answer from structured data.

        Args:
            contract: Compiled query contract for the current request.

        Returns:
            FieldAnswer | None: Structured field answer, or ``None`` when the
            query should fall back to standard RAG.
        """

        if contract.predicate is not PredicateType.LOOKUP_FIELD:
            return None
        field_type = self._resolve_field_type(contract.field_name)
        if field_type is None:
            return None
        for canonical_id in self.resolve_target_entity_ids(contract):
            answer = self.lookup_field(canonical_id, field_type)
            if answer is None or self.should_fallback(answer):
                continue
            return answer
        return None

    def resolve_target_entity_ids(self, contract: QueryContract) -> list[str]:
        """Resolve candidate primary entities for a field lookup contract.

        Args:
            contract: Compiled query contract.

        Returns:
            list[str]: Candidate canonical entity IDs in lookup order.
        """

        if contract.primary_entities:
            primary = contract.primary_entities[0]
            return self._candidate_entity_ids(
                canonical_id=primary.canonical_id,
                entity_type=primary.entity_type,
                source_doc_ids=primary.source_doc_ids,
            )
        enriched = self._entity_registry.enrich_query(contract.query_text)
        candidates: list[str] = []
        for canonical_id in enriched.canonical_entity_ids:
            candidates.extend(self._candidate_entity_ids(canonical_id=canonical_id))
        return candidates

    def lookup_field(self, canonical_id: str, field_type: FieldType) -> FieldAnswer | None:
        """Look up one structured field for a canonical entity.

        Args:
            canonical_id: Canonical entity ID.
            field_type: Normalized field type.

        Returns:
            FieldAnswer | None: Matching structured answer if present.
        """

        answer = self._lookup_table.lookup(canonical_id, field_type)
        if answer is not None:
            return answer
        for fallback_id in self._fallback_canonical_ids(canonical_id):
            answer = self._lookup_table.lookup(fallback_id, field_type)
            if answer is not None:
                return answer
        return None

    @staticmethod
    def format_answer(field_answer: FieldAnswer, answer_type: str) -> str:
        """Format a field answer for the requested response type.

        Args:
            field_answer: Structured field answer from the lookup table.
            answer_type: Requested answer type from the query payload.

        Returns:
            str: Final answer text ready for emission.
        """

        normalized_answer_type = answer_type.strip().lower()
        value = field_answer.value.strip()
        if not value:
            return ""
        if normalized_answer_type == "boolean":
            return "Yes"
        if normalized_answer_type in {"name", "date", "number"}:
            head, *_tail = [part.strip() for part in value.split(",") if part.strip()]
            return head if head else value
        return value

    def should_fallback(self, field_answer: FieldAnswer) -> bool:
        """Decide whether a structured answer is too weak to trust directly.

        Args:
            field_answer: Candidate structured answer.

        Returns:
            bool: True when standard RAG should be used instead.
        """

        if not field_answer.value.strip():
            return True
        return float(field_answer.confidence) < self._confidence_threshold

    @staticmethod
    def _resolve_field_type(field_name: str) -> FieldType | None:
        """Normalize query-contract field names into lookup-table field types.

        Args:
            field_name: Raw field name from the query contract.

        Returns:
            FieldType | None: Normalized field type or ``None`` when unsupported.
        """

        normalized = field_name.strip().lower()
        mapping = {
            "title": FieldType.TITLE,
            "issued_by": FieldType.ISSUED_BY,
            "authority": FieldType.AUTHORITY,
            "date": FieldType.DATE,
            "commencement_date": FieldType.COMMENCEMENT_DATE,
            "law_number": FieldType.LAW_NUMBER,
            "claimant": FieldType.CLAIMANT,
            "respondent": FieldType.RESPONDENT,
            "party": FieldType.PARTY,
            "judge": FieldType.JUDGE,
            "case_number": FieldType.CASE_NUMBER,
            "outcome": FieldType.OUTCOME,
        }
        return mapping.get(normalized)

    @staticmethod
    def _fallback_canonical_ids(canonical_id: str) -> list[str]:
        """Return conservative fallback entity IDs for lookup retries.

        Args:
            canonical_id: Canonical entity ID from the contract.

        Returns:
            list[str]: Candidate fallback IDs in retry order.
        """

        return []

    @staticmethod
    def _build_slug_index(lookup_table: FieldLookupTable) -> dict[str, str]:
        """Build a reverse index from normalized doc-id slugs to canonical IDs.

        Maps normalized case numbers (e.g. ``cfi_057_2025`` or ``cfi0572025``)
        and law number slugs to their canonical lookup-table keys so that
        source_doc_ids produced by the query-contract compiler can be resolved
        to the SHA256-based keys used in the lookup table.

        Args:
            lookup_table: Populated field lookup table.

        Returns:
            dict[str, str]: Slug-to-canonical-ID mapping.
        """

        index: dict[str, str] = {}
        for canonical_id, fields in lookup_table.table.items():
            # Index by case_number field for case entities
            cn_field = fields.get("case_number")
            if cn_field and cn_field.value.strip():
                raw = cn_field.value.strip()
                # Underscore-separated: cfi_057_2025
                slug_underscore = raw.lower().replace(" ", "_").replace("/", "_")
                index.setdefault(slug_underscore, canonical_id)
                # Compact: cfi0572025 (as produced by query contract canonical IDs)
                slug_compact = _SLUG_CLEAN_RE.sub("", raw.lower())
                index.setdefault(slug_compact, canonical_id)

            # Index by law_number field for law/order entities
            ln_field = fields.get("law_number")
            if ln_field and ln_field.value.strip():
                raw = ln_field.value.strip()
                slug_underscore = raw.lower().replace(" ", "_").replace("/", "_")
                index.setdefault(slug_underscore, canonical_id)
                slug_compact = _SLUG_CLEAN_RE.sub("", raw.lower())
                index.setdefault(slug_compact, canonical_id)

            # Index by raw doc_id extracted from canonical_id (e.g. "case:abcdef" -> "abcdef")
            if ":" in canonical_id:
                raw_doc_id = canonical_id.split(":", 1)[1]
                index.setdefault(raw_doc_id, canonical_id)

        return index

    def _resolve_doc_id_via_slug(self, doc_id: str) -> str | None:
        """Resolve a source_doc_id slug to a canonical lookup-table key.

        Args:
            doc_id: Raw doc_id from query contract (may be a case-number slug).

        Returns:
            str | None: Matching canonical ID, or ``None`` when not found.
        """

        # Direct hit
        hit = self._slug_to_canonical.get(doc_id)
        if hit:
            return hit
        # Try compact form (strip non-alphanumeric)
        compact = _SLUG_CLEAN_RE.sub("", doc_id.lower())
        return self._slug_to_canonical.get(compact)

    def _candidate_entity_ids(
        self,
        *,
        canonical_id: str,
        entity_type: str = "",
        source_doc_ids: list[str] | None = None,
    ) -> list[str]:
        """Expand contract entities into lookup-table candidates.

        Args:
            canonical_id: Raw canonical ID from the resolver.
            entity_type: Resolver entity type label.
            source_doc_ids: Source documents attached to the resolver entity.

        Returns:
            list[str]: Candidate lookup-table IDs in deterministic priority order.
        """

        candidates: list[str] = []
        if canonical_id and canonical_id in self._lookup_table.table:
            candidates.append(canonical_id)
        # Try slug index for the canonical_id itself (e.g. "case_number:cfi0572025")
        if not candidates and canonical_id:
            slug_key = canonical_id.split(":", 1)[-1] if ":" in canonical_id else canonical_id
            resolved = self._resolve_doc_id_via_slug(slug_key)
            if resolved and resolved not in candidates:
                candidates.append(resolved)

        normalized_entity_type = entity_type.strip().lower()
        doc_ids = [doc_id for doc_id in (source_doc_ids or []) if doc_id]
        if not doc_ids:
            return candidates
        if normalized_entity_type.startswith("law"):
            prefixes = ("law", "order", "practice_direction")
        elif normalized_entity_type.startswith("case") or normalized_entity_type in {"judge", "party"}:
            prefixes = ("case",)
        elif normalized_entity_type == "authority":
            prefixes = ("law", "order", "practice_direction")
        else:
            prefixes = ("law", "case", "order", "practice_direction")
        for doc_id in doc_ids:
            # First try direct prefix:doc_id match (works when doc_id is SHA256)
            for prefix in prefixes:
                candidate = f"{prefix}:{doc_id}"
                if candidate in self._lookup_table.table and candidate not in candidates:
                    candidates.append(candidate)
            # Fallback: resolve via slug index (case-number or law-number slugs)
            resolved = self._resolve_doc_id_via_slug(doc_id)
            if resolved and resolved not in candidates:
                candidates.append(resolved)
        return candidates

    @staticmethod
    def _load_lookup_table(path: Path) -> FieldLookupTable:
        """Load a lookup table or derive one from a persisted corpus registry.

        Args:
            path: Artifact path on disk.

        Returns:
            FieldLookupTable: Loaded or derived lookup table.
        """

        if not path.exists():
            return FieldLookupTable()
        raw = path.read_text(encoding="utf-8")
        payload = json.loads(raw)
        if isinstance(payload, dict) and "table" in payload:
            return FieldLookupTable.model_validate(payload)
        registry = CorpusRegistry.model_validate(payload)
        return FieldLookupTable.build_from_registry(registry)
