"""Canonical alias clustering for compiled legal entities."""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, cast

from shafi.core.law_notice_support import (
    extract_law_number_year,
    is_law_like_order_title,
    normalize_law_like_title,
)
from shafi.core.legal_title_family import derive_law_title_aliases, title_key

if TYPE_CHECKING:
    from shafi.models.legal_objects import CorpusRegistry

_SPACE_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^a-z0-9]+")
_CASE_NUMBER_RE = re.compile(r"\b(?:CFI|CA|ARB|SCT|TCD|ENF|DEC)\s*\d{1,4}/\d{4}\b", re.IGNORECASE)
_NUMBERED_LAW_RE = re.compile(r"\bLaw\s+No\.?\s*(\d+)\s+of\s+(\d{4})\b", re.IGNORECASE)
_TITLE_PREFIX_RE = re.compile(r"^(?:justice|judge|chief justice|sir|registrar)\s+", re.IGNORECASE)
_TRAILING_J_RE = re.compile(r"\bJ\b\.?$", re.IGNORECASE)
_SHORT_TOKEN_RE = re.compile(r"^[A-Z]{2,8}$")
_ROLE_ALIAS_RE = re.compile(
    r"^(?:the\s+)?(?:claimant|respondent|appellant|applicant|defendant|plaintiff|petitioner)$",
    re.IGNORECASE,
)


class CanonicalEntityType(StrEnum):
    """Supported canonical alias categories."""

    LAW_TITLE = "law_title"
    CASE_NUMBER = "case_number"
    PARTY = "party"
    JUDGE = "judge"
    AUTHORITY = "authority"


@dataclass(frozen=True, slots=True)
class AliasCluster:
    """One canonical alias cluster exported by the compiler branch."""

    canonical_id: str
    canonical_form: str
    entity_type: CanonicalEntityType
    aliases: tuple[str, ...]
    source_doc_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        """Serialize the cluster into a JSON-safe dictionary.

        Returns:
            dict[str, object]: Serialized cluster payload.
        """

        payload = asdict(self)
        payload["entity_type"] = self.entity_type.value
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> AliasCluster:
        """Deserialize a cluster from JSON-safe data.

        Args:
            payload: Serialized cluster payload.

        Returns:
            AliasCluster: Reconstructed alias cluster.
        """

        alias_items = _coerce_object_list(payload.get("aliases", []))
        source_doc_id_items = _coerce_object_list(payload.get("source_doc_ids", []))
        aliases = tuple(str(value) for value in alias_items if str(value).strip())
        source_doc_ids = tuple(str(value) for value in source_doc_id_items if str(value).strip())
        return cls(
            canonical_id=str(payload.get("canonical_id") or ""),
            canonical_form=str(payload.get("canonical_form") or ""),
            entity_type=CanonicalEntityType(str(payload.get("entity_type") or CanonicalEntityType.PARTY.value)),
            aliases=aliases,
            source_doc_ids=source_doc_ids,
        )


@dataclass(slots=True)
class _MutableAliasBucket:
    """Mutable bucket used while assembling alias clusters."""

    canonical_form: str
    aliases: set[str]
    source_doc_ids: set[str]


def _clean_text(text: str) -> str:
    """Normalize whitespace and Unicode noise in entity text.

    Args:
        text: Raw text value.

    Returns:
        str: Cleaned text.
    """

    cleaned = unicodedata.normalize("NFKC", text or "")
    cleaned = _SPACE_RE.sub(" ", cleaned).strip(" ,.;:-")
    return cleaned


def _coerce_object_list(value: object) -> list[object]:
    """Coerce a JSON-like value into a list for strict typing.

    Args:
        value: Raw decoded JSON value.

    Returns:
        list[object]: Value as a list or an empty list.
    """

    if not isinstance(value, list):
        return []
    items_obj = cast("list[object]", value)
    items: list[object] = []
    for item in items_obj:
        items.append(item)
    return items


def _coerce_payload_dict(value: object) -> dict[str, object]:
    """Coerce a JSON-like value into a string-keyed dictionary.

    Args:
        value: Raw decoded JSON value.

    Returns:
        dict[str, object]: String-keyed dictionary or an empty dict.
    """

    if not isinstance(value, dict):
        return {}
    value_obj = cast("dict[object, object]", value)
    payload: dict[str, object] = {}
    for key, item in value_obj.items():
        if isinstance(key, str):
            payload[key] = item
    return payload


def _lookup_key(text: str) -> str:
    """Build a stable lookup key for free-form entity text.

    Args:
        text: Raw entity text.

    Returns:
        str: Comparison key.
    """

    cleaned = _clean_text(text).casefold()
    cleaned = _PUNCT_RE.sub(" ", cleaned)
    return _SPACE_RE.sub(" ", cleaned).strip()


def _slugify(text: str) -> str:
    """Build a stable slug for canonical entity IDs.

    Args:
        text: Raw entity key.

    Returns:
        str: URL-safe-ish slug.
    """

    slug = _PUNCT_RE.sub("-", _clean_text(text).casefold())
    slug = re.sub(r"-+", "-", slug).strip("-")
    return slug or "unknown"


def _case_number_key(text: str) -> str:
    """Normalize case-number text into a compact comparison key.

    Args:
        text: Raw case number.

    Returns:
        str: Stable case-number key.
    """

    return _lookup_key(text).replace(" ", "")


def _person_key(text: str) -> str:
    """Normalize a personal name into a canonical lookup key.

    Args:
        text: Raw person name.

    Returns:
        str: Stable personal-name key.
    """

    cleaned = _TITLE_PREFIX_RE.sub("", _clean_text(text))
    cleaned = _TRAILING_J_RE.sub("", cleaned)
    return _lookup_key(cleaned)


def _organization_key(text: str) -> str:
    """Normalize an organization-like surface form into a stable key.

    Args:
        text: Raw organization text.

    Returns:
        str: Stable organization lookup key.
    """

    cleaned = re.sub(r"^(?:the)\s+", "", _clean_text(text), flags=re.IGNORECASE)
    return _lookup_key(cleaned)


def _generate_acronym(text: str) -> str:
    """Generate an acronym alias for multi-token organization names.

    Args:
        text: Source organization text.

    Returns:
        str: Acronym alias or empty string.
    """

    tokens = [token for token in re.split(r"[^A-Za-z0-9]+", _clean_text(text)) if token]
    significant = [token for token in tokens if len(token) > 2 or token.isupper()]
    if len(significant) < 2:
        return ""
    acronym = "".join(token[0].upper() for token in significant if token[0].isalnum())
    return acronym if 3 <= len(acronym) <= 8 else ""


def _numbered_law_alias(title: str, law_number: str, year: str) -> list[str]:
    """Create numbered law aliases from structured law metadata.

    Args:
        title: Full law title.
        law_number: Parsed law number.
        year: Parsed law year.

    Returns:
        list[str]: Extra aliases derived from numbering.
    """

    if not law_number or not year:
        return []
    base_aliases = [
        f"Law No. {law_number} of {year}",
        f"{title} No. {law_number} of {year}",
    ]
    return [_clean_text(value) for value in base_aliases if _clean_text(value)]


def _law_title_lookup_key(text: str) -> str:
    """Build a lookup key for law-title aliases.

    Args:
        text: Raw law-title alias text.

    Returns:
        str: Exact numbered-law key when available, otherwise the family key.
    """

    cleaned = _clean_text(text)
    numbered_match = _NUMBERED_LAW_RE.search(cleaned)
    if numbered_match is not None:
        return f"law-no:{numbered_match.group(1)}:{numbered_match.group(2)}"
    return title_key(cleaned)


def _judge_aliases(name: str) -> set[str]:
    """Build conservative alias variants for a judicial name.

    Args:
        name: Raw judge name.

    Returns:
        set[str]: Alias candidates.
    """

    cleaned = _clean_text(name)
    bare = _TITLE_PREFIX_RE.sub("", cleaned)
    parts = [part for part in bare.split() if part]
    aliases = {cleaned, bare}
    if parts:
        aliases.add(f"{parts[-1]} J")
        aliases.add(f"Justice {parts[-1]}")
    return {value for value in aliases if value}


def _case_number_aliases(case_number: str) -> set[str]:
    """Build conservative aliases for a case number.

    Args:
        case_number: Raw case number.

    Returns:
        set[str]: Alias candidates.
    """

    cleaned = _clean_text(case_number)
    compact = cleaned.replace(" ", "")
    spaced = re.sub(r"(?i)\b([A-Z]{2,4})(\d)", r"\1 \2", compact)
    return {value for value in {cleaned, compact, spaced} if value}


def _organization_aliases(name: str) -> set[str]:
    """Build organization aliases including a conservative acronym.

    Args:
        name: Raw organization text.

    Returns:
        set[str]: Alias candidates.
    """

    cleaned = _clean_text(name)
    aliases = {cleaned}
    acronym = _generate_acronym(cleaned)
    if acronym:
        aliases.add(acronym)
    stripped = re.sub(r"^(?:the)\s+", "", cleaned, flags=re.IGNORECASE)
    if stripped:
        aliases.add(stripped)
    return {value for value in aliases if value}


def _make_cluster(
    *,
    entity_type: CanonicalEntityType,
    key: str,
    canonical_form: str,
    aliases: set[str],
    source_doc_ids: set[str],
) -> AliasCluster:
    """Create an immutable alias cluster.

    Args:
        entity_type: Alias category.
        key: Stable normalized key.
        canonical_form: Preferred display form.
        aliases: Alias set.
        source_doc_ids: Source documents contributing aliases.

    Returns:
        AliasCluster: Immutable alias cluster record.
    """

    canonical_id = f"{entity_type.value}:{_slugify(key)}"
    return AliasCluster(
        canonical_id=canonical_id,
        canonical_form=_clean_text(canonical_form),
        entity_type=entity_type,
        aliases=tuple(sorted({_clean_text(alias) for alias in aliases if _clean_text(alias)})),
        source_doc_ids=tuple(sorted(doc_id for doc_id in source_doc_ids if doc_id)),
    )


def build_law_aliases(registry: CorpusRegistry) -> list[AliasCluster]:
    """Build canonical law-title alias clusters from a compiled registry.

    Args:
        registry: Compiled corpus registry.

    Returns:
        list[AliasCluster]: Law-title alias clusters.
    """

    buckets: dict[str, _MutableAliasBucket] = {}
    for law in registry.laws.values():
        aliases = set(derive_law_title_aliases(law.title, law.short_title))
        aliases.update(_numbered_law_alias(law.short_title or law.title, law.law_number, law.year))
        aliases.update(_clean_text(alias) for alias in law.aliases if _clean_text(alias))
        cluster_key = title_key(law.short_title or law.title or law.doc_id)
        bucket = buckets.setdefault(
            cluster_key,
            _MutableAliasBucket(
                canonical_form=law.short_title or law.title,
                aliases=set(),
                source_doc_ids=set(),
            ),
        )
        bucket.aliases.update(aliases)
        bucket.source_doc_ids.add(law.doc_id)
    for order in registry.orders.values():
        if not is_law_like_order_title(title=order.title, source_text=order.source_text):
            continue
        normalized_title = normalize_law_like_title(title=order.title, source_text=order.source_text)
        law_number, year = extract_law_number_year(title=order.title, source_text=order.source_text)
        aliases = set(derive_law_title_aliases(normalized_title, normalized_title))
        aliases.update(_numbered_law_alias(normalized_title, law_number, year))
        aliases.update(_clean_text(alias) for alias in order.aliases if _clean_text(alias))
        cluster_key = title_key(normalized_title or order.doc_id)
        bucket = buckets.setdefault(
            cluster_key,
            _MutableAliasBucket(
                canonical_form=normalized_title or order.doc_id,
                aliases=set(),
                source_doc_ids=set(),
            ),
        )
        bucket.aliases.update(aliases)
        bucket.source_doc_ids.add(order.doc_id)
    return [
        _make_cluster(
            entity_type=CanonicalEntityType.LAW_TITLE,
            key=key,
            canonical_form=bucket.canonical_form,
            aliases=bucket.aliases,
            source_doc_ids=bucket.source_doc_ids,
        )
        for key, bucket in sorted(buckets.items())
    ]


def build_case_aliases(registry: CorpusRegistry) -> list[AliasCluster]:
    """Build canonical case-number alias clusters from a compiled registry.

    Args:
        registry: Compiled corpus registry.

    Returns:
        list[AliasCluster]: Case-number alias clusters.
    """

    buckets: dict[str, _MutableAliasBucket] = {}
    for case in registry.cases.values():
        if not case.case_number:
            continue
        key = _case_number_key(case.case_number)
        bucket = buckets.setdefault(
            key,
            _MutableAliasBucket(canonical_form=case.case_number, aliases=set(), source_doc_ids=set()),
        )
        bucket.aliases.update(_case_number_aliases(case.case_number))
        bucket.aliases.update(_clean_text(alias) for alias in case.aliases if _clean_text(alias))
        bucket.source_doc_ids.add(case.doc_id)
    return [
        _make_cluster(
            entity_type=CanonicalEntityType.CASE_NUMBER,
            key=key,
            canonical_form=bucket.canonical_form,
            aliases=bucket.aliases,
            source_doc_ids=bucket.source_doc_ids,
        )
        for key, bucket in sorted(buckets.items())
    ]


def build_party_aliases(registry: CorpusRegistry) -> list[AliasCluster]:
    """Build canonical party alias clusters from a compiled registry.

    Args:
        registry: Compiled corpus registry.

    Returns:
        list[AliasCluster]: Party alias clusters.
    """

    buckets: dict[str, _MutableAliasBucket] = {}
    for case in registry.cases.values():
        for party in case.parties:
            if not party.name:
                continue
            key = _organization_key(party.name)
            bucket = buckets.setdefault(
                key,
                _MutableAliasBucket(canonical_form=party.name, aliases=set(), source_doc_ids=set()),
            )
            bucket.aliases.update(_organization_aliases(party.name))
            if party.role and not _ROLE_ALIAS_RE.match(party.role):
                bucket.aliases.add(_clean_text(party.role))
            bucket.source_doc_ids.add(case.doc_id)
    return [
        _make_cluster(
            entity_type=CanonicalEntityType.PARTY,
            key=key,
            canonical_form=bucket.canonical_form,
            aliases=bucket.aliases,
            source_doc_ids=bucket.source_doc_ids,
        )
        for key, bucket in sorted(buckets.items())
    ]


def build_judge_aliases(registry: CorpusRegistry) -> list[AliasCluster]:
    """Build canonical judge alias clusters from a compiled registry.

    Args:
        registry: Compiled corpus registry.

    Returns:
        list[AliasCluster]: Judge alias clusters.
    """

    buckets: dict[str, _MutableAliasBucket] = {}
    for case in registry.cases.values():
        for judge in case.judges:
            if not judge:
                continue
            key = _person_key(judge)
            bucket = buckets.setdefault(
                key,
                _MutableAliasBucket(
                    canonical_form=_clean_text(judge),
                    aliases=set(),
                    source_doc_ids=set(),
                ),
            )
            bucket.aliases.update(_judge_aliases(judge))
            bucket.source_doc_ids.add(case.doc_id)
    return [
        _make_cluster(
            entity_type=CanonicalEntityType.JUDGE,
            key=key,
            canonical_form=bucket.canonical_form,
            aliases=bucket.aliases,
            source_doc_ids=bucket.source_doc_ids,
        )
        for key, bucket in sorted(buckets.items())
    ]


def build_authority_aliases(registry: CorpusRegistry) -> list[AliasCluster]:
    """Build canonical authority alias clusters from a compiled registry.

    Args:
        registry: Compiled corpus registry.

    Returns:
        list[AliasCluster]: Authority alias clusters.
    """

    buckets: dict[str, _MutableAliasBucket] = {}

    def add_authority(name: str, doc_id: str) -> None:
        if not name:
            return
        key = _organization_key(name)
        bucket = buckets.setdefault(
            key,
            _MutableAliasBucket(
                canonical_form=_clean_text(name),
                aliases=set(),
                source_doc_ids=set(),
            ),
        )
        bucket.aliases.update(_organization_aliases(name))
        bucket.source_doc_ids.add(doc_id)

    for law in registry.laws.values():
        add_authority(law.issuing_authority, law.doc_id)
    for order in registry.orders.values():
        add_authority(order.issued_by, order.doc_id)
    for direction in registry.practice_directions.values():
        add_authority(direction.issued_by, direction.doc_id)
    for case in registry.cases.values():
        add_authority(case.court, case.doc_id)

    return [
        _make_cluster(
            entity_type=CanonicalEntityType.AUTHORITY,
            key=key,
            canonical_form=bucket.canonical_form,
            aliases=bucket.aliases,
            source_doc_ids=bucket.source_doc_ids,
        )
        for key, bucket in sorted(buckets.items())
    ]


class EntityAliasResolver:
    """Resolve textual aliases into canonical legal entity IDs."""

    def __init__(self, clusters: list[AliasCluster]) -> None:
        """Initialize a resolver from immutable alias clusters.

        Args:
            clusters: Canonical alias clusters to index.
        """

        self._clusters = {cluster.canonical_id: cluster for cluster in clusters}
        self._alias_lookup: dict[tuple[CanonicalEntityType, str], str] = {}
        self._surface_aliases: list[tuple[str, str, CanonicalEntityType]] = []
        for cluster in clusters:
            for alias in cluster.aliases:
                lookup_key = self._alias_lookup_key(alias, cluster.entity_type)
                if not lookup_key:
                    continue
                self._alias_lookup[(cluster.entity_type, lookup_key)] = cluster.canonical_id
                self._surface_aliases.append((alias, cluster.canonical_id, cluster.entity_type))
        self._surface_aliases.sort(key=lambda item: len(item[0]), reverse=True)

    @classmethod
    def build_from_registry(cls, corpus_registry: CorpusRegistry) -> EntityAliasResolver:
        """Build a resolver from a compiled corpus registry.

        Args:
            corpus_registry: Compiled corpus registry.

        Returns:
            EntityAliasResolver: Built alias resolver.
        """

        clusters = [
            *build_law_aliases(corpus_registry),
            *build_case_aliases(corpus_registry),
            *build_party_aliases(corpus_registry),
            *build_judge_aliases(corpus_registry),
            *build_authority_aliases(corpus_registry),
        ]
        return cls(clusters)

    @staticmethod
    def _alias_lookup_key(surface_form: str, entity_type: CanonicalEntityType) -> str:
        """Build a normalized alias lookup key by entity type.

        Args:
            surface_form: Raw surface form.
            entity_type: Alias category.

        Returns:
            str: Stable lookup key.
        """

        if entity_type is CanonicalEntityType.LAW_TITLE:
            return _law_title_lookup_key(surface_form)
        if entity_type is CanonicalEntityType.CASE_NUMBER:
            return _case_number_key(surface_form)
        if entity_type is CanonicalEntityType.JUDGE:
            return _person_key(surface_form)
        return _organization_key(surface_form)

    def resolve(self, surface_form: str, entity_type: CanonicalEntityType | None = None) -> str | None:
        """Resolve a single surface form into a canonical ID.

        Args:
            surface_form: Raw surface form to resolve.
            entity_type: Optional entity-type constraint.

        Returns:
            str | None: Canonical entity ID if found.
        """

        if entity_type is not None:
            key = self._alias_lookup_key(surface_form, entity_type)
            return self._alias_lookup.get((entity_type, key))
        for candidate_type in CanonicalEntityType:
            resolved = self.resolve(surface_form, candidate_type)
            if resolved:
                return resolved
        return None

    def resolve_all(self, text: str) -> list[tuple[tuple[int, int], str]]:
        """Resolve all known entity mentions inside a text.

        Args:
            text: Raw input text.

        Returns:
            list[tuple[tuple[int, int], str]]: Matched spans and canonical IDs.
        """

        haystack = text or ""
        matches: list[tuple[tuple[int, int], str]] = []
        seen: set[tuple[int, int, str]] = set()
        for alias, canonical_id, _entity_type in self._surface_aliases:
            if len(alias) < 3:
                continue
            pattern = re.compile(rf"(?<![A-Za-z0-9]){re.escape(alias)}(?![A-Za-z0-9])", re.IGNORECASE)
            for match in pattern.finditer(haystack):
                key = (match.start(), match.end(), canonical_id)
                if key in seen:
                    continue
                seen.add(key)
                matches.append(((match.start(), match.end()), canonical_id))
        matches.sort(key=lambda item: (item[0][0], item[0][1], item[1]))
        return matches

    def resolve_query_ids(self, query_text: str) -> list[str]:
        """Resolve all canonical IDs mentioned in a query.

        Args:
            query_text: Query text to analyze.

        Returns:
            list[str]: Sorted canonical IDs.
        """

        return sorted({canonical_id for _span, canonical_id in self.resolve_all(query_text)})

    def resolve_known_values(
        self,
        *,
        law_titles: list[str] | tuple[str, ...] = (),
        case_numbers: list[str] | tuple[str, ...] = (),
        party_names: list[str] | tuple[str, ...] = (),
        authority_names: list[str] | tuple[str, ...] = (),
        judge_names: list[str] | tuple[str, ...] = (),
    ) -> list[str]:
        """Resolve structured known metadata values into canonical IDs.

        Args:
            law_titles: Candidate law-title values.
            case_numbers: Candidate case-number values.
            party_names: Candidate party values.
            authority_names: Candidate authority or court values.
            judge_names: Candidate judge values.

        Returns:
            list[str]: Sorted canonical IDs.
        """

        resolved: set[str] = set()
        for value in law_titles:
            canonical_id = self.resolve(value, CanonicalEntityType.LAW_TITLE)
            if canonical_id:
                resolved.add(canonical_id)
        for value in case_numbers:
            canonical_id = self.resolve(value, CanonicalEntityType.CASE_NUMBER)
            if canonical_id:
                resolved.add(canonical_id)
        for value in party_names:
            canonical_id = self.resolve(value, CanonicalEntityType.PARTY)
            if canonical_id:
                resolved.add(canonical_id)
        for value in authority_names:
            canonical_id = self.resolve(value, CanonicalEntityType.AUTHORITY)
            if canonical_id:
                resolved.add(canonical_id)
        for value in judge_names:
            canonical_id = self.resolve(value, CanonicalEntityType.JUDGE)
            if canonical_id:
                resolved.add(canonical_id)
        return sorted(resolved)

    def add_alias(self, canonical_id: str, surface_form: str) -> None:
        """Add one alias to an existing cluster.

        Args:
            canonical_id: Existing cluster ID.
            surface_form: New alias surface form.

        Raises:
            KeyError: If the canonical ID does not exist.
        """

        cluster = self._clusters[canonical_id]
        aliases = set(cluster.aliases)
        aliases.add(_clean_text(surface_form))
        updated = AliasCluster(
            canonical_id=cluster.canonical_id,
            canonical_form=cluster.canonical_form,
            entity_type=cluster.entity_type,
            aliases=tuple(sorted(aliases)),
            source_doc_ids=cluster.source_doc_ids,
        )
        self._clusters[canonical_id] = updated
        lookup_key = self._alias_lookup_key(surface_form, cluster.entity_type)
        self._alias_lookup[(cluster.entity_type, lookup_key)] = canonical_id
        self._surface_aliases.append((_clean_text(surface_form), canonical_id, cluster.entity_type))
        self._surface_aliases.sort(key=lambda item: len(item[0]), reverse=True)

    def export(self, path: str | Path) -> Path:
        """Persist clusters into a JSON registry file.

        Args:
            path: Output JSON path.

        Returns:
            Path: Resolved output path.
        """

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"clusters": [cluster.to_dict() for cluster in self.iter_clusters()]}
        output_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True), encoding="utf-8")
        return output_path

    @classmethod
    def load(cls, path: str | Path) -> EntityAliasResolver:
        """Load a resolver from a JSON registry file.

        Args:
            path: Path to the alias registry JSON file.

        Returns:
            EntityAliasResolver: Loaded alias resolver.
        """

        payload = _coerce_payload_dict(json.loads(Path(path).read_text(encoding="utf-8")))
        cluster_items = _coerce_object_list(payload.get("clusters", []))
        clusters: list[AliasCluster] = []
        for cluster in cluster_items:
            cluster_payload = _coerce_payload_dict(cluster)
            if cluster_payload:
                clusters.append(AliasCluster.from_dict(cluster_payload))
        return cls(clusters)

    def iter_clusters(self) -> list[AliasCluster]:
        """Return all clusters in stable order.

        Returns:
            list[AliasCluster]: Sorted clusters.
        """

        return [self._clusters[key] for key in sorted(self._clusters)]
