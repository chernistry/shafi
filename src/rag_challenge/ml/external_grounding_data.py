# pyright: reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownLambdaType=false, reportMissingTypeStubs=false
"""Normalization helpers for external legal-grounding datasets."""

from __future__ import annotations

import hashlib
import io
import json
import re
import urllib.request
import zipfile
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pandas as pd
import yaml
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence
    from pathlib import Path

_CUAD_ROLE_RE = re.compile(r'related to "(?P<label>.+?)" that should be reviewed', re.IGNORECASE)
_NON_WORD_RE = re.compile(r"[^a-z0-9]+")
_CUAD_URL_RE = re.compile(r'^_URL\s*=\s*"(?P<url>[^"]+)"', re.MULTILINE)
_DEFAULT_CUAD_URL = "https://github.com/TheAtticusProject/cuad/raw/main/data.zip"


class NormalizedExternalRow(BaseModel):
    """One normalized external-supervision example.

    Args:
        source_dataset: Source dataset name.
        sample_id: Stable sample identifier.
        text: Evidence text or clause text.
        question: Question or hypothesis text.
        label_type: High-level supervision family.
        role_label: Optional role/clause label.
        scope_label: Optional scope/routing label.
        support_label: Optional support/entailment label.
        metadata_json: Deterministic JSON metadata string.
    """

    source_dataset: str
    sample_id: str
    text: str
    question: str
    label_type: str
    role_label: str
    scope_label: str
    support_label: str
    metadata_json: str


class NormalizedExternalManifest(BaseModel):
    """Manifest describing one normalized external-data export.

    Args:
        generated_at: ISO timestamp.
        output_dir: Output directory.
        max_rows_per_dataset: Optional cap used during export.
        total_rows: Total row count.
        row_counts_by_dataset: Counts grouped by source dataset.
        row_counts_by_label_type: Counts grouped by label type.
        schema_fields: Ordered output field names.
        source_paths: Input source roots.
    """

    generated_at: str
    output_dir: str
    max_rows_per_dataset: int
    total_rows: int
    row_counts_by_dataset: dict[str, int] = Field(default_factory=dict)
    row_counts_by_label_type: dict[str, int] = Field(default_factory=dict)
    schema_fields: list[str] = Field(default_factory=list)
    source_paths: dict[str, str] = Field(default_factory=dict)


def export_normalized_external_grounding_data(
    *,
    obliqa_root: Path,
    cuad_root: Path,
    contractnli_root: Path,
    ledgar_root: Path,
    output_dir: Path,
    max_rows_per_dataset: int = 0,
) -> NormalizedExternalManifest:
    """Normalize external legal datasets into one common schema bundle.

    Args:
        obliqa_root: Root of the ObliQA raw snapshot.
        cuad_root: Root of the CUAD raw snapshot.
        contractnli_root: Root of the ContractNLI raw snapshot.
        ledgar_root: Root of the LEDGAR raw snapshot.
        output_dir: Target normalized output directory.
        max_rows_per_dataset: Optional per-dataset cap. `0` means no cap.

    Returns:
        Manifest describing the normalized bundle.

    Raises:
        FileNotFoundError: If a required input root is missing.
        ValueError: If a dataset snapshot cannot be parsed.
    """
    rows = [
        *load_obliqa_rows(obliqa_root, max_rows=max_rows_per_dataset),
        *load_cuad_rows(cuad_root, max_rows=max_rows_per_dataset),
        *load_contractnli_rows(contractnli_root, max_rows=max_rows_per_dataset),
        *load_ledgar_rows(ledgar_root, max_rows=max_rows_per_dataset),
    ]
    ordered_rows = sorted(rows, key=lambda row: (row.source_dataset, row.sample_id))
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "normalized_rows.jsonl"
    rows_path.write_text(
        "".join(json.dumps(row.model_dump(mode="json"), ensure_ascii=True) + "\n" for row in ordered_rows),
        encoding="utf-8",
    )

    manifest = NormalizedExternalManifest(
        generated_at=datetime.now(UTC).isoformat(),
        output_dir=str(output_dir),
        max_rows_per_dataset=max_rows_per_dataset,
        total_rows=len(ordered_rows),
        row_counts_by_dataset=_count_by_key(ordered_rows, key="source_dataset"),
        row_counts_by_label_type=_count_by_key(ordered_rows, key="label_type"),
        schema_fields=list(NormalizedExternalRow.model_fields),
        source_paths={
            "obliqa_root": str(obliqa_root),
            "cuad_root": str(cuad_root),
            "contractnli_root": str(contractnli_root),
            "ledgar_root": str(ledgar_root),
        },
    )
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest.model_dump(mode="json"), ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def load_obliqa_rows(root: Path, *, max_rows: int = 0) -> list[NormalizedExternalRow]:
    """Load ObliQA rows from local JSON snapshots.

    Args:
        root: ObliQA snapshot root.
        max_rows: Optional per-dataset cap.

    Returns:
        Normalized rows for ObliQA.

    Raises:
        FileNotFoundError: If no ObliQA JSON files are present.
    """
    obliqa_dir = root / "ObliQA"
    files = [obliqa_dir / "ObliQA_train.json", obliqa_dir / "ObliQA_dev.json", obliqa_dir / "ObliQA_test.json"]
    if not any(path.exists() for path in files):
        raise FileNotFoundError(f"Missing ObliQA JSON files under {obliqa_dir}")

    rows: list[NormalizedExternalRow] = []
    for split, path in (("train", files[0]), ("dev", files[1]), ("test", files[2])):
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"Unexpected ObliQA payload in {path}")
        for record in payload:
            if not isinstance(record, dict):
                continue
            rows.append(normalize_obliqa_record(record, split=split))
            if max_rows > 0 and len(rows) >= max_rows:
                return rows
    return rows


def load_cuad_rows(root: Path, *, max_rows: int = 0) -> list[NormalizedExternalRow]:
    """Load CUAD rows from a local materialization or raw upstream zip.

    Args:
        root: CUAD snapshot root.
        max_rows: Optional per-dataset cap.

    Returns:
        Normalized CUAD rows.
    """
    materialized_dir = root / "materialized"
    if materialized_dir.exists():
        rows = list(_iter_cuad_materialized_rows(materialized_dir))
    else:
        rows = list(_iter_cuad_hub_rows(root=root, max_rows=max_rows))
    if max_rows > 0:
        rows = rows[:max_rows]
    return rows


def load_contractnli_rows(root: Path, *, max_rows: int = 0) -> list[NormalizedExternalRow]:
    """Load ContractNLI rows from local parquet files.

    Args:
        root: ContractNLI snapshot root.
        max_rows: Optional per-dataset cap.

    Returns:
        Normalized ContractNLI rows.
    """
    label_names = _load_label_names_from_readme(root / "README.md")
    rows: list[NormalizedExternalRow] = []
    for split in ("train", "validation", "dev", "test"):
        data_files = sorted((root / "data").glob(f"{split}-*.parquet"))
        if not data_files:
            continue
        for index, record in enumerate(_iter_parquet_records(root / "data", split=split), start=1):
            rows.append(normalize_contractnli_record(record, split=split, index=index, label_names=label_names))
            if max_rows > 0 and len(rows) >= max_rows:
                return rows
    if not rows:
        raise FileNotFoundError(f"Missing ContractNLI parquet shards under {root / 'data'}")
    return rows


def load_ledgar_rows(root: Path, *, max_rows: int = 0) -> list[NormalizedExternalRow]:
    """Load LEDGAR rows from local parquet files.

    Args:
        root: LEDGAR snapshot root.
        max_rows: Optional per-dataset cap.

    Returns:
        Normalized LEDGAR rows.
    """
    label_names = _load_label_names_from_readme(root / "README.md")
    rows: list[NormalizedExternalRow] = []
    for split in ("train", "validation", "test"):
        data_files = sorted((root / "data").glob(f"{split}-*.parquet"))
        if not data_files:
            continue
        for index, record in enumerate(_iter_parquet_records(root / "data", split=split), start=1):
            rows.append(normalize_ledgar_record(record, split=split, index=index, label_names=label_names))
            if max_rows > 0 and len(rows) >= max_rows:
                return rows
    if not rows:
        raise FileNotFoundError(f"Missing LEDGAR parquet shards under {root / 'data'}")
    return rows


def normalize_obliqa_record(record: dict[str, object], *, split: str) -> NormalizedExternalRow:
    """Normalize one ObliQA record.

    Args:
        record: Raw ObliQA record.
        split: Source split name.

    Returns:
        NormalizedExternalRow for ObliQA.
    """
    passages = [passage for passage in _coerce_list(record.get("Passages")) if isinstance(passage, dict)]
    question = _coerce_str(record.get("Question"))
    sample_id = _coerce_str(record.get("QuestionID")) or _stable_id("obliqa", split, question)
    passage_texts = [
        f"[{_coerce_str(passage.get('PassageID'))}] {_coerce_str(passage.get('Passage'))}".strip()
        for passage in passages
    ]
    metadata = {
        "split": split,
        "group": _coerce_str(record.get("Group")),
        "document_ids": [_coerce_scalar(passage.get("DocumentID")) for passage in passages],
        "passage_ids": [_coerce_str(passage.get("PassageID")) for passage in passages],
        "passage_count": len(passages),
    }
    return NormalizedExternalRow(
        source_dataset="obliqa",
        sample_id=sample_id,
        text="\n\n".join(text for text in passage_texts if text),
        question=question,
        label_type="support_scope",
        role_label="",
        scope_label="single_field_single_doc" if len(passages) <= 1 else "multi_passage_support",
        support_label="supported",
        metadata_json=_metadata_json(metadata),
    )


def normalize_cuad_record(record: dict[str, object], *, split: str) -> NormalizedExternalRow:
    """Normalize one CUAD record.

    Args:
        record: Raw CUAD record.
        split: Source split name.

    Returns:
        NormalizedExternalRow for CUAD.
    """
    question = _coerce_str(record.get("question"))
    answers = record.get("answers")
    answer_texts = []
    if isinstance(answers, dict):
        answer_texts = [_coerce_str(value) for value in _coerce_list(answers.get("text")) if _coerce_str(value)]
    label = _extract_cuad_role_label(question)
    metadata = {
        "split": split,
        "title": _coerce_str(record.get("title")),
        "answer_count": len(answer_texts),
        "answer_texts": answer_texts[:5],
    }
    return NormalizedExternalRow(
        source_dataset="cuad",
        sample_id=_coerce_str(record.get("id")) or _stable_id("cuad", split, question),
        text=_coerce_str(record.get("context")),
        question=question,
        label_type="role_label",
        role_label=label,
        scope_label="single_field_single_doc",
        support_label="supported" if answer_texts else "unanswerable",
        metadata_json=_metadata_json(metadata),
    )


def normalize_contractnli_record(
    record: dict[str, object],
    *,
    split: str,
    index: int,
    label_names: Sequence[str],
) -> NormalizedExternalRow:
    """Normalize one ContractNLI row.

    Args:
        record: Raw ContractNLI row.
        split: Source split name.
        index: 1-based row index within the split.
        label_names: Optional label-name lookup from the card.

    Returns:
        NormalizedExternalRow for ContractNLI.
    """
    label_value = record.get("gold_label")
    if not _coerce_str(label_value):
        label_value = _label_name_from_value(record.get("label"), label_names=label_names)
    support_label = _slugify(_coerce_str(label_value) or "unknown")
    metadata = {
        "split": split,
        "raw_label": _coerce_scalar(record.get("label")),
        "gold_label": _coerce_str(record.get("gold_label")),
    }
    sentence1 = _coerce_str(record.get("sentence1"))
    sentence2 = _coerce_str(record.get("sentence2"))
    return NormalizedExternalRow(
        source_dataset="contractnli",
        sample_id=_stable_id("contractnli", split, str(index), sentence1, sentence2),
        text=sentence1,
        question=sentence2,
        label_type="support_label",
        role_label="",
        scope_label="pair_entailment",
        support_label=support_label,
        metadata_json=_metadata_json(metadata),
    )


def normalize_ledgar_record(
    record: dict[str, object],
    *,
    split: str,
    index: int,
    label_names: Sequence[str],
) -> NormalizedExternalRow:
    """Normalize one LEDGAR row.

    Args:
        record: Raw LEDGAR row.
        split: Source split name.
        index: 1-based row index within the split.
        label_names: Label-name lookup from the card.

    Returns:
        NormalizedExternalRow for LEDGAR.
    """
    raw_label = record.get("label")
    label_name = _label_name_from_value(raw_label, label_names=label_names)
    metadata = {
        "split": split,
        "raw_label": _coerce_scalar(raw_label),
        "label_name": label_name,
    }
    text = _coerce_str(record.get("text"))
    return NormalizedExternalRow(
        source_dataset="ledgar",
        sample_id=_stable_id("ledgar", split, str(index), text),
        text=text,
        question="Which clause role best matches this provision?",
        label_type="role_label",
        role_label=_slugify(label_name or "unknown"),
        scope_label="single_field_single_doc",
        support_label="role_supervision",
        metadata_json=_metadata_json(metadata),
    )


def _iter_cuad_materialized_rows(materialized_dir: Path) -> Iterator[NormalizedExternalRow]:
    """Iterate normalized CUAD rows from local materialized JSONL files.

    Args:
        materialized_dir: Directory containing split JSONL files.

    Yields:
        Normalized CUAD rows.
    """
    for split_path in sorted(materialized_dir.glob("*.jsonl")):
        split = split_path.stem
        for line in split_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                yield normalize_cuad_record({str(key): value for key, value in payload.items()}, split=split)


def _iter_cuad_hub_rows(*, root: Path, max_rows: int) -> Iterator[NormalizedExternalRow]:
    """Iterate normalized CUAD rows from the upstream raw zip.

    Args:
        root: CUAD snapshot root used to resolve the upstream URL.
        max_rows: Optional per-dataset cap.

    Yields:
        Normalized CUAD rows.
    """
    emitted = 0
    zip_bytes = _download_cuad_zip(root)
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
        for split, member_name in (("train", "train_separate_questions.json"), ("test", "test.json")):
            with archive.open(member_name) as source:
                payload = json.load(source)
            data = payload.get("data", []) if isinstance(payload, dict) else []
            for record in _iter_cuad_json_examples(data):
                yield normalize_cuad_record(record, split=split)
                emitted += 1
                if max_rows > 0 and emitted >= max_rows:
                    return


def _iter_parquet_records(data_dir: Path, *, split: str) -> Iterator[dict[str, object]]:
    """Iterate rows from local parquet shards via pandas.

    Args:
        data_dir: Directory containing parquet shards.
        split: Split prefix, for example `train` or `validation`.

    Yields:
        Parquet rows as plain dictionaries.

    Raises:
        FileNotFoundError: If the requested split parquet shard is missing.
    """
    data_files = sorted(data_dir.glob(f"{split}-*.parquet"))
    if not data_files:
        raise FileNotFoundError(f"Missing parquet shards for split={split} under {data_dir}")

    for path in data_files:
        frame = pd.read_parquet(path)
        for record in frame.to_dict(orient="records"):
            yield _row_to_dict(record)


def _load_label_names_from_readme(path: Path) -> list[str]:
    """Extract class-label names from a dataset README front matter block.

    Args:
        path: README path.

    Returns:
        Ordered label names, or an empty list if none are declared.
    """
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    front_matter = _parse_front_matter(text)
    dataset_info = front_matter.get("dataset_info")
    if not isinstance(dataset_info, dict):
        return []
    features = dataset_info.get("features")
    if not isinstance(features, list):
        return []
    for feature in features:
        if not isinstance(feature, dict) or feature.get("name") != "label":
            continue
        dtype = feature.get("dtype")
        if not isinstance(dtype, dict):
            continue
        class_label = dtype.get("class_label")
        if not isinstance(class_label, dict):
            continue
        names = class_label.get("names")
        if isinstance(names, dict):
            return [
                _coerce_str(value)
                for _, value in sorted(
                    names.items(),
                    key=lambda item: int(item[0]) if str(item[0]).isdigit() else str(item[0]),
                )
            ]
    return []


def _parse_front_matter(text: str) -> dict[str, object]:
    """Parse a Hugging Face README front matter block.

    Args:
        text: README text.

    Returns:
        Parsed YAML mapping, or an empty dict if unavailable.
    """
    if not text.startswith("---\n"):
        return {}
    marker = "\n---"
    end_index = text.find(marker, 4)
    if end_index == -1:
        return {}
    payload = yaml.safe_load(text[4:end_index])
    return payload if isinstance(payload, dict) else {}


def _download_cuad_zip(root: Path) -> bytes:
    """Download the upstream CUAD zip referenced by the snapshot script.

    Args:
        root: CUAD snapshot root containing `cuad-qa.py`.

    Returns:
        Raw zip bytes for the CUAD data archive.
    """
    script_path = root / "cuad-qa.py"
    dataset_url = _DEFAULT_CUAD_URL
    if script_path.exists():
        script_text = script_path.read_text(encoding="utf-8")
        match = _CUAD_URL_RE.search(script_text)
        if match:
            dataset_url = match.group("url")
    with urllib.request.urlopen(dataset_url, timeout=120) as response:
        return response.read()


def _extract_cuad_role_label(question: str) -> str:
    """Extract and normalize the CUAD clause label from a question string.

    Args:
        question: CUAD question text.

    Returns:
        Slugified clause-role label.
    """
    match = _CUAD_ROLE_RE.search(question)
    if match:
        return _slugify(match.group("label"))
    if "Details:" in question:
        _, detail = question.split("Details:", 1)
        return _slugify(detail.strip())
    return "unknown"


def _label_name_from_value(value: object, *, label_names: Sequence[str]) -> str:
    """Resolve a label name from an integer/string raw value.

    Args:
        value: Raw label value.
        label_names: Optional ordered label names.

    Returns:
        Human-readable label name when available.
    """
    if isinstance(value, int) and 0 <= value < len(label_names):
        return label_names[value]
    value_str = _coerce_str(value)
    if value_str.isdigit():
        index = int(value_str)
        if 0 <= index < len(label_names):
            return label_names[index]
    return value_str


def _count_by_key(rows: Sequence[NormalizedExternalRow], *, key: str) -> dict[str, int]:
    """Count normalized rows by one model field.

    Args:
        rows: Normalized rows.
        key: Model field name to group by.

    Returns:
        Count mapping.
    """
    counts: dict[str, int] = {}
    for row in rows:
        value = getattr(row, key)
        counts[value] = counts.get(value, 0) + 1
    return counts


def _metadata_json(payload: Mapping[str, object]) -> str:
    """Serialize metadata in a stable JSON form.

    Args:
        payload: Metadata payload.

    Returns:
        Stable JSON string.
    """
    return json.dumps(payload, ensure_ascii=True, sort_keys=True)


def _stable_id(*parts: str) -> str:
    """Build a deterministic sample ID from stable text parts.

    Args:
        *parts: String parts to hash.

    Returns:
        Stable short sample identifier.
    """
    digest = hashlib.sha256("||".join(parts).encode("utf-8")).hexdigest()
    return digest[:16]


def _slugify(value: str) -> str:
    """Normalize a free-text label into a snake-style slug.

    Args:
        value: Raw label text.

    Returns:
        Slugified label.
    """
    lowered = value.casefold()
    slug = _NON_WORD_RE.sub("_", lowered).strip("_")
    return slug or "unknown"


def _coerce_list(value: object) -> list[object]:
    """Coerce a value into a list when possible.

    Args:
        value: Candidate list-like value.

    Returns:
        Plain list or an empty list.
    """
    return list(value) if isinstance(value, list | tuple) else []


def _coerce_str(value: object) -> str:
    """Coerce an arbitrary value into a trimmed string.

    Args:
        value: Candidate value.

    Returns:
        Trimmed string or an empty string.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _coerce_scalar(value: object) -> str | int | float | bool | None:
    """Coerce a metadata scalar into a JSON-friendly primitive.

    Args:
        value: Candidate value.

    Returns:
        JSON-friendly primitive or its string representation.
    """
    if value is None or isinstance(value, str | int | float | bool):
        return value
    return _coerce_str(value)


def _row_to_dict(row: object) -> dict[str, object]:
    """Convert a dataset-library row object into a plain dictionary.

    Args:
        row: Dataset-library row object.

    Returns:
        Plain dictionary with string keys.

    Raises:
        TypeError: If the row is not mapping-like.
    """
    if isinstance(row, dict):
        return {str(key): value for key, value in row.items()}
    raise TypeError(f"Unsupported dataset row type: {type(row)!r}")


def _iter_cuad_json_examples(data: object) -> Iterator[dict[str, object]]:
    """Iterate flattened CUAD QA examples from raw JSON payload data.

    Args:
        data: Raw `data` array from the CUAD source JSON.

    Yields:
        Flattened QA examples matching the Hugging Face dataset script output.
    """
    if not isinstance(data, list):
        return
    for example in data:
        if not isinstance(example, dict):
            continue
        title = _coerce_str(example.get("title"))
        for paragraph in _coerce_list(example.get("paragraphs")):
            if not isinstance(paragraph, dict):
                continue
            context = _coerce_str(paragraph.get("context"))
            for qa in _coerce_list(paragraph.get("qas")):
                if not isinstance(qa, dict):
                    continue
                answers = [
                    answer
                    for answer in _coerce_list(qa.get("answers"))
                    if isinstance(answer, dict)
                ]
                yield {
                    "title": title,
                    "context": context,
                    "question": _coerce_str(qa.get("question")),
                    "id": _coerce_str(qa.get("id")),
                    "answers": {
                        "answer_start": [
                            int(answer.get("answer_start", 0))
                            for answer in answers
                            if _coerce_str(answer.get("answer_start"))
                        ],
                        "text": [_coerce_str(answer.get("text")) for answer in answers if _coerce_str(answer.get("text"))],
                    },
                }
