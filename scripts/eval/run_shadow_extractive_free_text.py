# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownMemberType=false
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import cast

import fitz

JsonDict = dict[str, object]

_MONEY_RE = re.compile(r"\$?\s*([0-9]{1,3}(?:,[0-9]{3})+|[0-9]{4,})")
_DATE_RE = re.compile(
    r"\b(\d{1,2}\s+[A-Z][a-z]+\s+\d{4}|[A-Z][a-z]+\s+\d{1,2},\s+\d{4}|the date specified in the Enactment Notice(?: in respect of this Law)?)\b"
)


def _load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _coerce_dict_list(value: object) -> list[JsonDict]:
    if not isinstance(value, list):
        return []
    return [cast("JsonDict", item) for item in value if isinstance(item, dict)]


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip()).casefold()


def _coerce_int(value: object, *, default: int = 0) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str) and value.strip():
        try:
            return int(value)
        except ValueError:
            return default
    return default


def _load_truth_records(path: Path) -> dict[str, JsonDict]:
    payload = _load_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return {
        str(record.get("question_id") or "").strip(): record
        for record in _coerce_dict_list(payload.get("records"))
        if str(record.get("question_id") or "").strip()
    }


def _load_model_pack(path: Path) -> list[JsonDict]:
    payload = _load_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return _coerce_dict_list(payload.get("cases"))


def _support_snippets(record: JsonDict) -> list[tuple[int, str]]:
    snippets: list[tuple[int, str]] = []
    for preview in _coerce_dict_list(record.get("support_page_previews")):
        page = _coerce_int(preview.get("page"), default=0)
        snippet = str(preview.get("snippet") or "").strip()
        if snippet:
            snippets.append((page, snippet))
    return snippets


def _load_pdf_page_snippets(*, used_page_ids: list[str], document_root: Path | None) -> list[tuple[int, str]]:
    if document_root is None:
        return []

    snippets: list[tuple[int, str]] = []
    seen_pages: set[tuple[str, int]] = set()
    for page_id in used_page_ids:
        raw = str(page_id or "").strip()
        doc_id, _, page_raw = raw.rpartition("_")
        if not doc_id or not page_raw.isdigit():
            continue
        page_num = int(page_raw)
        if page_num <= 0:
            continue
        key = (doc_id, page_num)
        if key in seen_pages:
            continue
        seen_pages.add(key)
        pdf_path = document_root / f"{doc_id}.pdf"
        if not pdf_path.exists():
            continue
        try:
            doc = fitz.open(str(pdf_path))
        except Exception:
            continue
        try:
            if page_num > doc.page_count:
                continue
            text = doc.load_page(page_num - 1).get_text("text").strip()
        finally:
            doc.close()
        if text:
            snippets.append((page_num, text))
    return snippets


def _family(question: str) -> str:
    q = _normalize(question)
    if "who made this law" in q or "made this law" in q:
        return "who_made"
    if "what kind of liability" in q and "partner" in q:
        return "liability"
    if "maximum fine" in q:
        return "maximum_fine"
    if "appointing and dismissing the registrar" in q or ("appoint" in q and "dismiss" in q and "registrar" in q):
        return "registrar_authority"
    if "on what date" in q and "enacted" in q:
        return "enactment_date"
    if "what must it provide" in q and "language other than english" in q:
        return "translation_requirement"
    if "what was the outcome" in q and "case" in q:
        return "case_outcome"
    return ""


def _extract_answer(family: str, snippets: list[tuple[int, str]]) -> tuple[str, list[int]]:
    joined = " ".join(snippet for _, snippet in snippets)
    lowered = joined.casefold()
    if family == "who_made":
        match = re.search(r"made by (?:the )?([A-Z][A-Za-z ]+)", joined)
        if match:
            entity = match.group(1).strip().rstrip(".")
            return f"This Law was made by {entity}.", [page for page, _ in snippets[:1]]
    if family == "liability" and "jointly and severally liable" in lowered:
        return "Partners are jointly and severally liable.", [page for page, _ in snippets[:1]]
    if family == "maximum_fine":
        money_match = _MONEY_RE.search(joined)
        if money_match:
            return f"The maximum fine is ${money_match.group(1)}.", [page for page, _ in snippets[:1]]
    if family == "registrar_authority":
        match = re.search(
            r"(Board of Directors of the DIFCA)[^.]{0,140}(appoint|appointing)[^.]{0,140}(dismiss|dismissing)",
            joined,
            flags=re.IGNORECASE,
        )
        if match:
            entity = match.group(1)
            return f"{entity} appoints and may dismiss the Registrar.", [page for page, _ in snippets[:1]]
    if family == "enactment_date":
        match = _DATE_RE.search(joined)
        if match:
            value = match.group(1).strip()
            if value.casefold().startswith("the date specified in the enactment notice"):
                return "The date of enactment is the date specified in the Enactment Notice in respect of this Law.", [
                    page for page, _ in snippets[:1]
                ]
            return f"The date of enactment is {value}.", [page for page, _ in snippets[:1]]
    if family == "translation_requirement" and "english translation" in lowered:
        return "It must provide an English translation to the Relevant Authority.", [page for page, _ in snippets[:1]]
    if family == "case_outcome":
        match = re.search(r"\b(granted|refused|declined|dismissed|allowed)\b", joined, flags=re.IGNORECASE)
        if match:
            verb = match.group(1).lower()
            return f"The application was {verb}.", [page for page, _ in snippets[:1]]
    return "", []


def run_shadow_eval(*, model_route_pack: Path, truth_scaffold: Path, document_root: Path | None) -> JsonDict:
    model_cases = _load_model_pack(model_route_pack)
    truth_by_qid = _load_truth_records(truth_scaffold)

    results: list[JsonDict] = []
    attempted = 0
    wins = 0

    for case in model_cases:
        qid = str(case.get("question_id") or "").strip()
        family = _family(str(case.get("question") or ""))
        if not family:
            continue
        truth = truth_by_qid.get(qid)
        if truth is None:
            continue
        used_page_ids = [str(page_id) for page_id in cast("list[object]", case.get("used_page_ids") or []) if str(page_id).strip()]
        snippets = _load_pdf_page_snippets(used_page_ids=used_page_ids, document_root=document_root) or _support_snippets(truth)
        if not snippets:
            continue
        predicted_answer, predicted_pages = _extract_answer(family, snippets)
        baseline_answer = str(case.get("answer_text") or "").strip()
        baseline_pages = used_page_ids

        attempted += 1
        semantic_match = bool(predicted_answer) and (
            _normalize(predicted_answer) in _normalize(baseline_answer)
            or _normalize(baseline_answer) in _normalize(predicted_answer)
            or any(token in _normalize(baseline_answer) for token in _normalize(predicted_answer).split() if len(token) > 4)
        )
        page_win = bool(predicted_pages) and (not baseline_pages or len(predicted_pages) <= len(baseline_pages))
        concise_win = bool(predicted_answer) and len(predicted_answer) < len(baseline_answer)
        corrective_win = bool(predicted_answer) and ("do not specify" in baseline_answer.lower() or baseline_answer.lower() in {"", "null"})
        win = bool(predicted_answer) and page_win and (corrective_win or (semantic_match and concise_win))
        if win:
            wins += 1
        results.append(
            {
                "question_id": qid,
                "family": family,
                "question": str(case.get("question") or "").strip(),
                "baseline_answer": baseline_answer,
                "predicted_answer": predicted_answer,
                "baseline_used_page_count": len(baseline_pages),
                "predicted_used_page_count": len(predicted_pages),
                "evidence_source": "pdf_pages" if document_root is not None and snippets and len(snippets[0][1]) > 280 else "support_previews",
                "semantic_match": semantic_match,
                "concise_win": concise_win,
                "corrective_win": corrective_win,
                "win": win,
            }
        )

    return {
        "summary": {
            "attempted_case_count": attempted,
            "win_case_count": wins,
            "verdict": "win" if wins >= 4 else "fail_closed",
        },
        "cases": results,
    }


def _render_markdown(payload: JsonDict) -> str:
    summary = cast("JsonDict", payload["summary"])
    lines = [
        "# Shadow Extractive Free-Text Eval",
        "",
        f"- attempted_case_count: `{summary['attempted_case_count']}`",
        f"- win_case_count: `{summary['win_case_count']}`",
        f"- verdict: `{summary['verdict']}`",
        "",
        "## Cases",
        "",
    ]
    for case in cast("list[JsonDict]", payload["cases"]):
        lines.extend(
            [
                f"### {case['question_id']}",
                f"- family: `{case['family']}`",
                f"- win: `{case['win']}`",
                f"- baseline_used_page_count: `{case['baseline_used_page_count']}`",
                f"- predicted_used_page_count: `{case['predicted_used_page_count']}`",
                f"- question: {case['question']}",
                f"- baseline_answer: `{case['baseline_answer']}`",
                f"- predicted_answer: `{case['predicted_answer']}`",
                "",
            ]
        )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the deterministic extractive free-text shadow eval.")
    parser.add_argument("--model-route-pack", type=Path, required=True)
    parser.add_argument("--truth-scaffold-json", type=Path, required=True)
    parser.add_argument("--document-root", type=Path, default=None)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = run_shadow_eval(
        model_route_pack=args.model_route_pack,
        truth_scaffold=args.truth_scaffold_json,
        document_root=args.document_root,
    )
    args.out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.out_md.write_text(_render_markdown(payload) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
