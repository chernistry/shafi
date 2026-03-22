#!/usr/bin/env python3
"""Polish free_text answers: rewrite non-evidence-first answers to start with legal provision.
Run OFFLINE on submission JSON — no pipeline re-run needed.

Usage: uv run python scripts/polish_free_text.py data/private_submission_V17_HYBRID.json data/private_submission_V17_POLISHED.json
"""
import json, sys, re, httpx, time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
QUESTIONS = REPO / "dataset" / "private" / "questions.json"

REWRITE_PROMPT = """Rewrite this legal QA answer. Rules:
1. START with the governing legal provision: "Under Article X..." or "Rule Y provides..." or "Section Z of..."
2. Keep under 250 characters. Every sentence ends with period.
3. Preserve ALL facts from the original. Add NOTHING new.
4. If original mentions a case (CFI/SCT/CA/ARB), start with the case reference.

Original: {answer}
Question: {question}

Rewritten (under 250 chars, starts with provision):"""

EVIDENCE_PREFIXES = ["under ", "article ", "rule ", "regulation ", "section ", "pursuant ", "per ", "paragraph "]
CASE_PREFIXES = ["CFI", "SCT ", "CA ", "ARB", "TCD", "ENF"]


def needs_rewrite(answer: str) -> bool:
    """True if answer doesn't start evidence-first."""
    if not answer or "no information" in answer.lower():
        return False
    low = answer.lower()
    if any(low.startswith(p) for p in EVIDENCE_PREFIXES):
        return False
    if any(answer.upper().startswith(p) for p in CASE_PREFIXES):
        return False
    return True


def rewrite_answer(question: str, answer: str) -> str | None:
    """Call gpt-4.1-mini to rewrite answer with evidence-first opening."""
    try:
        import openai
        client = openai.OpenAI()
        resp = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": REWRITE_PROMPT.format(answer=answer, question=question)}],
            max_tokens=150,
            temperature=0.0,
        )
        rewritten = resp.choices[0].message.content.strip()
        # Validate: must start evidence-first and preserve key terms
        if not rewritten or len(rewritten) > 280:
            return None
        low = rewritten.lower()
        if not any(low.startswith(p) for p in EVIDENCE_PREFIXES + [p.lower() for p in CASE_PREFIXES]):
            return None
        # Check key term preservation (at least 50% of significant words match)
        orig_words = set(w.lower() for w in re.findall(r'\b\w{4,}\b', answer))
        new_words = set(w.lower() for w in re.findall(r'\b\w{4,}\b', rewritten))
        if orig_words and len(orig_words & new_words) / len(orig_words) < 0.3:
            return None  # Too much content changed
        if not rewritten.endswith("."):
            rewritten = rewritten.rstrip(",:;") + "."
        return rewritten
    except Exception as e:
        print(f"  Error: {e}", file=sys.stderr)
        return None


def main():
    if len(sys.argv) < 3:
        print("Usage: uv run python scripts/polish_free_text.py INPUT OUTPUT")
        sys.exit(1)
    
    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    
    d = json.loads(input_path.read_text())
    qs = {q["id"]: q for q in json.loads(QUESTIONS.read_text())}
    
    rewritten_count = 0
    skipped = 0
    failed = 0
    
    for ans in d["answers"]:
        q = qs.get(ans["question_id"], {})
        if q.get("answer_type") != "free_text":
            continue
        if not isinstance(ans["answer"], str):
            continue
        if not needs_rewrite(ans["answer"]):
            skipped += 1
            continue
        
        rewritten = rewrite_answer(q.get("question", ""), ans["answer"])
        if rewritten:
            ans["answer"] = rewritten
            rewritten_count += 1
        else:
            failed += 1
    
    output_path.write_text(json.dumps(d))
    print(f"Polished: {rewritten_count} rewritten, {skipped} already good, {failed} failed (kept original)")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
