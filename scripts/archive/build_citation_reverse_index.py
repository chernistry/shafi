import json
import hashlib
from pathlib import Path
import fitz
import re

def _normalize(text: str) -> str:
    # Normalize cited text for fuzzy lookup: lowercase, collapse whitespace,
    # strip punctuation that varies across sources (e.g. "NO." vs "NO")
    return re.sub(r"[\s\-\.]+", " ", text.strip().lower())

def get_doc_hash(path: Path) -> str:
    if len(path.stem) == 64:
        return path.stem[:16]
    return hashlib.sha256(str(path.resolve()).encode("utf-8")).hexdigest()[:16]

def extract_title(pdf_path: Path) -> str:
    try:
        with fitz.open(str(pdf_path)) as pdf:
            if len(pdf) == 0:
                return ""
            first_page = pdf[0]
            text = first_page.get_text("text").strip()
            # Heuristic: title is often in the first few lines
            lines = [l.strip() for l in text.split("\n") if l.strip()]
            if not lines:
                return ""
            
            # If it's a decision, the title might be "Claim No: ..." or "Judgment of ..."
            # But we want the case name like "Omid v Orah"
            # Often Case Name is above Claim No
            case_name = ""
            for i, line in enumerate(lines[:10]):
                if "BETWEEN" in line:
                    # Party A (Claimant) and Party B (Defendant)
                    # This is complex. Let's just use the first few lines for now.
                    break
            
            # Simple heuristic for laws/regulations:
            # "DIFC LAW NO. X OF 20XX"
            for line in lines[:5]:
                if re.search(r"DIFC LAW NO", line, re.I):
                    return line
            
            return " ".join(lines[:3]) # Fallback
    except Exception:
        return ""

def build_citation_indexes():
    enrichment_dir = Path("data/enrichments/private")
    warmup_enrichment_dir = Path("data/enrichments")
    doc_dir = Path("data/private")
    warmup_doc_dir = Path("dataset/dataset_documents")
    
    # 1. Load known titles from dataset_document_index.json
    title_to_hash = {}
    if Path("dataset_document_index.json").exists():
        with open("dataset_document_index.json", "r") as f:
            known_index = json.load(f)
            for h64, meta in known_index.items():
                title = meta.get("title", "")
                if title:
                    h16 = h64[:16]
                    title_to_hash[title] = h16
    
    # 2. Extract titles from private docs
    pdf_files = list(doc_dir.rglob("*.pdf"))
    print(f"Extracting titles from {len(pdf_files)} private PDFs...")
    for pdf_path in pdf_files:
        h16 = get_doc_hash(pdf_path)
        # Check if already in title_to_hash
        if h16 in title_to_hash.values():
            continue
        title = extract_title(pdf_path)
        if title:
            title_to_hash[title] = h16

    # 3. Build reverse index: title -> [chunk_ids]
    # We need to know which chunk_ids (pages) belong to which doc_hash
    # We can get this from the enrichment files filenames
    hash_to_chunks = {}
    
    all_enrichment_files = list(enrichment_dir.glob("*.json")) + list(warmup_enrichment_dir.glob("*.json"))
    for f in all_enrichment_files:
        if f.name in ("reverse_index.json", "summary_index.json", "forward_index.json"):
            continue
        stem = f.stem
        parts = stem.split("_")
        if len(parts) == 2:
            doc_hash, page_num = parts
            if doc_hash not in hash_to_chunks:
                hash_to_chunks[doc_hash] = []
            chunk_id = f"{doc_hash}_{page_num}"
            if chunk_id not in hash_to_chunks[doc_hash]:
                hash_to_chunks[doc_hash].append(chunk_id)

    reverse_index = {}
    for title, h16 in title_to_hash.items():
        if h16 in hash_to_chunks:
            chunks = hash_to_chunks[h16]
            reverse_index[title] = chunks
            # Also add normalized version? The expander normalizes anyway.
            # But let's add common variations if needed.
    
    # Also include the citing -> cited mapping?
    # No, AMIR specifically said cited_text -> destination_chunks.

    # 4. Save reverse_index.json
    output_path = Path("data/enrichments/reverse_index.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(reverse_index, f, indent=2, ensure_ascii=False)
    
    # 5. Save forward_index.json (title -> [doc_hash])
    forward_index = {title: [h16] for title, h16 in title_to_hash.items()}
    forward_path = Path("data/enrichments/forward_index.json")
    with open(forward_path, "w", encoding="utf-8") as f:
        json.dump(forward_index, f, indent=2, ensure_ascii=False)
    
    print(f"Built reverse index with {len(reverse_index)} titles mapping to {sum(len(v) for v in reverse_index.values())} chunks.")
    print(f"Built forward index with {len(forward_index)} titles mapping to doc hashes.")

if __name__ == "__main__":
    build_citation_indexes()
