import json
import glob
import os
from pathlib import Path

def build_indexes():
    enrichment_dir = Path("data/enrichments")
    if not enrichment_dir.exists():
        print("No enrichments directory.")
        return

    reverse_index = {}
    summary_index = {}

    for file_path in enrichment_dir.glob("*.json"):
        if file_path.name in ("reverse_index.json", "summary_index.json"):
            continue
            
        stem = file_path.stem
        parts = stem.split("_")
        if len(parts) < 2:
            continue
            
        doc_hash = parts[0]
        page_num = parts[1]
        chunk_id = f"{doc_hash}_{page_num}"
        
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                continue
                
        citations = data.get("external_citations", [])
        
        if doc_hash not in summary_index:
            summary_index[doc_hash] = {}
            
        summary_index[doc_hash][page_num] = {
            "has_citations": len(citations) > 0,
            "citation_count": len(citations)
        }
        
        for citation in citations:
            cited_text = citation.get("cited", "").strip()
            if not cited_text:
                continue
                
            if cited_text not in reverse_index:
                reverse_index[cited_text] = []
            if chunk_id not in reverse_index[cited_text]:
                reverse_index[cited_text].append(chunk_id)

    # Write output
    with open(enrichment_dir / "reverse_index.json", "w", encoding="utf-8") as f:
        json.dump(reverse_index, f, indent=2, ensure_ascii=False)
        
    with open(enrichment_dir / "summary_index.json", "w", encoding="utf-8") as f:
        json.dump(summary_index, f, indent=2, ensure_ascii=False)

    print(f"NOAM ENRICHMENT INDEX BUILT. {len(summary_index)} docs indexed, {len(reverse_index)} unique citation targets. File: data/enrichments/reverse_index.json")

if __name__ == "__main__":
    build_indexes()
