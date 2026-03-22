import json
import re

def build_forward_index():
    index_path = "dataset_document_index.json"
    try:
        with open(index_path, "r") as f:
            doc_index = json.load(f)
    except FileNotFoundError:
        print("Could not find dataset_document_index.json")
        return

    forward_index = {}

    for doc_hash, meta in doc_index.items():
        title = meta.get("title", "")
        if not title:
            continue
            
        doc_hash_prefix = doc_hash[:16]
        
        # Normalize and add full title
        norm_title = title.strip().lower()
        if norm_title not in forward_index:
            forward_index[norm_title] = []
        if doc_hash_prefix not in forward_index[norm_title]:
            forward_index[norm_title].append(doc_hash_prefix)
            
        # Extract sub-titles like "DIFC Law No. 7 of 2018"
        # Often in parentheses: "DIFC Operating Law (DIFC Law No. 7 of 2018)"
        match = re.search(r'\(([^)]+)\)', title)
        if match:
            sub_title = match.group(1).strip().lower()
            if sub_title not in forward_index:
                forward_index[sub_title] = []
            if doc_hash_prefix not in forward_index[sub_title]:
                forward_index[sub_title].append(doc_hash_prefix)

    out_path = "data/enrichments/forward_index.json"
    with open(out_path, "w") as f:
        json.dump(forward_index, f, indent=2, ensure_ascii=False)
        
    print(f"Forward index built at {out_path} with {len(forward_index)} titles.")

if __name__ == "__main__":
    build_forward_index()
