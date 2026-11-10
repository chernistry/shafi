import fitz
from pathlib import Path
import json

def audit_ocr(pdf_dir: str):
    pdf_dir = Path(pdf_dir)
    pdf_files = list(pdf_dir.rglob("*.pdf"))
    
    results = {}
    
    print(f"Auditing OCR quality for {len(pdf_files)} PDFs...")
    
    for i, pdf_path in enumerate(pdf_files, 1):
        doc_id = pdf_path.stem[:16]
        try:
            with fitz.open(str(pdf_path)) as pdf:
                total_chars = 0
                total_pages = len(pdf)
                if total_pages == 0:
                    results[doc_id] = {"status": "EMPTY", "pages": 0}
                    continue
                
                bad_pages = 0
                for page in pdf:
                    text = page.get_text("text").strip()
                    # Heuristic: char density
                    # Average page area is roughly 600x800 pts
                    area = page.rect.width * page.rect.height
                    density = len(text) / area if area > 0 else 0
                    
                    # Heuristic: garbled text (high ratio of symbols/non-ascii)
                    non_ascii = len(re.findall(r'[^\x00-\x7F]', text))
                    garbled_ratio = non_ascii / len(text) if len(text) > 0 else 0
                    
                    if density < 0.001 or garbled_ratio > 0.3:
                        bad_pages += 1
                    
                    total_chars += len(text)
                
                avg_chars_per_page = total_chars / total_pages
                
                status = "OK"
                if avg_chars_per_page < 100:
                    status = "SCAN_ONLY"
                elif bad_pages / total_pages > 0.5:
                    status = "NOISY"
                
                results[doc_id] = {
                    "status": status,
                    "pages": total_pages,
                    "avg_chars": avg_chars_per_page,
                    "bad_pages": bad_pages
                }
        except Exception as e:
            results[doc_id] = {"status": "ERROR", "error": str(e)}
        
        if i % 50 == 0:
            print(f"Progress: {i}/{len(pdf_files)} audited.")

    with open("data/ocr_audit_report.json", "w") as f:
        json.dump(results, f, indent=2)
    
    flagged = [k for k, v in results.items() if v["status"] != "OK"]
    print(f"OCR AUDIT COMPLETE. {len(flagged)} docs flagged out of {len(pdf_files)}.")
    if flagged:
        print(f"Flagged docs: {flagged[:10]}...")

import re
if __name__ == "__main__":
    audit_ocr("data/private")
