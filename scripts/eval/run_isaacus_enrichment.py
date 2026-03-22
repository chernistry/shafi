import argparse
import asyncio
import hashlib
import json
import logging
import os
import sys
from pathlib import Path

import httpx
import fitz
import tiktoken

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("enrichment")

def get_doc_hash(path: Path) -> str:
    if len(path.stem) == 64:
        return path.stem[:16]
    return hashlib.sha256(str(path.resolve()).encode("utf-8")).hexdigest()[:16]

def extract_pages(pdf_path: Path) -> list[str]:
    pages = []
    try:
        with fitz.open(str(pdf_path)) as pdf:
            for page in pdf:
                text = page.get_text("text")
                if isinstance(text, str):
                    pages.append(text.strip())
    except Exception as e:
        logger.error(f"Failed to read PDF {pdf_path}: {e}")
    return pages

async def process_page(
    client: httpx.AsyncClient, 
    text: str, 
    doc_hash: str, 
    page_num: int, 
    output_dir: Path, 
    semaphore: asyncio.Semaphore,
    dry_run: bool
):
    out_path = output_dir / f"{doc_hash}_{page_num:04d}.json"
    if out_path.exists():
        return
    
    if dry_run:
        return

    async with semaphore:
        for attempt in range(3):
            try:
                resp = await client.post(
                    "https://api.isaacus.com/v1/enrichments",
                    json={
                        "model": "kanon-2-enricher",
                        "texts": [text],
                        "metadata": {"doc_id": doc_hash, "page_num": page_num}
                    },
                    timeout=30.0
                )
                resp.raise_for_status()
                data = resp.json()
                
                # Extract from API schema
                results = data.get("results", [])
                doc_obj = results[0].get("document", {}) if results else {}
                
                external_citations = []
                for ext in doc_obj.get("external_documents", []) + doc_obj.get("crossreferences", []):
                    name_span = ext.get("name", {}) if isinstance(ext, dict) else {}
                    name_start, name_end = name_span.get("start", 0), name_span.get("end", 0)
                    name_str = text[name_start:name_end] if name_end > name_start else ""
                    
                    pin_str = ""
                    for pin in ext.get("pinpoints", []):
                        p_start, p_end = pin.get("start", 0), pin.get("end", 0)
                        if p_end > p_start:
                            pin_str += text[p_start:p_end] + " "
                            
                    cited = f"{pin_str.strip()} of {name_str}".strip() if pin_str else name_str
                    if not cited and "text" in ext:
                        cited = ext["text"] # Fallback if crossreferences uses "text"
                        
                    start_idx = max(0, min(name_start or 0, 0) - 100)
                    end_idx = min(len(text), max(name_end or 0, 0) + 100)
                    context = text[start_idx:end_idx]
                    
                    if cited:
                        external_citations.append({"cited": cited, "context": context})

                final_json = {
                    "document_type": doc_obj.get("type"),
                    "jurisdiction": doc_obj.get("jurisdiction"),
                    "external_citations": external_citations,
                    "defined_terms": doc_obj.get("terms", []),
                    "persons": doc_obj.get("persons", []),
                    "typed_dates": doc_obj.get("dates", []),
                    "junk_spans": doc_obj.get("junk", [])
                }
                
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(final_json, f, ensure_ascii=False, indent=2)
                return
            except httpx.HTTPStatusError as e:
                if e.response.status_code in (429, 500, 502, 503, 504):
                    logger.warning(f"HTTP {e.response.status_code} for {doc_hash} page {page_num}, retrying...")
                    await asyncio.sleep(2 ** attempt)
                else:
                    logger.error(f"HTTP error {e.response.status_code} for {doc_hash} page {page_num}: {e.response.text}")
                    break
            except Exception as e:
                logger.warning(f"Error {e} for {doc_hash} page {page_num}, retrying...")
                await asyncio.sleep(2 ** attempt)
        
        logger.error(f"Failed to enrich {doc_hash} page {page_num} after 3 attempts.")

async def main():
    parser = argparse.ArgumentParser(description="Run Isaacus enrichment on PDFs")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    api_key = os.environ.get("ISAACUS_API_KEY")
    if not api_key:
        logger.error("ISAACUS_API_KEY environment variable not set")
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    pdf_files = list(args.input_dir.rglob("*.pdf"))
    logger.info(f"Found {len(pdf_files)} PDF files in {args.input_dir}")

    all_tasks = []
    total_tokens = 0
    enc = tiktoken.get_encoding("cl100k_base")

    doc_pages_map = {}
    for pdf_path in pdf_files:
        doc_hash = get_doc_hash(pdf_path)
        pages = extract_pages(pdf_path)
        doc_pages_map[(pdf_path, doc_hash)] = pages
        for p in pages:
            total_tokens += len(enc.encode(p))

    cost = (total_tokens / 1_000_000) * 3.50
    logger.info(f"Estimated tokens: {total_tokens:,}")
    logger.info(f"Estimated cost: ${cost:.2f}")

    if args.dry_run:
        logger.info("Dry run complete.")
        return

    semaphore = asyncio.Semaphore(10)
    
    async with httpx.AsyncClient(headers={"Authorization": f"Bearer {api_key}"}) as client:
        for i, ((pdf_path, doc_hash), pages) in enumerate(doc_pages_map.items(), 1):
            doc_tasks = [process_page(client, text, doc_hash, j, args.output_dir, semaphore, args.dry_run) for j, text in enumerate(pages, 1)]
            if doc_tasks:
                await asyncio.gather(*doc_tasks)
            if i % 10 == 0 or i == len(doc_pages_map):
                logger.info(f"Progress: {i}/{len(doc_pages_map)} docs enriched.")

if __name__ == "__main__":
    asyncio.run(main())
