import json

entry = {
    "from": "eyal",
    "timestamp": "2026-03-22T13:30:00Z",
    "type": "patch_ready",
    "topic": "recoverable_nopg_patches",
    "message": (
        "[EYAL] PATCH SCRIPT READY: scripts/patch_recoverable_nopg.py\n\n"
        "Found 2 confirmed retrieval failures with HIGH-confidence answers from corpus registry.\n"
        "Plus 1 DB answerer bug (CFI 043-2020, requires investigation).\n\n"
        "=== PATCH 1 — DRA Order 1/2016 Issuer ===\n"
        "QID: 88f0d2177403691514b6d4d39eefa4d0ff3ca2a712d5d9c9b5baf5aee17ef9e1\n"
        "Q: Who issued DRA Order No. 1 of 2016 in respect of the DIFC Courts Mandatory Code of Conduct?\n"
        "OLD: There is no information on this question.\n"
        "NEW: DRA Order No. 1 of 2016 was issued by Michael Hwang, Head of the Dispute Resolution Authority (DRA), in exercise of powers conferred by Article 8(5)(b) of Dubai Law No. 7 of 2014.\n"
        "Source: DRA Order page 1 text (direct quote from 'I, Michael Hwang, Head of the DRA, make the following Order')\n"
        "Pages: doc_id d403315f013ec5..., page 1\n\n"
        "=== PATCH 2 — Prescribed Company Regulations 'Control' Definition ===\n"
        "QID: bff953d173d948f7456e8a6705c97eab5637e10da2f978be865c6aa5166bf484\n"
        "Q: How does the Prescribed Company Regulations define 'Control'?\n"
        "OLD: There is no information on this question.\n"
        "NEW: The Prescribed Company Regulations define 'Control' as the power to secure that a company's affairs are conducted in accordance with one's wishes, through holding of shares, voting power (directly or indirectly), or powers conferred by the Articles of Association.\n"
        "Source: Prescribed Company Regulations page 4 (verbatim from definition section)\n"
        "Pages: doc_id 4c433b99adaf98..., page 4\n\n"
        "=== BUG REPORT — DB Answerer Short-Circuit ===\n"
        "QID: 9e0912e4139637a56a8aa04484231b287f8c0a21105666f90d3832d554453a8e\n"
        "Q: In CFI 043-2020 Bank Of Baroda...what was Ellen Radley's conclusion as the Claimant's expert on the disputed signatures?\n"
        "SYMPTOM: DB answerer returned 'no information' in 2ms, 0 pages, 0 tokens.\n"
        "BUG: DB answerer incorrectly routed this complex case-law question and short-circuited to no-info without RAG.\n"
        "FIX NEEDED: query_contract.py should NOT route CFI case-analysis questions to DB answerer.\n"
        "OWNER: OREV or DAGAN.\n\n"
        "=== OTHER CONFIRMED RETRIEVAL FAILURES (for OREV) ===\n"
        "QID 4157cb37360c (Order 2/2017 Part 44): doc exists in corpus, 0 pages returned\n"
        "QID 81228aad5d70 (DPL + Leasing commencement dates): registry has both dates\n"
        "  DPL 2020 commence = 1 September 2023 | Leasing Law No. 1 of 2020 commence = 22 February 2023\n\n"
        "TO APPLY PATCHES: uv run python scripts/patch_recoverable_nopg.py --output data/private_submission_V15_HYBRID_patched.json\n"
        "THEN: SASHA reviews + verifies before submitting patched version."
    )
}

with open(".sdd/agents/BULLETIN.jsonl", "a") as f:
    f.write(json.dumps(entry) + "\n")

print("BULLETIN posted")
