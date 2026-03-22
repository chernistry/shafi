"""Manual top-EV overrides for law-like documents."""

from __future__ import annotations

from rag_challenge.ingestion.manual_domain_overrides import ManualDomainOverride

M = ManualDomainOverride

MANUAL_LAW_OVERRIDES: dict[str, ManualDomainOverride] = {
    "72ea171147bf30326fe6fd2e6798f607c7cef4bf9d43761dbccd2f1b6a356849": M(
        title="Operating Law DIFC Law No. 7 of 2018",
        short_title="Operating Law",
        aliases=("OPERATING LAW", "DIFC Law No. 7 of 2018"),
        law_number="7",
        year="2018",
        note="Normalize Operating Law title and numbering from first-page caption.",
    ),
    "302a0bd8d67775e8dc5960ecec7879be566300d8b32c4b0153ba15ebdb279425": M(
        title="General Partnership Law DIFC Law No. 11 of 2004",
        short_title="General Partnership Law",
        aliases=("GENERAL PARTNERSHIP LAW", "DIFC Law No. 11 of 2004"),
        law_number="11",
        year="2004",
        note="Normalize General Partnership Law title and numbering from first-page caption.",
    ),
    "b82ac8228e051d96bf8d706d3251893ebff1c9457b066fce3b7cb99af956f2a7": M(
        title="Limited Liability Partnership Law DIFC Law No. 5 of 2004",
        short_title="Limited Liability Partnership Law",
        aliases=("LIMITED LIABILITY PARTNERSHIP LAW", "DIFC Law No. 5 of 2004"),
        law_number="5",
        year="2004",
        note="Normalize Limited Liability Partnership Law title and numbering from first-page caption.",
    ),
    "ff746f7b583490a80ba104361c0a82a1ebbf7ed9097cd03dc49d744cb5057761": M(
        title="Law on the Application of Civil and Commercial Laws in the DIFC DIFC Law No. 3 of 2004",
        short_title="Law on the Application of Civil and Commercial Laws in the DIFC",
        aliases=(
            "LAW ON THE APPLICATION OF CIVIL AND COMMERCIAL LAWS IN THE DIFC",
            "DIFC Law No. 3 of 2004",
        ),
        law_number="3",
        year="2004",
        note="Expand truncated Application of Civil and Commercial Laws title from first-page caption.",
    ),
    "22442c5ee999e2519c68de908be511875a84f2b810ed540c2dcfcbcc65031434": M(
        title="Foundations Law DIFC Law No. 3 of 2018",
        short_title="Foundations Law",
        aliases=("FOUNDATIONS LAW", "DIFC Law No. 3 of 2018"),
        law_number="3",
        year="2018",
        note="Normalize Foundations Law title and numbering from first-page caption.",
    ),
    "efa6f9ce52c6a8f3f2725d44a7c328211ad9ad1dd2fdb7ff0f0dd1005369a91a": M(
        title="Companies Law DIFC Law No. 5 of 2018",
        short_title="Companies Law",
        aliases=("COMPANIES LAW", "DIFC Law No. 5 of 2018"),
        law_number="5",
        year="2018",
        note="Normalize Companies Law title and numbering from order-like bucket.",
    ),
    "33bc02044716acdfedb164b065bdaec098aaadcae863c591f9931c88e7307d16": M(
        title="Employment Law DIFC Law No. 2 of 2019",
        short_title="Employment Law",
        aliases=("EMPLOYMENT LAW", "DIFC Law No. 2 of 2019"),
        law_number="2",
        year="2019",
        note="Normalize Employment Law title and numbering.",
    ),
    "8c6d34f8833e88c664c99576d875f0d1bcab6bd6360e9c6fe6dec3f50f9bde01": M(
        title="Leasing Law DIFC Law No. 1 of 2020",
        short_title="Leasing Law",
        aliases=("LEASING LAW", "DIFC Law No. 1 of 2020"),
        law_number="1",
        year="2020",
        note="Normalize Leasing Law title and numbering.",
    ),
    "93740190cd7d30dfa50217867b46a896d33c4bfe188d6cab5586eaa781aa7cb3": M(
        title="Strata Title Law DIFC Law No. 5 of 2007",
        short_title="Strata Title Law",
        aliases=("STRATA TITLE LAW", "DIFC Law No. 5 of 2007"),
        law_number="5",
        year="2007",
        note="Normalize Strata Title Law title and numbering.",
    ),
    "02ec30ff6c98fa3adccd3834d73d76a2ae11dab91f2b4ea3501ff24db27436cc": M(
        title="Leasing Regulations",
        short_title="Leasing Regulations",
        aliases=("LEASING REGULATIONS",),
        issued_by="Board of Directors of the DIFCA",
        law_number="1",
        year="2020",
        note="Add issuing authority and stable title for Leasing Regulations.",
    ),
    "69ecacbd263a1e75765e4f12caaf73a08012b1c2e11e5c14bff43bf6331f3003": M(
        title="Employment Regulations",
        short_title="Employment Regulations",
        aliases=("EMPLOYMENT REGULATIONS",),
        law_number="2",
        year="2019",
        note="Remove junk commencement metadata and stabilize Employment Regulations title.",
    ),
    "ac3ba9cf24ee4f5db90ea091b7df7da839b78daeb0ff4020afdce8bb3299d357": M(
        title="Strata Title Regulations",
        short_title="Strata Title Regulations",
        aliases=("STRATA TITLE REGULATIONS",),
        issued_by="Board of Directors of the DIFCA",
        law_number="5",
        year="2007",
        note="Add authority and stable title for Strata Title Regulations.",
    ),
    "bcbf1b4067fd79dca18a033ecb026559be2e75c80eae582ae17f94e7b04ae19e": M(
        title="Financial Collateral Regulations",
        short_title="Financial Collateral Regulations",
        aliases=("FINANCIAL COLLATERAL REGULATIONS",),
        issued_by="Board of Directors of the DIFC",
        law_number="9",
        year="2005",
        note="Add authority and numbering for Financial Collateral Regulations.",
    ),
    "20be16a68c4d24768f569bcde5ddb29437204006c0c4262296c350b365672cd1": M(
        title="Dematerialised Investments Regulations (DIR)",
        short_title="Dematerialised Investments Regulations",
        aliases=("DIR", "DEMATERIALISED INVESTMENTS REGULATIONS"),
        note="Expand truncated Dematerialised title into stable regulation form.",
    ),
    "406365706456d37cf871ccb1cfc4618bca6af7d838fbefca08b4a87a3cbe583d": M(
        title="Incorporated Cell Company (ICC) Regulations",
        short_title="Incorporated Cell Company (ICC) Regulations",
        aliases=("ICC Regulations", "INCORPORATED CELL COMPANY (ICC) REGULATIONS"),
        enactment_date="1 May 2019",
        note="Expand ICC title and restore first-page effective date.",
    ),
}
