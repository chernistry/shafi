"""Manual top-EV overrides for case documents."""

from __future__ import annotations

from shafi.ingestion.manual_domain_overrides import ManualDomainOverride

M = ManualDomainOverride

MANUAL_CASE_OVERRIDES: dict[str, ManualDomainOverride] = {
    "6306079a16b1dec85690f75c715cdbd78b0685a3e19ee30250d481bc32f2e29a": M(
        title="Okpara v Oralee [2025] DIFC SCT 514",
        case_number="SCT 514/2025",
        claimant=("Okpara",),
        respondent=("Oralee",),
        judges=("Justice Sapna Jhangiani", "Maha Al Mheiri"),
        aliases=("Okpara v Oralee", "[2025] DIFC SCT 514"),
        note="Restore SCT 514/2025 case metadata from order-like document.",
    ),
    "5d3df6d69fac3ef91e13ac835b43a35e9e434fbc7e72ea5c01e288d69b66e6a2": M(
        title="ENF 269/2023 (1) Ozias (2) Ori (3) Octavio v (1) Obadiah (2) Oaklen",
        case_number="ENF 269/2023",
        judges=("Chief Justice Wayne Martin",),
        aliases=("ENF 269/2023", "Ozias v Obadiah", "Oaklen"),
        note="Keep only text-grounded caption, case number, panel, and aliases for mixed-role enforcement appeal.",
    ),
    "09660f78c26cd56c08c7253ed21ba01fb246092f482ccd8acd8e6f9b6fd2d917": M(
        title="Olexa v Odon [2025] DIFC SCT 295",
        case_number="SCT 295/2025",
        claimant=("Olexa",),
        respondent=("Odon",),
        judges=("Justice Sapna Jhangiani", "Maitha AlShehhi"),
        aliases=("Olexa v Odon", "[2025] DIFC SCT 295"),
        note="Remove bogus respondent fragment and restore SCT 295/2025 metadata.",
    ),
    "3f8a5ea0e051ba3af7a02da340c911fe0970ebece6c852c3e61a10c00cac6d6e": M(
        title=(
            "DEC 001/2025 Techteryx Ltd v (1) Aria Commodities DMCC (2) Mashreq Bank PSC "
            "(3) Emirates NBD Bank PJSC (4) Abu Dhabi Islamic Bank PJSC"
        ),
        case_number="DEC 001/2025",
        claimant=("Techteryx Ltd.",),
        judges=("Justice Michael Black KC",),
        aliases=("DEC 001/2025", "Techteryx v Aria Commodities", "[2025] DIFC DEC 001"),
        note="Drop ungrounded respondent roles from DEC 001/2025 amended order packet.",
    ),
    "78ffe994cdc61ce6a2a6937c79fc52751bb5d2b4eaa4019f088fbccf70569c26": M(
        title="CA 004/2025 (1) Mr Oran (2) Oaken v Oved",
        case_number="CA 004/2025",
        claimant=("Mr Oran", "Oaken"),
        respondent=("Oved",),
        judges=("Chief Justice Wayne Martin",),
        aliases=("CA 004/2025",),
        note="Restore CA 004/2025 caption and missing judge.",
    ),
    "03b621728fe29eb6113fcdb57f6458d793fd2d5c5b833ae26d40f04a29c85359": M(
        title="CA 005/2025 LXT Real Estate Broker L.L.C v SIR Real Estate LLC",
        case_number="CA 005/2025",
        claimant=("LXT Real Estate Broker L.L.C",),
        respondent=("SIR Real Estate LLC",),
        judges=("Chief Justice Wayne Martin", "Justice Sir Peter Gross", "Justice Rene Le Miere"),
        aliases=("CA 005/2025",),
        note="Remove bogus respondent/judge fragments from CA 005/2025.",
    ),
    "437568a801115019fe8278385c0484bdf07ab86f9a499ecaba2b7969b37c764b": M(
        title="CA 005/2025 LXT Real Estate Broker L.L.C v SIR Real Estate LLC",
        case_number="CA 005/2025",
        claimant=("LXT Real Estate Broker L.L.C",),
        respondent=("SIR Real Estate LLC",),
        judges=("Chief Justice Wayne Martin", "Justice Sir Peter Gross", "Justice Rene Le Miere"),
        aliases=("CA 005/2025", "Reasons of the Court of Appeal for the Order dated 13 January 2026"),
        note="Align separate CA 005/2025 reasons document with clean caption/panel metadata.",
    ),
    "c98c1475692bc22f4abab6a7a7d7969467c94e46a7e68919aaf127179ebf3f54": M(
        title="TCD 001/2024 Architeriors Interior Design (L.L.C) v Emirates National Investment Co (L.L.C)",
        case_number="TCD 001/2024",
        claimant=("Architeriors Interior Design (L.L.C)",),
        respondent=("Emirates National Investment Co (L.L.C)",),
        judges=("Chief Justice Wayne Martin", "Justice Roger Stewart"),
        aliases=("TCD 001/2024", "Architeriors Interior Design (L.L.C) v Emirates National Investment Co (L.L.C)"),
        note="Expand truncated respondent and add full caption alias for TCD 001/2024 appeal material.",
    ),
    "6248961b681ea0deb189f354be0c8286f35974dcdb211c13c921c3dd0e566a6e": M(
        title="CFI 016/2025 Omar Ben Hallam v Natixis",
        case_number="CFI 016/2025",
        claimant=("Omar Ben Hallam",),
        judges=("Deputy Chief Justice Ali Al Madhani",),
        aliases=("CFI 016/2025", "Omar Ben Hallam v Natixis"),
        note="Drop ungrounded respondent role from CFI 016/2025 defendant caption.",
    ),
    "1b446e196b4d1752241c8ff689a31ea705e98ad0c16b9d343c303664f72b64a1": M(
        title="CFI 057/2025 Clyde & Co LLP v (1) Union Properties P.J.S.C. (2) UPP Capital Investment LLC",
        case_number="CFI 057/2025",
        claimant=("Clyde & Co LLP",),
        judges=("Justice Roger Stewart KC",),
        aliases=("CFI 057/2025",),
        note="Drop ungrounded respondent roles from CFI 057/2025 defendants caption.",
    ),
    "839de9798f377492eee68f82b202d7cd3544be83d799f6226a02f3678c9bb914": M(
        title=(
            "DEC 001/2025 Techteryx Ltd v (1) Aria Commodities DMCC (2) Mashreq Bank PSC "
            "(3) Emirates NBD Bank PJSC (4) Abu Dhabi Islamic Bank PJSC"
        ),
        case_number="DEC 001/2025",
        claimant=("Techteryx Ltd.",),
        judges=("Justice Michael Black KC",),
        aliases=("DEC 001/2025", "[2025] DIFC DEC 001"),
        note="Drop ungrounded respondent roles from alternate DEC 001/2025 packet.",
    ),
    "897ab23ed5a70034d3d708d871ad1da8bc7b6608d94b1ca46b5d578d985d3c13": M(
        title="CFI 067/2025 Coinmena B.S.C. (C) v Foloosi Technologies Ltd",
        case_number="CFI 067/2025",
        claimant=("Coinmena B.S.C. (C)",),
        judges=("Justice Shamlan Al Sawalehi",),
        aliases=("CFI 067/2025",),
        note="Drop ungrounded respondent role from CFI 067/2025 defendant caption.",
    ),
    "0471e83c1ea18086cfb6b3ff51da6f22b0efee337f10315b2593f782297ccb84": M(
        title="TCD 001/2024 Architeriors Interior Design (L.L.C) v Emirates National Investment Co (L.L.C)",
        case_number="TCD 001/2024",
        claimant=("Architeriors Interior Design (L.L.C)",),
        respondent=("Emirates National Investment Co (L.L.C)",),
        judges=("Chief Justice Wayne Martin",),
        aliases=("TCD 001/2024", "Architeriors Interior Design (L.L.C) v Emirates National Investment Co (L.L.C)"),
        note="Expand truncated respondent and add full caption alias for first TCD 001/2024 packet.",
    ),
    "1a255edc261961ec64870466a27ac4e25b5ebc2abe298e1b69f8dd2fc27288f6": M(
        title=(
            "DEC 001/2025 Techteryx Ltd v (1) Aria Commodities DMCC (2) Mashreq Bank PSC "
            "(3) Emirates NBD Bank PJSC (4) Abu Dhabi Islamic Bank PJSC"
        ),
        case_number="DEC 001/2025",
        claimant=("Techteryx Ltd.",),
        judges=("Justice Michael Black KC",),
        aliases=("DEC 001/2025", "[2025] DIFC DEC 001"),
        note="Drop ungrounded respondent roles from DEC 001/2025 merits packet.",
    ),
    "43b9033ec25f016635f416e4b49558f5bc29f6c55bbd72bea1d1d840b170a371": M(
        title="ENF 053/2025 Standard Chartered PLC v Standard Chartered Bank",
        case_number="ENF 053/2025",
        claimant=("Standard Chartered PLC",),
        respondent=("Standard Chartered Bank",),
        judges=("Deputy Chief Justice Ali Al Madhani",),
        aliases=("ENF 053/2025", "Standard Chartered PLC v Standard Chartered Bank"),
        note="Expand respondent and add full caption alias for Standard Chartered enforcement order.",
    ),
    "c66b8374d326d16ec5ff48195825798c3ba696c787ad8a59c23a7a2908f34a0d": M(
        title="ENF 053/2025 Standard Chartered PLC v Standard Chartered Bank",
        case_number="ENF 053/2025",
        claimant=("Standard Chartered PLC",),
        respondent=("Standard Chartered Bank",),
        judges=("Deputy Chief Justice Ali Al Madhani",),
        aliases=("ENF 053/2025", "Standard Chartered PLC v Standard Chartered Bank"),
        note="Expand respondent and add full caption alias for second ENF 053/2025 packet.",
    ),
    "62930da32fa3172edf2f2bbf3da268455bd99a7b5fab34d72358730d8cd5da30": M(
        title="TCD 001/2024 Architeriors Interior Design (L.L.C) v Emirates National Investment Co (L.L.C)",
        case_number="TCD 001/2024",
        claimant=("Architeriors Interior Design (L.L.C)",),
        respondent=("Emirates National Investment Co (L.L.C)",),
        judges=("Justice Roger Stewart",),
        aliases=("TCD 001/2024", "Architeriors Interior Design (L.L.C) v Emirates National Investment Co (L.L.C)"),
        note="Expand truncated respondent and add full caption alias for merits TCD 001/2024 packet.",
    ),
}
