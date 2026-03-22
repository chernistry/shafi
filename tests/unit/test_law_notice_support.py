from __future__ import annotations

from shafi.core.law_notice_support import (
    extract_enactment_authority,
    extract_enactment_date,
    extract_law_number_year,
    is_law_like_order_title,
    normalize_law_like_title,
)


def test_normalize_law_like_title_accepts_uae_decree_and_resolution_titles() -> None:
    decree_title = normalize_law_like_title(
        title="__________________________________",
        source_text="DUBAI DECREE No. 12 OF 2024\nSome other text.",
    )
    resolution_title = normalize_law_like_title(
        title="",
        source_text="CABINET RESOLUTION No. 7 OF 2023\nPublished in the Official Gazette.",
    )

    assert decree_title == "DUBAI DECREE No. 12 OF 2024"
    assert resolution_title == "CABINET RESOLUTION No. 7 OF 2023"


def test_extract_law_number_year_accepts_uae_decree_and_resolution_titles() -> None:
    decree_number, decree_year = extract_law_number_year(
        title="Dubai Decree No. 12 of 2024",
        source_text="",
    )
    resolution_number, resolution_year = extract_law_number_year(
        title="",
        source_text="Federal Resolution No. 7 of 2023",
    )

    assert (decree_number, decree_year) == ("12", "2024")
    assert (resolution_number, resolution_year) == ("7", "2023")


def test_extract_enactment_authority_and_date_handle_dubai_notice_language() -> None:
    source_text = (
        "ENACTMENT NOTICE\n"
        "We, Mohammed bin Rashid Al Maktoum, Ruler of Dubai hereby enact\n"
        "on this 01 day of March 2024\n"
        "the\n"
        "DATA PROTECTION LAW\n"
    )

    authority = extract_enactment_authority(source_text=source_text, fallback="the")
    date = extract_enactment_date(source_text=source_text, fallback="")

    assert authority == "Mohammed bin Rashid Al Maktoum, Ruler of Dubai"
    assert date == "1 March 2024"


def test_extract_enactment_authority_handles_issued_and_administered_notice_phrases() -> None:
    issued = extract_enactment_authority(
        source_text="Issued by DIFC Authority. Commencement date 1 January 2005.",
        fallback="",
    )
    administered = extract_enactment_authority(
        source_text="This Law is administered by the DFSA.",
        fallback="",
    )
    registrar = extract_enactment_authority(
        source_text="By order of the Registrar.",
        fallback="",
    )

    assert issued == "DIFC Authority"
    assert administered == "DFSA"
    assert registrar == "Registrar"


def test_extract_enactment_date_handles_commencement_phrases() -> None:
    in_force = extract_enactment_date(
        source_text="FINANCIAL COLLATERAL REGULATIONS\nIn force on 1 November 2019.",
        fallback="",
    )
    effective_from = extract_enactment_date(
        source_text="This Regulation is effective from 15 February 2024.",
        fallback="",
    )

    assert in_force == "1 November 2019"
    assert effective_from == "15 February 2024"


def test_extract_enactment_date_ignores_relative_commencement_language() -> None:
    relative = extract_enactment_date(
        source_text="This Law shall come into force on the 5th business day after enactment.",
        fallback="",
    )

    assert relative == ""


def test_extract_enactment_date_ignores_relative_fallback_phrase() -> None:
    relative = extract_enactment_date(
        source_text="No literal date appears in this notice.",
        fallback="the date specified in the Enactment Notice in respect of this Law",
    )

    assert relative == ""


def test_is_law_like_order_title_matches_uae_legal_instrument_terms() -> None:
    assert is_law_like_order_title(
        title="UAE Decree No. 12 of 2024",
        source_text="",
    )
