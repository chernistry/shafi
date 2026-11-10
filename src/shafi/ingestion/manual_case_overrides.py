"""Manual metadata overrides for case documents.

Overrides are supplied by the user and none ship with the repository. Keys are
stable parsed-document IDs; values are ``ManualDomainOverride`` payloads that
patch case title, case number, parties, judges and aliases after compilation.
"""

from __future__ import annotations

from shafi.ingestion.manual_domain_overrides import ManualDomainOverride

M = ManualDomainOverride

MANUAL_CASE_OVERRIDES: dict[str, ManualDomainOverride] = {}
