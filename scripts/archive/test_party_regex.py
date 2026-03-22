import re

_PARTY_ROLE_RE = re.compile(
    r"\b(?:claimant|claimants|respondent|respondents|appellant|appellants|applicant|applicants|"
    r"defendant|defendants|plaintiff|plaintiffs|petitioner|petitioners)\b[:\s-]*([A-Z][A-Za-z0-9&.,'()/-]{2,80})",
    re.IGNORECASE,
)

samples = [
    "The Claimant was represented by Mr. Smith.",
    "RESPONDENT: Odon",
    "The claimant claims that...",
    "Appellants: (1) Oran (2) Oaken",
    "The Claimant was from...",
]

for s in samples:
    matches = _PARTY_ROLE_RE.finditer(s)
    for m in matches:
        print(f"Sample: '{s}' -> Match: '{m.group(1)}'")
