import re

_PARTY_ROLE_RE_NEW = re.compile(
    r"\b(?i:claimant|claimants|respondent|respondents|appellant|appellants|applicant|applicants|"
    r"defendant|defendants|plaintiff|plaintiffs|petitioner|petitioners)\b[:\s-]*([A-Z\(][A-Za-z0-9&.,'()/-]{2,80})"
)

samples = [
    "The Claimant was represented by Mr. Smith.",
    "RESPONDENT: Odon",
    "The claimant claims that...",
    "Appellants: (1) Oran (2) Oaken",
    "The Claimant was from...",
    "THE CLAIMANT WAS FROM...",
]

for s in samples:
    matches = _PARTY_ROLE_RE_NEW.finditer(s)
    matched = False
    for m in matches:
        matched = True
        print(f"Sample: '{s}' -> Match: '{m.group(1)}'")
    if not matched:
        print(f"Sample: '{s}' -> NO MATCH")
