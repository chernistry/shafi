# Golden Labels Scoring Report

- **Matched questions**: 100 / 100
- **Unmatched**: 0
- **Overall Grounding F-beta (beta=2.5)**: 0.4218
- **Weighted Grounding F-beta**: 0.4443 (high=1.0, medium=0.5, low=0.25)
- **Trusted-only Grounding F-beta**: 0.4645 (high confidence only)
- **Exact Match Rate (deterministic types)**: 0.7714 (54/70)
- **Trusted Exact Match Rate**: 0.7576
- **Trusted Regressions**: 16

## By Answer Type

| Type | F-beta (mean) | F-beta (n) | Exact Match | Exact (n) |
|------|--------------|------------|-------------|-----------|
| boolean | 0.3914 | 35 | 0.7143 | 35 |
| date | 1.0000 | 1 | 1.0000 | 1 |
| free_text | 0.4536 | 30 | - | - |
| name | 0.3214 | 14 | 0.7857 | 14 |
| names | 0.7890 | 3 | 1.0000 | 3 |
| number | 0.4118 | 17 | 0.8235 | 17 |

## By Confidence Tier

| Confidence | F-beta (mean) | F-beta (n) | Exact Match | Exact (n) |
|------------|--------------|------------|-------------|-----------|
| high | 0.4645 | 81 | 0.7576 | 66 |
| low | 0.2000 | 5 | 1.0000 | 1 |
| medium | 0.2534 | 14 | 1.0000 | 3 |

## By Question Family

| Family | F-beta (mean) | F-beta (n) | Exact Match | Exact (n) |
|--------|--------------|------------|-------------|-----------|
| administration | 0.5735 | 4 | 0.0000 | 1 |
| boolean | 0.3414 | 11 | 0.8182 | 11 |
| compare | 0.4889 | 30 | 0.8077 | 26 |
| enactment | 0.2861 | 13 | 0.5000 | 6 |
| free_text_other | 0.3941 | 11 | - | - |
| names | 0.3223 | 12 | 0.8333 | 12 |
| number | 0.3077 | 13 | 0.7692 | 13 |
| outcome_costs | 0.7685 | 2 | 1.0000 | 1 |
| unsupported_trap | 1.0000 | 4 | - | - |

## Worst Grounding Cases (F-beta < 0.5)

| QID (short) | Type | F-beta | P | R | Our Pages | Gold Pages | Match? |
|-------------|------|--------|---|---|-----------|------------|--------|
| d204a130 | number | 0.0000 | 0.00 | 0.00 | 1 | 1 | False |
| fcabd6aa | free_text | 0.0000 | 0.00 | 0.00 | 3 | 19 | None |
| 571f6013 | free_text | 0.0000 | 0.00 | 0.00 | 2 | 2 | None |
| 4aa0f4e2 | free_text | 0.0000 | 0.00 | 0.00 | 8 | 20 | None |
| bd8d0bef | boolean | 0.0000 | 0.00 | 0.00 | 24 | 2 | False |
| bb67fc19 | boolean | 0.0000 | 0.00 | 0.00 | 18 | 2 | False |
| 54d56331 | boolean | 0.0000 | 0.00 | 0.00 | 2 | 2 | True |
| 3c19ecbe | boolean | 0.0000 | 0.00 | 0.00 | 2 | 2 | True |
| acd3200d | free_text | 0.0000 | 0.00 | 0.00 | 1 | 3 | None |
| b4d8c1cc | free_text | 0.0000 | 0.00 | 0.00 | 2 | 2 | None |
| 5d8fd833 | free_text | 0.0000 | 0.00 | 0.00 | 3 | 2 | None |
| 115a9bca | free_text | 0.0000 | 0.00 | 0.00 | 3 | 8 | None |
| 2180c758 | free_text | 0.0000 | 0.00 | 0.00 | 2 | 3 | None |
| 1107e284 | free_text | 0.0000 | 0.00 | 0.00 | 1 | 2 | None |
| 8e3b4683 | boolean | 0.0000 | 0.00 | 0.00 | 2 | 2 | True |
| 6e8d0c41 | free_text | 0.0000 | 0.00 | 0.00 | 8 | 20 | None |
| b9dc2dae | name | 0.0000 | 0.00 | 0.00 | 1 | 2 | True |
| 0f6e75bd | name | 0.0000 | 0.00 | 0.00 | 1 | 2 | True |
| d9d27c9c | name | 0.0000 | 0.00 | 0.00 | 2 | 2 | True |
| fbe661b9 | name | 0.0000 | 0.00 | 0.00 | 1 | 2 | True |

## Answer Mismatches

| QID (short) | Type | Golden | Ours |
|-------------|------|--------|------|
| d204a130 | number | 405351504 | 250499.26 |
| bd8d0bef | boolean | True | null |
| bb67fc19 | boolean | False | null |
| d5bc7441 | boolean | False | null |
| 82664b58 | name | Employment Law Amendment Law DIFC Law No. 4 of 2021 | The Law of Damages and Remedies 2005 |
| 4ced374a | boolean | False | Yes |
| f378457d | number | 2 | 1 |
| 6976d6d2 | boolean | False | Yes |
| 75bf397c | boolean | True | No |
| f2ea23e9 | number | 3 | 6 |
| cd0c8f36 | name | gross and net remuneration | gross and net |
| 30ab0e56 | boolean | False | null |
| af8d4690 | boolean | True | null |
| 47cb314a | boolean | False | null |
| b249b41b | boolean | False | null |
| 3dc92e33 | name | SCT 169/2025 | CFI 010/2024 |
