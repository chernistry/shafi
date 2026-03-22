# Golden Labels Scoring Report

- **Matched questions**: 100 / 100
- **Unmatched**: 0
- **Overall Grounding F-beta (beta=2.5)**: 0.379
- **Weighted Grounding F-beta**: 0.4062 (high=1.0, medium=0.5, low=0.25)
- **Trusted-only Grounding F-beta**: 0.4326 (high confidence only)
- **Exact Match Rate (deterministic types)**: 0.7857 (55/70)
- **Trusted Exact Match Rate**: 0.7727
- **Trusted Regressions**: 15

## By Answer Type

| Type | F-beta (mean) | F-beta (n) | Exact Match | Exact (n) |
|------|--------------|------------|-------------|-----------|
| boolean | 0.2377 | 35 | 0.7143 | 35 |
| date | 1.0000 | 1 | 1.0000 | 1 |
| free_text | 0.4090 | 30 | - | - |
| name | 0.5463 | 14 | 0.8571 | 14 |
| names | 1.0000 | 3 | 1.0000 | 3 |
| number | 0.3331 | 17 | 0.8235 | 17 |

## By Confidence Tier

| Confidence | F-beta (mean) | F-beta (n) | Exact Match | Exact (n) |
|------------|--------------|------------|-------------|-----------|
| high | 0.4326 | 81 | 0.7727 | 66 |
| low | 0.1744 | 5 | 1.0000 | 1 |
| medium | 0.1419 | 14 | 1.0000 | 3 |

## By Question Family

| Family | F-beta (mean) | F-beta (n) | Exact Match | Exact (n) |
|--------|--------------|------------|-------------|-----------|
| administration | 0.4260 | 4 | 0.0000 | 1 |
| boolean | 0.2649 | 11 | 0.9091 | 11 |
| compare | 0.3558 | 30 | 0.7692 | 26 |
| enactment | 0.2283 | 13 | 0.5000 | 6 |
| free_text_other | 0.3253 | 11 | - | - |
| names | 0.7207 | 12 | 0.9167 | 12 |
| number | 0.2214 | 13 | 0.7692 | 13 |
| outcome_costs | 0.2685 | 2 | 1.0000 | 1 |
| unsupported_trap | 1.0000 | 4 | - | - |

## Worst Grounding Cases (F-beta < 0.5)

| QID (short) | Type | F-beta | P | R | Our Pages | Gold Pages | Match? |
|-------------|------|--------|---|---|-----------|------------|--------|
| d204a130 | number | 0.0000 | 0.00 | 0.00 | 2 | 1 | False |
| eeae1069 | free_text | 0.0000 | 0.00 | 0.00 | 1 | 2 | None |
| fcabd6aa | free_text | 0.0000 | 0.00 | 0.00 | 1 | 19 | None |
| 17022172 | free_text | 0.0000 | 0.00 | 0.00 | 1 | 2 | None |
| d9c08834 | free_text | 0.0000 | 0.00 | 0.00 | 1 | 1 | None |
| 4aa0f4e2 | free_text | 0.0000 | 0.00 | 0.00 | 14 | 20 | None |
| bd8d0bef | boolean | 0.0000 | 0.00 | 0.00 | 24 | 2 | False |
| bb67fc19 | boolean | 0.0000 | 0.00 | 0.00 | 18 | 2 | False |
| bfa089d5 | boolean | 0.0000 | 0.00 | 0.00 | 2 | 2 | True |
| 52a35cfa | boolean | 0.0000 | 0.00 | 0.00 | 2 | 2 | True |
| fba6e86a | boolean | 0.0000 | 0.00 | 0.00 | 2 | 2 | True |
| 3c19ecbe | boolean | 0.0000 | 0.00 | 0.00 | 2 | 2 | True |
| b4d8c1cc | free_text | 0.0000 | 0.00 | 0.00 | 6 | 2 | None |
| 5d8fd833 | free_text | 0.0000 | 0.00 | 0.00 | 1 | 2 | None |
| 2180c758 | free_text | 0.0000 | 0.00 | 0.00 | 8 | 3 | None |
| 8e3b4683 | boolean | 0.0000 | 0.00 | 0.00 | 2 | 2 | True |
| 6e8d0c41 | free_text | 0.0000 | 0.00 | 0.00 | 16 | 20 | None |
| 46927f37 | boolean | 0.0000 | 0.00 | 0.00 | 1 | 2 | True |
| b249b41b | boolean | 0.0000 | 0.00 | 0.00 | 1 | 2 | False |
| d5bc7441 | boolean | 0.0000 | 0.00 | 0.00 | 9 | 2 | False |

## Answer Mismatches

| QID (short) | Type | Golden | Ours |
|-------------|------|--------|------|
| d204a130 | number | 405351504 | 250499.26 |
| bd8d0bef | boolean | True | null |
| bb67fc19 | boolean | False | null |
| b249b41b | boolean | False | Yes |
| d5bc7441 | boolean | False | null |
| 82664b58 | name | Employment Law Amendment Law DIFC Law No. 4 of 2021 | The Law of Damages and Remedies 2005 |
| 4ced374a | boolean | False | Yes |
| 75bf397c | boolean | True | No |
| f2ea23e9 | number | 3 | 6 |
| 30ab0e56 | boolean | False | null |
| 96bccc8b | boolean | False | null |
| af8d4690 | boolean | True | null |
| 47cb314a | boolean | False | null |
| 3dc92e33 | name | SCT 169/2025 | CFI 010/2024 |
| f378457d | number | 2 | 1 |
