# Golden Labels Scoring Report

- **Matched questions**: 100 / 100
- **Unmatched**: 0
- **Overall Grounding F-beta (beta=2.5)**: 0.4362
- **Weighted Grounding F-beta**: 0.4583 (high=1.0, medium=0.5, low=0.25)
- **Trusted-only Grounding F-beta**: 0.4776 (high confidence only)
- **Exact Match Rate (deterministic types)**: 0.7429 (52/70)
- **Trusted Exact Match Rate**: 0.7273
- **Trusted Regressions**: 18

## By Answer Type

| Type | F-beta (mean) | F-beta (n) | Exact Match | Exact (n) |
|------|--------------|------------|-------------|-----------|
| boolean | 0.3782 | 35 | 0.6857 | 35 |
| date | 1.0000 | 1 | 1.0000 | 1 |
| free_text | 0.4675 | 30 | - | - |
| name | 0.4365 | 14 | 0.7143 | 14 |
| names | 0.7890 | 3 | 1.0000 | 3 |
| number | 0.4046 | 17 | 0.8235 | 17 |

## By Confidence Tier

| Confidence | F-beta (mean) | F-beta (n) | Exact Match | Exact (n) |
|------------|--------------|------------|-------------|-----------|
| high | 0.4776 | 81 | 0.7273 | 66 |
| low | 0.2000 | 5 | 1.0000 | 1 |
| medium | 0.2808 | 14 | 1.0000 | 3 |

## By Question Family

| Family | F-beta (mean) | F-beta (n) | Exact Match | Exact (n) |
|--------|--------------|------------|-------------|-----------|
| administration | 0.6003 | 4 | 0.0000 | 1 |
| boolean | 0.3414 | 11 | 0.7273 | 11 |
| compare | 0.3402 | 30 | 0.7692 | 26 |
| enactment | 0.3797 | 13 | 0.5000 | 6 |
| free_text_other | 0.4025 | 11 | - | - |
| names | 0.7065 | 12 | 0.8333 | 12 |
| number | 0.2984 | 13 | 0.7692 | 13 |
| outcome_costs | 0.7685 | 2 | 1.0000 | 1 |
| unsupported_trap | 1.0000 | 4 | - | - |

## Worst Grounding Cases (F-beta < 0.5)

| QID (short) | Type | F-beta | P | R | Our Pages | Gold Pages | Match? |
|-------------|------|--------|---|---|-----------|------------|--------|
| d204a130 | number | 0.0000 | 0.00 | 0.00 | 2 | 1 | False |
| 571f6013 | free_text | 0.0000 | 0.00 | 0.00 | 2 | 2 | None |
| 4aa0f4e2 | free_text | 0.0000 | 0.00 | 0.00 | 8 | 20 | None |
| bd8d0bef | boolean | 0.0000 | 0.00 | 0.00 | 24 | 2 | False |
| bb67fc19 | boolean | 0.0000 | 0.00 | 0.00 | 18 | 2 | False |
| 54d56331 | boolean | 0.0000 | 0.00 | 0.00 | 2 | 2 | True |
| 3c19ecbe | boolean | 0.0000 | 0.00 | 0.00 | 2 | 2 | True |
| acd3200d | free_text | 0.0000 | 0.00 | 0.00 | 3 | 3 | None |
| b4d8c1cc | free_text | 0.0000 | 0.00 | 0.00 | 2 | 2 | None |
| 5d8fd833 | free_text | 0.0000 | 0.00 | 0.00 | 3 | 2 | None |
| 115a9bca | free_text | 0.0000 | 0.00 | 0.00 | 3 | 8 | None |
| 2180c758 | free_text | 0.0000 | 0.00 | 0.00 | 2 | 3 | None |
| 1107e284 | free_text | 0.0000 | 0.00 | 0.00 | 1 | 2 | None |
| 8e3b4683 | boolean | 0.0000 | 0.00 | 0.00 | 2 | 2 | True |
| 6e8d0c41 | free_text | 0.0000 | 0.00 | 0.00 | 8 | 20 | None |
| d5bc7441 | boolean | 0.0000 | 0.00 | 0.00 | 9 | 2 | False |
| 82664b58 | name | 0.0000 | 0.00 | 0.00 | 1 | 1 | False |
| 4ced374a | boolean | 0.0000 | 0.00 | 0.00 | 2 | 1 | False |
| b52c749f | boolean | 0.0000 | 0.00 | 0.00 | 2 | 1 | True |
| be535a44 | number | 0.0000 | 0.00 | 0.00 | 1 | 1 | True |

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
| 06034335 | name | Body Corporate | a Body Corporate |
| f2ea23e9 | number | 3 | 6 |
| 30ab0e56 | boolean | False | null |
| af8d4690 | boolean | True | null |
| 47cb314a | boolean | False | null |
| df0f24b2 | boolean | False | Yes |
| b249b41b | boolean | False | null |
| 3dc92e33 | name | SCT 169/2025 | CFI 010/2024 |
| cd0c8f36 | name | gross and net remuneration | gross and net |
