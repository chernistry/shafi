# Golden Labels Scoring Report

- **Matched questions**: 100 / 100
- **Unmatched**: 0
- **Overall Grounding F-beta (beta=2.5)**: 0.3825
- **Weighted Grounding F-beta**: 0.4044 (high=1.0, medium=0.5, low=0.25)
- **Trusted-only Grounding F-beta**: 0.4251 (high confidence only)
- **Exact Match Rate (deterministic types)**: 0.7429 (52/70)
- **Trusted Exact Match Rate**: 0.7273
- **Trusted Regressions**: 18

## By Answer Type

| Type | F-beta (mean) | F-beta (n) | Exact Match | Exact (n) |
|------|--------------|------------|-------------|-----------|
| boolean | 0.3692 | 35 | 0.7429 | 35 |
| date | 1.0000 | 1 | 1.0000 | 1 |
| free_text | 0.4142 | 30 | - | - |
| name | 0.3214 | 14 | 0.7857 | 14 |
| names | 0.6667 | 3 | 1.0000 | 3 |
| number | 0.3175 | 17 | 0.6471 | 17 |

## By Confidence Tier

| Confidence | F-beta (mean) | F-beta (n) | Exact Match | Exact (n) |
|------------|--------------|------------|-------------|-----------|
| high | 0.4251 | 81 | 0.7273 | 66 |
| low | 0.2000 | 5 | 1.0000 | 1 |
| medium | 0.2010 | 14 | 1.0000 | 3 |

## By Question Family

| Family | F-beta (mean) | F-beta (n) | Exact Match | Exact (n) |
|--------|--------------|------------|-------------|-----------|
| administration | 0.3290 | 4 | 1.0000 | 1 |
| boolean | 0.3414 | 11 | 0.8182 | 11 |
| compare | 0.4889 | 30 | 0.8077 | 26 |
| enactment | 0.2243 | 13 | 0.3333 | 6 |
| free_text_other | 0.2869 | 11 | - | - |
| names | 0.2917 | 12 | 0.8333 | 12 |
| number | 0.2613 | 13 | 0.6154 | 13 |
| outcome_costs | 0.7685 | 2 | 1.0000 | 1 |
| unsupported_trap | 1.0000 | 4 | - | - |

## Worst Grounding Cases (F-beta < 0.5)

| QID (short) | Type | F-beta | P | R | Our Pages | Gold Pages | Match? |
|-------------|------|--------|---|---|-----------|------------|--------|
| d204a130 | number | 0.0000 | 0.00 | 0.00 | 1 | 1 | False |
| 6f9c0b19 | names | 0.0000 | 0.00 | 0.00 | 1 | 3 | True |
| fcabd6aa | free_text | 0.0000 | 0.00 | 0.00 | 4 | 19 | None |
| 89fd4fbc | free_text | 0.0000 | 0.00 | 0.00 | 7 | 17 | None |
| 17022172 | free_text | 0.0000 | 0.00 | 0.00 | 1 | 2 | None |
| 571f6013 | free_text | 0.0000 | 0.00 | 0.00 | 2 | 2 | None |
| 4aa0f4e2 | free_text | 0.0000 | 0.00 | 0.00 | 7 | 20 | None |
| bd8d0bef | boolean | 0.0000 | 0.00 | 0.00 | 2 | 2 | False |
| bb67fc19 | boolean | 0.0000 | 0.00 | 0.00 | 21 | 2 | False |
| 54d56331 | boolean | 0.0000 | 0.00 | 0.00 | 2 | 2 | True |
| 3c19ecbe | boolean | 0.0000 | 0.00 | 0.00 | 2 | 2 | True |
| acd3200d | free_text | 0.0000 | 0.00 | 0.00 | 0 | 3 | None |
| b4d8c1cc | free_text | 0.0000 | 0.00 | 0.00 | 0 | 2 | None |
| 5d8fd833 | free_text | 0.0000 | 0.00 | 0.00 | 1 | 2 | None |
| 115a9bca | free_text | 0.0000 | 0.00 | 0.00 | 2 | 8 | None |
| 2180c758 | free_text | 0.0000 | 0.00 | 0.00 | 1 | 3 | None |
| 8e3b4683 | boolean | 0.0000 | 0.00 | 0.00 | 2 | 2 | True |
| 6e8d0c41 | free_text | 0.0000 | 0.00 | 0.00 | 9 | 20 | None |
| b9dc2dae | name | 0.0000 | 0.00 | 0.00 | 1 | 2 | True |
| 0f6e75bd | name | 0.0000 | 0.00 | 0.00 | 1 | 2 | True |

## Answer Mismatches

| QID (short) | Type | Golden | Ours |
|-------------|------|--------|------|
| d204a130 | number | 405351504 | 550000 |
| bd8d0bef | boolean | True | No |
| bb67fc19 | boolean | False | null |
| d5bc7441 | boolean | False | null |
| 4cbb1883 | number | 2021 | 5 |
| 82664b58 | name | Employment Law Amendment Law DIFC Law No. 4 of 2021 | The Law of Damages and Remedies 2005 |
| 4ced374a | boolean | False | Yes |
| f0329296 | number | 3 | 1 |
| f378457d | number | 2 | 5 |
| 6976d6d2 | boolean | False | Yes |
| 75bf397c | boolean | True | No |
| f2ea23e9 | number | 3 | 6 |
| cd0c8f36 | name | gross and net remuneration | gross and net |
| af8d4690 | boolean | True | null |
| 47cb314a | boolean | False | null |
| 7700103c | number | 4 | null |
| b249b41b | boolean | False | null |
| 3dc92e33 | name | SCT 169/2025 | CFI 010/2024 |
