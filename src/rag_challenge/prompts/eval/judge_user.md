I will provide:
- question
- answer_type (boolean/number/name/names/date/free_text)
- answer (system output)
- used_pages: list of `pdf_id_page` (assume 1-based)
- sources_text: concatenated text extracted from those used pages

Tasks:
1) Strict-type format validation:
- boolean: answer must be exactly "Yes" or "No"
- number: answer must be exactly one numeric value (digits with optional decimal)
- date: answer must be exactly one date in YYYY-MM-DD format
- name: answer must be exactly one name/title, not a sentence
- names: answer must be a list of names (comma-separated ok), not a sentence

2) Grounding:
- Every factual/legal claim must be supported by sources_text.
- If sources_text does not support the premise or answer, the correct behavior is "There is no information on this question." (for free_text). For strict types, accept `null`.

3) Out-of-jurisdiction premise:
If the question assumes US concepts (jury, Miranda rights, parole, plea bargain) and sources_text does not explicitly mention/define them, treat any direct answer as hallucination.

4) Efficiency:
Answer should be as short as possible while correct.

Now evaluate:
question={question}
answer_type={answer_type}
answer={answer}
used_pages={used_pages}
sources_text={sources_text}
