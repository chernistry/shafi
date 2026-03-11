Question: {question}
Answer type: {answer_type}

Sources:
{formatted_context}

Output rules:
- Output ONLY the final {answer_type} value.
- Do NOT output explanations, citations, markdown, or extra words.
- For `number`: answer from the PRINCIPAL law matching the EXACT title in the question (not its amendment). "X Law Amendment Law" ≠ "X Law". If sources contain multiple law numbers, find the one whose title verbatim matches the question.
- If asked for a year: match the EXACT law title from the question. "Employment Law Amendment Law" (enacted 2021) ≠ "Employment Law" (enacted 2019).
- If asked for years in a date-number format like "six (6) years", output the digit in parentheses.
- For `name`/`names`, return only party/entity names; remove case ID prefixes.
- For `date`, return exactly one date in `YYYY-MM-DD` format.
- For `boolean`, output ONLY `Yes` or `No`.

Verification steps — do these BEFORE answering:
1. If the question references a SPECIFIC ARTICLE NUMBER (e.g., "Article 11(1)", "Article 28(4)"): verify that exact article heading appears in the sources. If not found, output: null.
2. If the question compares two entities (e.g., "Was X enacted the same year as Y?"): verify BOTH entities' relevant information appears in sources. If only one is present, output: null.
3. If the question asks for a law number or year: locate the enactment notice page for the EXACTLY named law. If missing, output: null.
4. Final check: is your answer directly stated in the sources? If you are guessing or inferring from parametric knowledge NOT in the sources, output: null.
