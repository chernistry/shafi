#!/usr/bin/env python3
"""Build V16 hybrid: V16 free_text answers + V15 pages + all corrections.

Run after V16 eval completes (900/900 in checkpoint).
"""
import json, re, os

# Load sources
v16_cp = [json.loads(l) for l in open('data/tzuf_private1_checkpoint.jsonl') if l.strip()]
v16_by_id = {}
for l in v16_cp:
    v16_by_id[l['id']] = l

v15 = json.loads(open('data/private_submission_V15_ULTIMATE_FINAL.json').read())
qs = json.loads(open('dataset/private/questions.json').read())
q_map = {q['id']: q for q in qs}

# Start from V15_ULTIMATE_FINAL (has best pages + corrections)
hybrid = json.loads(open('data/private_submission_V15_ULTIMATE_FINAL.json').read())

# Upgrade free_text answers where V16 used gpt-4.1 and has better content
upgraded = 0
for ans in hybrid['answers']:
    qid = ans['question_id']
    q = q_map[qid]
    v16 = v16_by_id.get(qid)
    
    if not v16 or q['answer_type'] != 'free_text':
        continue
    if v16.get('model') != 'gpt-4.1-2025-04-14':
        continue
    
    v16_ans = str(v16.get('answer', ''))
    v15_ans = str(ans['answer'] or '')
    
    # Skip if V16 is noinfo
    if 'no information' in v16_ans.lower():
        continue
    # Skip if V15 already has our manual correction
    if v15_ans and 'no information' not in v15_ans.lower() and len(v15_ans) > 100:
        continue
    
    # Strip cite markers from V16
    v16_clean = re.sub(r'\s*\(cite:[^)]*\)', '', v16_ans).strip()
    if v16_clean and v16_clean[-1] not in '.!?)':
        v16_clean = v16_clean.rstrip(',:;') + '.'
    
    # Fix newlines
    v16_clean = re.sub(r'\n+', '; ', v16_clean)
    
    # Truncate to 280
    if len(v16_clean) > 280:
        sentences = re.split(r'(?<=[.!?])\s+', v16_clean)
        truncated = ''
        for s in sentences:
            c = truncated + ' ' + s if truncated else s
            if len(c) <= 280:
                truncated = c
            else:
                break
        v16_clean = truncated if truncated else v16_clean[:277] + '...'
    
    if len(v16_clean) > len(v15_ans) + 20:
        ans['answer'] = v16_clean
        upgraded += 1

# Verify
nulls = sum(1 for x in hybrid['answers'] if x['answer'] is None)
nopg = sum(1 for x in hybrid['answers'] if x['telemetry']['retrieval']['retrieved_chunk_pages'] == [])
noinfo = sum(1 for x in hybrid['answers'] if isinstance(x['answer'], str) and 'no information' in x['answer'].lower())
over280 = sum(1 for x in hybrid['answers'] if isinstance(x['answer'], str) and len(x['answer']) > 280)

output = 'data/private_submission_V16_HYBRID.json'
s = json.dumps(hybrid, separators=(',', ':'), ensure_ascii=False)
with open(output, 'w') as f:
    f.write(s)
    f.flush()
    os.fsync(f.fileno())

print(f'V16 HYBRID: upgraded={upgraded} free_text answers')
print(f'null={nulls}, nopg={nopg}, noinfo={noinfo}, over280={over280}')
print(f'File: {output} ({os.path.getsize(output)/1024:.1f}KB)')
