# Parser Fix Analysis

## Problem

The original prediction parser in `LocalLLMClassifier` used `re.search(r'\b[01]\b', response)` which finds the **first** occurrence of 0 or 1 in the response text.

This caused incorrect predictions because reasoning models typically:
1. Discuss both possibilities during reasoning (mentioning both 0 and 1)
2. Provide the final answer at the **end** of their reasoning

### Example of the Bug

**Reasoning text:**
```
Gluon jets tend to have higher multiplicity (0). But this jet has few
particles with high pt. Likely a quark jet. I'd answer 1. 1
```

- **Old parser**: Found first "0" → predicted gluon (WRONG)
- **Correct answer**: Last "1" → quark (CORRECT)

## Solution

Implemented a multi-strategy parser that prioritizes the final answer:

### Strategy 1: Smart Pattern Matching
Look for common answer patterns near the end:
- `answer: 1`, `label: 0`, `prediction: 1`
- `0` or `1` at the very end
- After conclusion words: `therefore`, `thus`, `hence`

### Strategy 2: Last Occurrence
Find ALL occurrences of 0 or 1, return the **last** one

### Strategy 3: Keyword Fallback
Check for "quark" (→1) or "gluon" (→0) if no numeric answer found

## Results

### Test Cases Performance

| Parser | Correct | Accuracy |
|--------|---------|----------|
| Old (first occurrence) | 1/4 | 25% |
| New (last occurrence) | 4/4 | 100% |

### Real-World Impact

On actual jet classification tests:
- Async mode accuracy improved
- Predictions now match the model's actual conclusions
- More reliable extraction of final answers

## Code Changes

```python
# OLD: Takes first occurrence
match = re.search(r'\b[01]\b', response)
if match:
    return int(match.group())

# NEW: Takes last occurrence with smart patterns
end_patterns = [
    r'(?:answer|label|prediction|output|result)[:=\s]+([01])\b',
    r'\b([01])\s*$',
    r'(?:thus|therefore|so|hence).*?([01])\b',
]

for pattern in end_patterns:
    matches = list(re.finditer(pattern, response, re.IGNORECASE))
    if matches:
        return int(matches[-1].group(1))  # LAST match

# Fallback: All occurrences, take last
all_matches = list(re.finditer(r'\b[01]\b', response))
if all_matches:
    return int(all_matches[-1].group())
```

## Testing

Run the demonstration script:
```bash
uv run python scripts/test_parser_fix.py
```

This shows 4 example cases where the old parser fails and the new parser succeeds.

## Lessons Learned

1. **Reasoning models think step-by-step**: They explore multiple possibilities before concluding
2. **The final answer matters most**: Look at the end of the response, not the beginning
3. **Pattern matching helps**: Keywords like "answer:" or "therefore" are strong signals
4. **Always test edge cases**: The bug only showed up with verbose reasoning output

## Future Improvements

Potential enhancements:
1. Look for explicit formatting like "Final answer: X"
2. Use the model's summary if available
3. Implement confidence scoring based on answer position
4. Add validation (reject if multiple different answers at the end)
