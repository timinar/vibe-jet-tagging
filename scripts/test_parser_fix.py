"""Demonstrate the parser fix for LocalLLMClassifier.

This script shows how the improved parser correctly extracts the final answer
from reasoning text that may contain multiple occurrences of 0 and 1.
"""

import re


def parse_old(response: str) -> int:
    """OLD parser: Takes the FIRST occurrence."""
    match = re.search(r'\b[01]\b', response)
    if match:
        return int(match.group())
    return 0


def parse_new(response: str) -> int:
    """NEW parser: Takes the LAST occurrence or uses smart patterns."""
    # Strategy 1: Look for common answer patterns at the end
    end_patterns = [
        r'(?:answer|label|prediction|output|result)[:=\s]+([01])\b',
        r'\b([01])\s*$',  # 0 or 1 at the very end
        r'(?:thus|therefore|so|hence).*?([01])\b',  # After conclusion words
    ]

    for pattern in end_patterns:
        matches = list(re.finditer(pattern, response, re.IGNORECASE))
        if matches:
            return int(matches[-1].group(1))

    # Strategy 2: Find ALL occurrences, take the last one
    all_matches = list(re.finditer(r'\b[01]\b', response))
    if all_matches:
        return int(all_matches[-1].group())

    return 0


def main():
    """Test cases showing the improvement."""
    # Example reasoning text from the model
    reasoning_examples = [
        {
            "text": "Gluon jets tend to have higher multiplicity (0). But this jet has few particles with high pt. Likely a quark jet. I'd answer 1. 1",
            "expected": 1,
            "description": "Model reasons about gluon (0) first, then concludes quark (1)"
        },
        {
            "text": "Many high-pt particles (1), many photons. Could be quark (1) or gluon (0). But overall many particles suggests gluon. Answer: 0",
            "expected": 0,
            "description": "Multiple mentions of 1, but final answer is 0"
        },
        {
            "text": "Hard to tell. Could be 0 or could be 1. Actually thinking about it more, the leading particle fraction suggests 1",
            "expected": 1,
            "description": "Uncertainty with both values, final conclusion is 1"
        },
        {
            "text": "The multiplicity is moderate (34 particles). Average pt ~16 GeV. This seems like quark. Therefore answer 1",
            "expected": 1,
            "description": "Mentions number 34 in reasoning, but answer is 1"
        }
    ]

    print("=" * 80)
    print("PARSER COMPARISON TEST")
    print("=" * 80)
    print()

    for i, example in enumerate(reasoning_examples, 1):
        text = example["text"]
        expected = example["expected"]
        description = example["description"]

        old_result = parse_old(text)
        new_result = parse_new(text)

        print(f"Test {i}: {description}")
        print(f"Expected: {expected}")
        print(f"Old parser: {old_result} {'✗ WRONG' if old_result != expected else '✓'}")
        print(f"New parser: {new_result} {'✗ WRONG' if new_result != expected else '✓ CORRECT'}")
        print()
        print(f"Text: {text}")
        print("-" * 80)
        print()

    # Summary
    old_correct = sum(1 for ex in reasoning_examples if parse_old(ex["text"]) == ex["expected"])
    new_correct = sum(1 for ex in reasoning_examples if parse_new(ex["text"]) == ex["expected"])

    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Old parser: {old_correct}/{len(reasoning_examples)} correct ({old_correct/len(reasoning_examples)*100:.0f}%)")
    print(f"New parser: {new_correct}/{len(reasoning_examples)} correct ({new_correct/len(reasoning_examples)*100:.0f}%)")
    print()
    print(f"Improvement: {new_correct - old_correct} additional correct predictions")


if __name__ == "__main__":
    main()
