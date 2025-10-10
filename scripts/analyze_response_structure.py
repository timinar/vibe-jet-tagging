"""Analyze the response structure from local OpenAI-compatible API.

This script explores the response object in detail to understand:
- How to access reasoning traces
- How to get token counts for prompt, reasoning, and output
- The complete structure of the response
"""

import json
from pathlib import Path

import numpy as np
from openai import OpenAI


def pretty_print_object(obj, name="Object", indent=0):
    """Recursively print object structure."""
    prefix = "  " * indent
    print(f"{prefix}{name}:")

    # Print type
    print(f"{prefix}  Type: {type(obj).__name__}")

    # If it has attributes, print them
    if hasattr(obj, '__dict__'):
        print(f"{prefix}  Attributes:")
        for attr_name in dir(obj):
            if not attr_name.startswith('_'):
                try:
                    attr_value = getattr(obj, attr_name)
                    if not callable(attr_value):
                        attr_type = type(attr_value).__name__
                        if isinstance(attr_value, (str, int, float, bool)):
                            print(f"{prefix}    {attr_name} ({attr_type}): {attr_value}")
                        elif isinstance(attr_value, list):
                            print(f"{prefix}    {attr_name} (list[{len(attr_value)}])")
                        else:
                            print(f"{prefix}    {attr_name} ({attr_type})")
                except Exception as e:
                    print(f"{prefix}    {attr_name}: Error - {e}")

    # If it's a dict, print keys
    elif isinstance(obj, dict):
        print(f"{prefix}  Keys: {list(obj.keys())}")


def analyze_response_detailed(response):
    """Analyze response object in detail."""
    print("=" * 80)
    print("DETAILED RESPONSE ANALYSIS")
    print("=" * 80)
    print()

    # Top level
    pretty_print_object(response, "Response")
    print()

    # Output structure
    if hasattr(response, 'output'):
        print("-" * 80)
        print("OUTPUT STRUCTURE:")
        print("-" * 80)
        output = response.output
        print(f"Output type: {type(output)}")
        print(f"Output length: {len(output) if isinstance(output, list) else 'N/A'}")
        print()

        if isinstance(output, list):
            for i, item in enumerate(output):
                print(f"\nOutput[{i}]:")
                pretty_print_object(item, f"Item {i}", indent=1)

                # Drill down into content
                if hasattr(item, 'content'):
                    print(f"\n  Content structure:")
                    content = item.content
                    print(f"    Type: {type(content)}")
                    if isinstance(content, list):
                        print(f"    Length: {len(content)}")
                        for j, content_item in enumerate(content):
                            print(f"\n    Content[{j}]:")
                            pretty_print_object(content_item, f"Content Item {j}", indent=2)

                            # Get the actual text
                            if hasattr(content_item, 'text'):
                                text = content_item.text
                                print(f"      Text length: {len(text)} characters")
                                print(f"      Text preview: {text[:200]}...")

    # Usage/Token information
    print()
    print("-" * 80)
    print("TOKEN USAGE INFORMATION:")
    print("-" * 80)
    if hasattr(response, 'usage'):
        usage = response.usage
        pretty_print_object(usage, "Usage", indent=0)
        print()

        # Extract specific token counts
        token_info = {}
        for attr in ['prompt_tokens', 'completion_tokens', 'reasoning_tokens',
                     'total_tokens', 'input_tokens', 'output_tokens']:
            if hasattr(usage, attr):
                token_info[attr] = getattr(usage, attr)

        print("Token Counts Summary:")
        for key, value in token_info.items():
            print(f"  {key}: {value:,}")
    else:
        print("  No usage information available")

    print()

    # Additional metadata
    print("-" * 80)
    print("ADDITIONAL METADATA:")
    print("-" * 80)
    for attr in ['id', 'model', 'created', 'object']:
        if hasattr(response, attr):
            print(f"  {attr}: {getattr(response, attr)}")


def extract_reasoning_trace(response):
    """Extract the reasoning trace from response."""
    print("=" * 80)
    print("REASONING TRACE EXTRACTION")
    print("=" * 80)
    print()

    if not hasattr(response, 'output'):
        print("No 'output' attribute found")
        return None

    output = response.output
    if not isinstance(output, list) or len(output) == 0:
        print("Output is not a list or is empty")
        return None

    # Look for reasoning items
    reasoning_texts = []
    output_texts = []

    for i, item in enumerate(output):
        print(f"Item {i}:")
        print(f"  Type: {type(item)}")

        if hasattr(item, 'type'):
            print(f"  Item type: {item.type}")

        if hasattr(item, 'content') and isinstance(item.content, list):
            for j, content_item in enumerate(item.content):
                if hasattr(content_item, 'text'):
                    text = content_item.text
                    text_type = getattr(content_item, 'type', 'unknown')
                    print(f"    Content[{j}] type: {text_type}")
                    print(f"    Text length: {len(text)} chars")

                    if text_type == 'reasoning_text' or (hasattr(item, 'type') and item.type == 'reasoning'):
                        reasoning_texts.append(text)
                        print(f"    → This is REASONING")
                    else:
                        output_texts.append(text)
                        print(f"    → This is OUTPUT")

    print()
    print("Summary:")
    print(f"  Reasoning texts found: {len(reasoning_texts)}")
    print(f"  Output texts found: {len(output_texts)}")

    return {
        'reasoning': reasoning_texts,
        'output': output_texts
    }


def get_token_counts(response):
    """Extract all available token counts from response."""
    print("=" * 80)
    print("TOKEN COUNTS EXTRACTION")
    print("=" * 80)
    print()

    if not hasattr(response, 'usage'):
        print("No 'usage' attribute found")
        return {}

    usage = response.usage
    counts = {}

    # Try to get various token count attributes
    possible_attrs = [
        'prompt_tokens',
        'completion_tokens',
        'reasoning_tokens',
        'total_tokens',
        'input_tokens',
        'output_tokens',
        'thoughts_token_count',
        'prompt_token_count',
        'candidates_token_count',
    ]

    for attr in possible_attrs:
        if hasattr(usage, attr):
            value = getattr(usage, attr)
            counts[attr] = value
            print(f"✓ {attr}: {value:,}")
        else:
            print(f"✗ {attr}: not available")

    print()
    print("Categorized:")
    if 'prompt_tokens' in counts or 'input_tokens' in counts:
        prompt = counts.get('prompt_tokens') or counts.get('input_tokens', 0)
        print(f"  Prompt/Input:  {prompt:,} tokens")

    if 'reasoning_tokens' in counts:
        print(f"  Reasoning:     {counts['reasoning_tokens']:,} tokens")

    if 'completion_tokens' in counts or 'output_tokens' in counts:
        completion = counts.get('completion_tokens') or counts.get('output_tokens', 0)
        print(f"  Completion:    {completion:,} tokens")

    if 'total_tokens' in counts:
        print(f"  Total:         {counts['total_tokens']:,} tokens")

    return counts


def main():
    """Main analysis function."""
    print("Starting OpenAI API Response Analysis")
    print("=" * 80)
    print()

    # Load data
    data_path = Path(__file__).parent.parent / 'data' / 'qg_jets.npz'
    if not data_path.exists():
        print(f"Error: Data not found at {data_path}")
        return

    data = np.load(data_path)
    X = data['X']

    # Simple test prompt
    from vibe_jet_tagging.utils.formatters import fill_template, load_template

    template_dir = Path(__file__).parent.parent / 'templates'
    template = load_template('simple_list', str(template_dir))
    prompt = fill_template(template, X[0], 'list')

    print("Creating client and making request...")
    print(f"Prompt length: {len(prompt)} characters")
    print()

    # Create client
    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="EMPTY"
    )

    # Make request with medium reasoning
    print("Sending request to local API...")
    try:
        response = client.responses.create(
            model="openai/gpt-oss-120b",
            instructions="You are a helpful assistant.",
            input=prompt,
            reasoning={
                "effort": "medium",
                "summary": "auto"
            }
        )
        print("✓ Request successful!")
        print()

        # Analyze the response
        analyze_response_detailed(response)
        print()

        # Extract reasoning trace
        traces = extract_reasoning_trace(response)
        print()

        # Get token counts
        token_counts = get_token_counts(response)
        print()

        # Save example response structure for reference
        print("=" * 80)
        print("SAVING EXAMPLE DATA")
        print("=" * 80)

        # Try to convert to dict for JSON serialization
        try:
            if traces:
                output_path = Path(__file__).parent / 'response_example.txt'
                with open(output_path, 'w') as f:
                    f.write("REASONING TRACE:\n")
                    f.write("=" * 80 + "\n")
                    for i, text in enumerate(traces['reasoning']):
                        f.write(f"\nReasoning {i+1}:\n")
                        f.write(text)
                        f.write("\n" + "-" * 80 + "\n")

                    f.write("\n\nOUTPUT:\n")
                    f.write("=" * 80 + "\n")
                    for i, text in enumerate(traces['output']):
                        f.write(f"\nOutput {i+1}:\n")
                        f.write(text)
                        f.write("\n" + "-" * 80 + "\n")

                    f.write("\n\nTOKEN COUNTS:\n")
                    f.write("=" * 80 + "\n")
                    for key, value in token_counts.items():
                        f.write(f"{key}: {value:,}\n")

                print(f"✓ Saved example to: {output_path}")
        except Exception as e:
            print(f"Could not save example: {e}")

    except Exception as e:
        print(f"✗ Request failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
