# OpenAI Responses API Structure Guide

This guide documents the response structure from the OpenAI-compatible Responses API used by reasoning models.

## Response Structure

### Top-Level Response Object

```python
response = client.responses.create(
    model="openai/gpt-oss-120b",
    instructions="You are a helpful assistant.",
    input=prompt,
    reasoning={"effort": "medium", "summary": "auto"}
)
```

**Key Attributes:**
- `response.id` - Unique response identifier
- `response.model` - Model name used
- `response.status` - Completion status ("completed")
- `response.output` - List containing reasoning and output items
- `response.output_text` - Quick access to final output text
- `response.usage` - Token usage information

### Output Structure

The `response.output` is a **list** with 2 items:

#### 1. Reasoning Item (`output[0]`)

```python
reasoning_item = response.output[0]
# Type: ResponseReasoningItem
# reasoning_item.type == "reasoning"
# reasoning_item.content[0].type == "reasoning_text"
```

**Access reasoning trace:**
```python
reasoning_trace = response.output[0].content[0].text
```

**Structure:**
```python
{
    "type": "reasoning",
    "id": "rs_841bf066...",
    "content": [
        {
            "type": "reasoning_text",
            "text": "We need to classify jet as quark or gluon..."
        }
    ],
    "summary": []  # Empty unless summary requested
}
```

#### 2. Output Message (`output[1]`)

```python
output_message = response.output[1]
# Type: ResponseOutputMessage
# output_message.type == "message"
# output_message.content[0].type == "output_text"
```

**Access final output:**
```python
final_output = response.output[1].content[0].text
# Or use shortcut:
final_output = response.output_text
```

**Structure:**
```python
{
    "type": "message",
    "id": "msg_5597183a...",
    "role": "assistant",
    "status": "completed",
    "content": [
        {
            "type": "output_text",
            "text": "1",
            "annotations": []
        }
    ]
}
```

## Token Usage

### Token Count Attributes

The `response.usage` object provides:

```python
usage = response.usage

# Available attributes:
usage.input_tokens      # Input/prompt tokens: 695
usage.output_tokens     # Output tokens (reasoning + completion): 397
usage.total_tokens      # Total tokens: 1,092

# Detailed breakdowns (if available):
usage.input_tokens_details   # InputTokensDetails object
usage.output_tokens_details  # OutputTokensDetails object
```

### Calculating Reasoning vs Completion Tokens

**Note:** The API does **not** provide separate `reasoning_tokens` and `completion_tokens` counts directly.

The `output_tokens` includes **both**:
- Reasoning tokens (the internal thinking)
- Completion tokens (the final output)

**To estimate:**
```python
# Approximate based on character counts
reasoning_text = response.output[0].content[0].text
output_text = response.output[1].content[0].text

reasoning_chars = len(reasoning_text)
output_chars = len(output_text)
total_chars = reasoning_chars + output_chars

# Rough estimation (assuming ~4 chars per token on average)
estimated_reasoning_tokens = reasoning_chars // 4
estimated_output_tokens = output_chars // 4

# Or use the total and ratio:
output_tokens = response.usage.output_tokens
reasoning_ratio = reasoning_chars / total_chars
estimated_reasoning = int(output_tokens * reasoning_ratio)
estimated_completion = output_tokens - estimated_reasoning
```

### Token Details Objects

```python
# Input details
input_details = response.usage.input_tokens_details
# May contain: cached_tokens, text_tokens, etc.

# Output details
output_details = response.usage.output_tokens_details
# May contain: reasoning_tokens, text_tokens, etc.
```

## Code Examples

### Extract Everything

```python
# Reasoning trace
reasoning_trace = response.output[0].content[0].text

# Final output
final_output = response.output[1].content[0].text
# or: final_output = response.output_text

# Token counts
prompt_tokens = response.usage.input_tokens
output_tokens = response.usage.output_tokens
total_tokens = response.usage.total_tokens

print(f"Prompt: {prompt_tokens} tokens")
print(f"Output: {output_tokens} tokens (reasoning + completion)")
print(f"Total:  {total_tokens} tokens")
print(f"\nReasoning length: {len(reasoning_trace)} chars")
print(f"Output length:    {len(final_output)} chars")
```

### Verbose Logging

```python
def log_response_details(response, verbose=True):
    """Log detailed response information."""
    # Extract content
    reasoning = response.output[0].content[0].text
    output = response.output_text

    # Token counts
    usage = response.usage
    input_tok = usage.input_tokens
    output_tok = usage.output_tokens
    total_tok = usage.total_tokens

    if verbose:
        print(f"\n{'='*60}")
        print("RESPONSE DETAILS")
        print(f"{'='*60}")
        print(f"Response ID: {response.id}")
        print(f"Model:      {response.model}")
        print(f"Status:     {response.status}")
        print()

        print(f"{'─'*60}")
        print("TOKEN USAGE")
        print(f"{'─'*60}")
        print(f"Input:  {input_tok:6,} tokens")
        print(f"Output: {output_tok:6,} tokens")
        print(f"Total:  {total_tok:6,} tokens")
        print()

        print(f"{'─'*60}")
        print("REASONING TRACE")
        print(f"{'─'*60}")
        print(f"Length: {len(reasoning)} characters")
        print(f"Preview: {reasoning[:200]}...")
        print()

        print(f"{'─'*60}")
        print("FINAL OUTPUT")
        print(f"{'─'*60}")
        print(output)
        print(f"{'='*60}\n")
```

## Important Notes

1. **Reasoning Trace Access**: Always through `output[0].content[0].text`
2. **Final Output Access**: Use `output_text` shortcut or `output[1].content[0].text`
3. **Token Separation**: API doesn't split reasoning/completion tokens - must estimate
4. **Character Length**: Useful for rough token estimation (~4 chars/token)
5. **List Structure**: `output` is always a list with 2 items (reasoning, message)

## Migration from Gemini API

| Gemini API | OpenAI Responses API |
|------------|---------------------|
| `response.text` | `response.output_text` |
| `usage.thoughts_token_count` | Estimated from `output_tokens` + char length |
| `usage.prompt_token_count` | `usage.input_tokens` |
| `usage.candidates_token_count` | `usage.output_tokens` |
| `usage.total_token_count` | `usage.total_tokens` |

## Example Output

```
RESPONSE DETAILS
================================================================
Response ID: resp_fa18c682b61643e2b78bd39afeab8fbc
Model:      openai/gpt-oss-120b
Status:     completed

────────────────────────────────────────────────────────────
TOKEN USAGE
────────────────────────────────────────────────────────────
Input:     695 tokens
Output:    397 tokens
Total:   1,092 tokens

────────────────────────────────────────────────────────────
REASONING TRACE
────────────────────────────────────────────────────────────
Length: 1279 characters
Preview: We need to classify jet as quark or gluon based on
properties. Usually gluon jets have higher particle
multiplicity...

────────────────────────────────────────────────────────────
FINAL OUTPUT
────────────────────────────────────────────────────────────
1
================================================================
```
