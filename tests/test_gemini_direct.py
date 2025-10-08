"""
Test Gemini directly using Google's SDK to understand thinking_budget behavior.

This tests:
1. Basic Gemini API call works
2. Different thinking_budget values affect token usage
3. Complex prompts use more reasoning tokens
"""

import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Configure API
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("ERROR: GEMINI_API_KEY not found in environment")
    print("Please set it in .env file: GEMINI_API_KEY='your-key-here'")
    exit(1)

client = genai.Client(api_key=api_key)

print("="*80)
print("TESTING GEMINI 2.5 FLASH LITE DIRECTLY")
print("="*80)

# Test 1: Basic call with no thinking_budget
print("\n" + "="*80)
print("TEST 1: No thinking_budget specified")
print("="*80)

response = client.models.generate_content(
    model="gemini-2.5-flash-lite-preview-09-2025",
    contents="What is 2+2?"
)

print(f"Response: {response.text}")
print(f"Usage metadata: {response.usage_metadata}")

# Test 2: Simple prompt with different thinking_budgets
print("\n" + "="*80)
print("TEST 2: Simple prompt with different thinking_budgets")
print("="*80)

simple_prompt = "Is 100 greater than 50? Answer yes or no."

# Note: Flash-Lite has a MINIMUM of 512 tokens per the docs!
for budget in [0, 512, 1000, 5000, 10000]:
    print(f"\nThinking budget: {budget}")
    print("-"*80)
    
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite-preview-09-2025",
            contents=simple_prompt,
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=budget)
            )
        )
        
        usage = response.usage_metadata
        print(f"Total tokens: {usage.total_token_count}")
        print(f"Prompt tokens: {usage.prompt_token_count}")
        print(f"Candidates tokens: {usage.candidates_token_count}")
        
        # Check for thinking tokens
        if hasattr(usage, 'thoughts_token_count'):
            print(f"Thinking tokens: {usage.thoughts_token_count}")
        
        print(f"Response: {response.text[:50]}...")
        
    except Exception as e:
        print(f"ERROR: {e}")

# Test 3: Complex prompt requiring reasoning
print("\n" + "="*80)
print("TEST 3: Complex prompt with different thinking_budgets")
print("="*80)

complex_prompt = """
You are a physics expert. Solve this step-by-step:

A ball is thrown upward at 20 m/s from a height of 10 meters.
Calculate:
1. The maximum height reached
2. The time to reach maximum height
3. The total time until it hits the ground
4. The velocity when it hits the ground

Use g = 9.8 m/s². Show your work for each step.
"""

# Test with valid range for Flash-Lite: 512 to 24576
for budget in [512, 1000, 5000, 15000, 24000]:
    print(f"\nThinking budget: {budget}")
    print("-"*80)
    
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite-preview-09-2025",
            contents=complex_prompt,
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=budget),
                max_output_tokens=3000
            )
        )
        
        usage = response.usage_metadata
        print(f"Total tokens: {usage.total_token_count}")
        print(f"Prompt tokens: {usage.prompt_token_count}")
        print(f"Candidates tokens: {usage.candidates_token_count}")
        
        # Check for thinking tokens
        if hasattr(usage, 'thoughts_token_count'):
            print(f"Thinking tokens: {usage.thoughts_token_count}")
        
        response_preview = response.text[:150].replace('\n', ' ')
        print(f"Response preview: {response_preview}...")
        
    except Exception as e:
        print(f"ERROR: {e}")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print("If thinking_budget is respected:")
print("  - Token usage should vary with different budgets")
print("  - Higher budgets should allow more detailed responses")
print("  - Lower budgets should produce shorter/simpler responses")
print("\nIf thinking_budget is NOT respected:")
print("  - Token usage stays roughly the same regardless of budget")
print("  - Response quality/length doesn't change with budget")
print("="*80)

