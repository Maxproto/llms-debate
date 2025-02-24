"""
experiments/demo_api_all_models.py

Demonstrates calling each model function in model_interface.py.
Run:
  python experiments/demo_api_all_models.py
to see if each model can produce a response.
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from debate_agent_lc_all import (
    call_gpt4o,
    call_o3_mini,
    call_claude_haiku,
    call_mistral_small,
    call_llama32,
    call_gemini2_flash
)

def main():
    # Simple prompt for testing
    prompt = "Give me a quick fun fact about space travel."

    print("=== Testing GPT-4o ===")
    resp_4o = call_gpt4o(prompt)
    print(resp_4o, "\n")

    print("=== Testing o3-mini ===")
    resp_o3 = call_o3_mini(prompt)
    print(resp_o3, "\n")

    print("=== Testing Claude 3.5 Haiku ===")
    resp_claude = call_claude_haiku(prompt)
    print(resp_claude, "\n")

    print("=== Testing Mistral Small 3 ===")
    resp_mistral = call_mistral_small(prompt)
    print(resp_mistral, "\n")

    print("=== Testing LLaMA 3.2-3B (local) ===")
    resp_llama = call_llama32(prompt)
    print(resp_llama, "\n")

    print("=== Testing Gemini 2.0 Flash ===")
    resp_gemini = call_gemini2_flash(prompt)
    print(resp_gemini, "\n")

if __name__ == "__main__":
    main()
