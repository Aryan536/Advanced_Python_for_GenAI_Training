import json
import re

# Real-world example: LLMs are asked to return JSON, but they do not always obey.
llm_outputs = [
    '{"sentiment": "positive", "score": 0.92}',
    'Sure! Here is the result:\n{"sentiment": "negative", "score": 0.15}\nHope this helps.',
    'I think the sentiment is neutral.',
    '{"sentiment": "positive", "score": 0.8',          # broken JSON (missing brace)
]

def parse_llm_json(text):
    """Try strict JSON first, then extract a {...} block, else return a safe default."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return {"sentiment": "unknown", "score": None}

results = []
for raw in llm_outputs:
    parsed = parse_llm_json(raw)
    results.append(parsed)
    print(f"{raw[:35]!r:40} -> {parsed}")

# try / except / else / finally
def safe_divide(a, b):
    try:
        result = a / b
    except ZeroDivisionError:
        print("Cannot divide by zero - returning 0")
        result = 0
    else:
        print("Division successful")
    finally:
        print("Done")
    return result

print("Conversion rate:", safe_divide(45, 900))
print("Conversion rate:", safe_divide(45, 0))

# Save and reload results as a JSON file
with open("llm_results.json", "w") as f:
    json.dump(results, f, indent=2)
with open("llm_results.json") as f:
    print("Reloaded from file:", json.load(f))
