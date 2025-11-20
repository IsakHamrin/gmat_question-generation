import os
import json
from dotenv import load_dotenv
from google import genai

# -------------------------------------------------------------------
# Load API key
# -------------------------------------------------------------------
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY is not set. Check your .env file.")

client = genai.Client(api_key=api_key)

# -------------------------------------------------------------------
# Path handling
# -------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GENERATED_DIR = os.path.join(BASE_DIR, "Generated")
os.makedirs(GENERATED_DIR, exist_ok=True)


# -------------------------------------------------------------------
# JSON cleaner
# -------------------------------------------------------------------
def clean_and_parse_json(text: str):
    text = text.strip()

    # Remove markdown code fences if the model added them
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].lstrip()

    return json.loads(text)


# -------------------------------------------------------------------
# Generate Critical Reasoning (GMAT requires exactly 6)
# -------------------------------------------------------------------
def generate_cr():

    cr_prompt = """
You are an expert GMAT Critical Reasoning question writer.

Produce EXACTLY **6 Critical Reasoning questions** (this is the real GMAT Focus count).

Output MUST be:
- either a JSON list of 6 objects
- OR a JSON object with {"questions": [...]}

Each CR question must follow:

{
  "topic": "verbal",
  "subtopic": "critical_reasoning",
  "id": "VCR_x",
  "subsubtopic": "strengthen/weaken/assumption/inference/evaluate/flaw",
  "skill": "short description",
  "difficulty": "easy/medium/hard",
  "question": "...",
  "options": {"A": "...", "B": "...", "C": "...", "D": "...", "E": "..."},
  "answer": "A/B/C/D/E",
  "solution": "short explanation"
}

RULES:
- IDs must be VCR_1 through VCR_6
- Only one correct answer
- No politics, religion, or sensitive topics
- Must be authentic GMAT reasoning
- MUST return valid JSON only with no markdown

BEGIN NOW.
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=cr_prompt
    )

    data = clean_and_parse_json(response.text)

    # Normalize
    if isinstance(data, dict) and "questions" in data:
        questions = data["questions"]
    elif isinstance(data, list):
        questions = data
    else:
        raise ValueError("Unexpected JSON returned for CR.")

    cr_obj = {
        "topic": "verbal",
        "subtopic": "critical_reasoning",
        "questions": questions
    }

    out_path = os.path.join(GENERATED_DIR, "CR_generated.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(cr_obj, f, ensure_ascii=False, indent=2)

    print(f"Saved CR questions to {out_path}")


# -------------------------------------------------------------------
# Generate Reading Comprehension (GMAT requires exactly 17)
# -------------------------------------------------------------------
def generate_rc():

    rc_prompt = """
You are an expert GMAT Reading Comprehension generator.

Generate EXACTLY **17 RC questions**, grouped under **3–4 passages**.
This matches the real GMAT Focus exam.

Output MUST be:
- either a JSON list of 17 question objects
- OR {"questions": [...]}

REQUIREMENTS:
- 3 to 4 passages (each 120–170 words)
- Each passage has 3–6 questions
- Total must equal exactly 17
- Academic, neutral GMAT tone

Each question object must follow:

{
  "topic": "verbal",
  "subtopic": "reading_comprehension",
  "id": "VRC_x",
  "subsubtopic": "main idea / purpose / inference / detail / attitude / structure",
  "skill": "short description",
  "difficulty": "easy/medium/hard",
  "passage": "ONLY on first question of each passage; null otherwise",
  "question": "...",
  "options": {"A": "...", "B": "...", "C": "...", "D": "...", "E": "..."},
  "answer": "A/B/C/D/E",
  "solution": "short explanation"
}

NAMING RULES:
- IDs must be VRC_1 through VRC_17
- Passages appear exactly once per block

QUALITY:
- Accurate reasoning
- GMAT-style structure
- Valid JSON only, no markdown

BEGIN NOW.
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=rc_prompt
    )

    data = clean_and_parse_json(response.text)

    # Normalize
    if isinstance(data, dict) and "questions" in data:
        questions = data["questions"]
    elif isinstance(data, list):
        questions = data
    else:
        raise ValueError("Unexpected JSON returned for RC.")

    rc_obj = {
        "topic": "verbal",
        "subtopic": "reading_comprehension",
        "questions": questions
    }

    out_path = os.path.join(GENERATED_DIR, "RC_generated.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rc_obj, f, ensure_ascii=False, indent=2)

    print(f"Saved RC questions to {out_path}")


# -------------------------------------------------------------------
# Run both generators
# -------------------------------------------------------------------
if __name__ == "__main__":
    generate_cr()
    generate_rc()
