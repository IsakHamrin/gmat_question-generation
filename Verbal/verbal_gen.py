import os
import json
from dotenv import load_dotenv
from google import genai

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

def clean_and_parse_json(text: str):
    """
    Cleans possible markdown fences and parses JSON.
    Expects the model to return either:
    - a list of question objects
    - OR an object with 'questions' already.
    """
    text = text.strip()

    # Strip markdown fences if present
    if text.startswith("```"):
        # Remove ```json or ``` and the trailing ```
        text = text.strip("`")
        # In some cases there might still be a leading 'json\n'
        if text.lower().startswith("json"):
            text = text[4:].lstrip()

    data = json.loads(text)
    return data


def generate_cr():
    cr_prompt = """
You are an expert GMAT Critical Reasoning generator.
Produce EXACTLY 6 GMAT-critical-reasoning questions in the following strict JSON schema:

Return EITHER:
1) A JSON array of 6 question objects, OR
2) A JSON object with a key "questions" that holds an array of 6 question objects.

Each CR question must follow this structure:

{
  "topic": "verbal",
  "subtopic": "critical_reasoning",
  "id": "VCR_x",
  "subsubtopic": "strengthen/weaken/assumption/inference/evaluate/flaw",
  "skill": "short description of reasoning skill tested",
  "difficulty": "easy/medium/hard",
  "question": "...",
  "options": {"A": "...", "B": "...", "C": "...", "D": "...", "E": "..."},
  "answer": "A/B/C/D/E",
  "solution": "Clear and concise explanation."
}

RULES:
- Generate EXACTLY 6 CR items.
- IDs must be VCR_1 through VCR_6.
- Only ONE answer can be correct for each item.
- Keep the CR stimuli realistic, logical, and GMAT-authentic.
- Avoid any political, religious, or sensitive topics.
- Output MUST be valid JSON with NO trailing text, no markdown, no comments.
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=cr_prompt
    )

    data = clean_and_parse_json(response.text)

    # Normalize to a list of questions
    if isinstance(data, dict) and "questions" in data:
        questions = data["questions"]
    elif isinstance(data, list):
        questions = data
    else:
        raise ValueError("Unexpected JSON structure from CR generation")

    # Wrap into your desired cr.json format
    cr_obj = {
        "topic": "verbal",
        "subtopic": "critical_reasoning",
        "questions": questions
    }

    with open("CR_generated.json", "w", encoding="utf-8") as f:
        json.dump(cr_obj, f, ensure_ascii=False, indent=2)

    print("Saved CR questions to CR_generated.json")


def generate_rc():
    rc_prompt = """
You are an expert GMAT Reading Comprehension generator.
Produce a complete RC question set of EXACTLY 17 questions across 3–4 passages.

You may return EITHER:
1) A JSON array of 17 question objects, OR
2) A JSON object with a key "questions" that holds an array of 17 question objects.

REQUIREMENTS:
- 3 to 4 passages.
- Each passage must be 120–170 words.
- Each passage must contain 3–6 questions.
- Total questions MUST be exactly 17.
- Topics must vary (science, economics, sociology, history, business).
- Tone must be academic, neutral, and GMAT-like.

FORMAT FOR EACH QUESTION OBJECT:
{
  "topic": "verbal",
  "subtopic": "reading_comprehension",
  "id": "VRC_x",
  "subsubtopic": "main idea / purpose / inference / detail / attitude / structure",
  "skill": "short description",
  "difficulty": "easy/medium/hard",
  "passage": "ONLY INCLUDED on the FIRST question for each passage block, null for others",
  "question": "...",
  "options": { "A": "...", "B": "...", "C": "...", "D": "...", "E": "..." },
  "answer": "A/B/C/D/E",
  "solution": "Clear explanation."
}

NAMING RULES:
- IDs must be VRC_1 through VRC_17.
- Passages must ONLY appear in the FIRST item of their group.
- All other questions under that passage must have "passage": null.

QUALITY RULES:
- Questions must be GMAT-quality and unambiguous.
- Ensure multiple question types: main idea, purpose, inference, detail, structure, attitude.
- Distractors must be plausible but wrong.
- Output must be VALID JSON ONLY. No markdown, no commentary.
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=rc_prompt
    )

    data = clean_and_parse_json(response.text)

    # Normalize to a list of questions
    if isinstance(data, dict) and "questions" in data:
        questions = data["questions"]
    elif isinstance(data, list):
        questions = data
    else:
        raise ValueError("Unexpected JSON structure from RC generation")

    # Wrap into your desired rc.json format
    rc_obj = {
        "topic": "verbal",
        "subtopic": "reading_comprehension",
        "questions": questions
    }

    with open("RC_generated.json", "w", encoding="utf-8") as f:
        json.dump(rc_obj, f, ensure_ascii=False, indent=2)

    print("Saved RC questions to RC_generated.json")


if __name__ == "__main__":
    generate_cr()
    generate_rc()
