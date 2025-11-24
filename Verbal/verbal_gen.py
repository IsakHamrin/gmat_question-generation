import os
import json
from dotenv import load_dotenv
from google import genai
import re

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
    # Remove code fences
    text = re.sub(r"```(?:json)?", "", text).strip()

    # Attempt to extract the largest JSON-like block
    matches = re.findall(r"\{[\s\S]*\}|\[[\s\S]*\]", text)
    if matches:
        text = max(matches, key=len)

    # Fix trailing commas in objects and arrays
    text = re.sub(r",\s*}", "}", text)
    text = re.sub(r",\s*]", "]", text)

    # Normalize fancy quotes to standard quotes
    text = text.replace("“", "\"").replace("”", "\"")
    text = text.replace("\u201c", "\"").replace("\u201d", "\"")

    # Compress whitespace
    text = re.sub(r"\s+", " ", text)

    # Try to load JSON
    try:
        return json.loads(text)
    except Exception as e:
        print("\n===== JSON PARSE FAILURE =====")
        print("Raw cleaned text:", text)
        print("==============================\n")
        raise e


# -------------------------------------------------------------------
# Generate Critical Reasoning (GMAT requires exactly 6)
# -------------------------------------------------------------------
def generate_cr():

    cr_prompt = """
You are an expert GMAT Critical Reasoning (CR) writer.

Your task: Produce EXACTLY **6 Critical Reasoning questions** that follow authentic
GMAT Focus reasoning. All content must be fully self-contained inside each question.

------------------------------------------------------------------------------------
CONTENT REQUIREMENTS
------------------------------------------------------------------------------------
• Select **SIX DISTINCT CR subtypes** from this list (no duplicates):
  strengthen, weaken, assumption, inference, evaluate, flaw, paradox, complete_the_argument

• For each chosen subtype:
  – randomly generate ONE completely new question
  – no reuse of earlier questions
  – no duplicated reasoning structures
  – fully self-contained: the prompt must include all information needed
  – no sensitive topics (no politics, religion, social controversies)

------------------------------------------------------------------------------------
OUTPUT FORMAT (STRICT JSON)
------------------------------------------------------------------------------------
Return EITHER:
• a JSON array of 6 objects
OR
• {"questions": [ ... ]}

Each question object must follow this exact schema:

{
  "topic": "verbal",
  "subtopic": "critical_reasoning",
  "id": "VCR_x",                      // x = 1 through 6
  "subsubtopic": "<chosen CR subtype>",
  "skill": "<short description>",
  "difficulty": "easy/medium/hard",
  "question": "<full GMAT-style self-contained scenario and question>",
  "options": {
      "A": "...",
      "B": "...",
      "C": "...",
      "D": "...",
      "E": "..."
  },
  "answer": "A/B/C/D/E",
  "solution": "<short explanation>"
}

------------------------------------------------------------------------------------
HARD RULES
------------------------------------------------------------------------------------
• MUST use **6 different subsubtopics**.
• MUST output **exactly 6** questions.
• IDs must be numbered VCR_1, VCR_2, ..., VCR_6.
• Every question must contain 1 fully self-contained argument (never partial).
• All content must be newly generated.
• MUST return **valid JSON only**, never markdown or prose outside the JSON.

BEGIN NOW.
"""



    response = client.models.generate_content(
    model="gemini-2.5-flash-lite",
    contents=cr_prompt
)

    raw = response.candidates[0].content.parts[0].text
    data = clean_and_parse_json(raw)



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
You are an expert GMAT Reading Comprehension (RC) writer.

Your task: Produce EXACTLY **17 RC questions**, following authentic GMAT Focus format.
QUALITY MUST BE EXCELLENT.

------------------------------------------------------------------------------------
STRUCTURE REQUIREMENTS
------------------------------------------------------------------------------------
• Create **3 or 4 passages**, each 120–170 words.
• Each passage must be academic, neutral, and balanced.
• Distribute 17 questions across passages:
    – Each passage must have 3–6 questions.
    – Total MUST equal 17.

• Passages must be diverse:
    – Choose from: science, history, economics, humanities, social science.
    – No politics, religion, or other sensitive topics.
    – No duplicated passages or trivial variations.

------------------------------------------------------------------------------------
OUTPUT FORMAT (STRICT JSON)
------------------------------------------------------------------------------------
Return EITHER:
• a JSON array of 17 objects
OR
• {"questions": [ ... ]}

Each question object MUST follow this exact schema:

{
  "topic": "verbal",
  "subtopic": "reading_comprehension",
  "id": "VRC_x",                                  // x = 1 through 17
  "subsubtopic": "main idea / purpose / inference / detail / attitude / structure",
  "skill": "<short description>",
  "difficulty": "easy/medium/hard",
  "passage": "<full passage text (MUST be repeated on EVERY question linked to that passage)>",
  "question": "<the question stem>",
  "options": {
      "A": "...",
      "B": "...",
      "C": "...",
      "D": "...",
      "E": "..."
  },
  "answer": "A/B/C/D/E",
  "solution": "<short explanation>"
}

------------------------------------------------------------------------------------
HARD RULES
------------------------------------------------------------------------------------
• IDs must be exactly VRC_1 through VRC_17 in order.
• For every question object, the field `passage` MUST contain the full passage text
  corresponding to that question.
  – NEVER use null, empty string, or placeholders in `passage`.
  – All questions tied to the same passage must use EXACTLY the same passage string.
• Each question must be logically tied to its passage.
• MUST return **valid JSON only**, no markdown.
• Do NOT include any commentary or text outside JSON.

BEGIN NOW.
"""


    response = client.models.generate_content(
    model="gemini-2.5-flash-lite",
    contents=rc_prompt
)

    raw = response.candidates[0].content.parts[0].text
    data = clean_and_parse_json(raw)


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
