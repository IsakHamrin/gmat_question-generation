import os
import re
import json
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import chain
import random

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=API_KEY)


default_model = "gemini-2.5-flash-lite"

# with open("hp.json", "r", encoding="utf-8") as f:
#     data = json.load(f)

texts = []

def extract_json_from_text(text):
    """Extract and parse JSON array from model text output"""
    json_match = re.search(r'\[.*\]', text, re.DOTALL)
    if not json_match:
        print("No JSON array found in text")
        return []

    json_str = json_match.group(0)
    try:
        data = json.loads(json_str)
        return data if isinstance(data, list) else [data]
    except json.JSONDecodeError:
        items = re.findall(r'\{[^{}]+\}', text, re.DOTALL)
        parsed = []
        for item in items:
            try:
                parsed.append(json.loads(item))
            except:
                continue
        return parsed


model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode([t["text"] for t in texts], normalize_embeddings=True)


def retrieve_section():
    """Return all chunks from a specific section (e.g. 'XYZ', 'KVA', 'NOG')."""
    return texts


DATA_DIR = "."


def _load_json(filename):
    path = os.path.join(DATA_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "items" in data:
        return data["items"]
    if isinstance(data, list):
        return data
    return []


def _truncate(s, n):
    return s if len(s) <= n else s[:n] + "..."


def _build_context_block(name, docs, max_items=24, max_char_per_item=320):
    lines = [f"{name}—referensutdrag:"]
    for i, d in enumerate(docs[:max_items]):
        sect = d.get("section", name.upper())
        txt = d.get("text") or d.get("question") or d.get("body") or ""
        lines.append(f"{i+1}. [{sect}] {_truncate(str(txt), max_char_per_item)}")
    return "\n".join(lines)


def _load_all_refs():
    xyz_refs = _load_json("xyz.json")
    kva_refs = _load_json("kva.json")
    nog_refs = _load_json("nog.json")
    return xyz_refs, kva_refs, nog_refs

# -------------------------------
# JSON Schemas (ENFORCEMENT)  # NEW
# -------------------------------
xyz_schema = {
  "type": "array",
  "minItems": 12, "maxItems": 12,
  "items": {
    "type": "object",
    "properties": {
      "id":   {"type": "integer"},
      "type": {"type": "string"},
      "text": {"type": "string"},
      "explanation": {"type": "string"},
      "answer": {"type": "integer"},
      "options": {
        "type": "array",
        "minItems": 4, "maxItems": 4,
        "items": {"type": "string"}
      }
    },
    "required": ["id","type","text","explanation","answer","options"]
  }
}
kva_schema = {
  "type": "array",
  "minItems": 10, "maxItems": 10,
  "items": {
    "type": "object",
    "properties": {
      "id":   {"type": "integer"},
      "type": {"type": "string"},
      "text": {"type": "string"},
      "explanation": {"type": "string"},
      "answer": {"type": "integer"},
      "options": {
        "type": "array",
        "minItems": 4, "maxItems": 4,
        "items": {"type": "string"}
      }
    },
    "required": ["id","type","text","explanation","answer","options"]
  }
}
nog_schema = {
  "type": "array",
  "minItems": 6, "maxItems": 6,
  "items": {
    "type": "object",
    "properties": {
      "id":   {"type": "integer"},
      "type": {"type": "string"},
      "text": {"type": "string"},
      "explanation": {"type": "string"},
      "answer": {"type": "integer"},
      "options": {
        "type": "array",
        "minItems": 5, "maxItems": 5,
        "items": {"type": "string"}
      }
    },
    "required": ["id","type","text","explanation","answer","options"]
  }
}
# -------------------------------


system_msg_xyz = """
Du är en uppgiftsmakare för Högskoleprovet. Skapa nya, originella frågor i svensk text.

UTDATAFORMAT
- Returnera ENDAST en JSON-lista (array) av objekt. Ingen annan text.
- Varje objekt har exakt:
  - "id": heltal, starta på 1 och öka med 1
  - "type": "XYZ"
  - "text": sträng med fullständig frågetext
  - "explanation": kort lösningsförklaring och motivering
  - "answer": heltal 0-3 som anger index i "options"
  - "options": lista med exakt 4 svarsalternativ [str, str, str, str], rätt alternativ måste vara med!


MÄNGDKRAV
- Skapa exakt: 12 st "XYZ".
- Id 1..12 i ordning.

KVALITET OCH BEGRÄNSNINGAR
- Frågorna ska vara originella men inspirerade av referensmaterialets stil och nivå.
- Ingen plagiering eller återanvändning av exakt text.
- Endast en korrekt lösning per fråga.
- Förklaringen ska visa nyckelsteg eller kort logik.
- "answer" måste peka på rätt alternativ (0-3) och ska bero på förklaringen.

TYPREGLER
- XYZ: Kvantitativ uppgift med 4 alternativ. Variation: aritmetik, algebra, procent, geometri, tabell/diagramtolkning.

VALIDERING FÖRE SVAR
- Räkna ut facit. Det rätta måste finnas bland svarsalternativen.
- Kontrollera mängdkrav: 12 XYZ
- All text på svenska. Ingen känslig persondata.

UTDATA
- Returnera enbart JSON-arrayen.
"""
system_msg_kva = """
Du är en uppgiftsmakare för Högskoleprovet. Skapa nya, originella frågor i svensk text.

UTDATAFORMAT
- Returnera ENDAST en JSON-lista (array) av objekt. Ingen annan text.
- Varje objekt har exakt:
  - "id": heltal, starta på 13 och öka med 1
  - "type": "KVA"
  - "text": sträng med fullständig frågetext
  - "explanation": kort lösningsförklaring och motivering
  - "answer": heltal 0-3 som anger index i "options"
  - "options": lista med exakt 4 svarsalternativ [str, str, str, str], rätt alternativ måste vara med!


MÄNGDKRAV
- Skapa exakt: 10 st "KVA".
- Id 13..22 i ordning.

KVALITET OCH BEGRÄNSNINGAR
- Frågorna ska vara originella men inspirerade av referensmaterialets stil och nivå.
- Ingen plagiering eller återanvändning av exakt text.
- Endast en korrekt lösning per fråga.
- Förklaringen ska visa nyckelsteg eller kort logik.
- "answer" måste peka på rätt alternativ (0-3) och ska bero på förklaringen.

TYPREGLER
- KVA: Kvantitetsjämförelse. Textformat:
  "Kvantitet I: ..."
  "Kvantitet II: ..."
  Alternativ MÅSTE vara exakt:
    0) "I är större än II"
    1) "II är större än I"
    2) "I är lika med II"
    3) "Informationen är otillräcklig"

VALIDERING FÖRE SVAR
- Räkna ut facit. Det rätta måste finnas bland svarsalternativen.
- Kontrollera mängdkrav: 10 KVA
- All text på svenska. Ingen känslig persondata.

UTDATA
- Returnera enbart JSON-arrayen.
"""
system_msg_nog = """
Du är en uppgiftsmakare för Högskoleprovet. Skapa nya, originella frågor i svensk text.

UTDATAFORMAT
- Returnera ENDAST en JSON-lista (array) av objekt. Ingen annan text.
- Varje objekt har exakt:
  - "id": heltal, starta på 23 och öka med 1
  - "type": "NOG"
  - "text": sträng med fullständig frågetext
  - "explanation": kort lösningsförklaring och motivering
  - "answer": heltal 0-4 som anger index i "options"
  - "options": lista med exakt 4 svarsalternativ [str, str, str, str], rätt alternativ måste vara med!


MÄNGDKRAV
- Skapa exakt: 6 st "NOG".
- Totalt 6 frågor. Id 23..28 i ordning.

KVALITET OCH BEGRÄNSNINGAR
- Frågorna ska vara originella men inspirerade av referensmaterialets stil och nivå.
- Ingen plagiering eller återanvändning av exakt text.
- Endast en korrekt lösning per fråga.
- Förklaringen ska visa nyckelsteg eller kort logik.
- "answer" måste peka på rätt alternativ (0-3) och ska bero på förklaringen.

TYPREGLER
Data-sufficiency med två påståenden (1) och (2). 
Textraden: "Tillräcklig information erhålls" måste finnas med precis ovanför svarsalternativen!
Alternativ MÅSTE vara exakt:
    0) "i (1) men ej i (2)",
    1) "i (2) men ej i (1)",
    2) "i (1) tillsammans med (2)",
    3) "i (1) och (2) var för sig",
    4) "ej genom de båda påståendena"

VALIDERING FÖRE SVAR
- Räkna ut facit. Det rätta måste finnas bland svarsalternativen.
- Kontrollera mängdkrav: 6 NOG.
- All text på svenska. Ingen känslig persondata.

UTDATA
- Returnera enbart JSON-arrayen.
"""
system_mgs = [system_msg_xyz, system_msg_kva, system_msg_nog]

xyz_refs, kva_refs, nog_refs = _load_all_refs()    

# Taking out random samples of our training data to get a better randomness in the question generation
xyz_refs = random.sample(xyz_refs, k=len(xyz_refs)//2)
kva_refs = random.sample(kva_refs, k=len(kva_refs)//2)
nog_refs = random.sample(nog_refs, k=len(nog_refs)//2)

print(f'xyz refs --------------------------------{xyz_refs}, length: {len(xyz_refs)}')

ctx_xyz = _build_context_block("XYZ", xyz_refs, max_items=36, max_char_per_item=300)
ctx_kva = _build_context_block("KVA", kva_refs, max_items=24, max_char_per_item=300)
ctx_nog = _build_context_block("NOG", nog_refs, max_items=24, max_char_per_item=300)

ctxs = [ctx_xyz, ctx_kva, ctx_nog]

# Map each index to its schema  # NEW
schemas = [xyz_schema, kva_schema, nog_schema]

def generate(system_msg, ctx, schema):  # CHANGED: added schema
    full_context = (
        "Relevant information för nyproducerade frågor.\n"
        "Använd som stil- och svårighetsguide. Kopiera inte text ordagrant.\n\n"
        f"{ctx}"
    )

    full_message = (
        full_context
        + "\nSkapa nu frågorna enligt systeminstruktionen. "
          "Håll dem på liknande svårighet och stil som referensen men gör dem nya."
    )

    contents = []
    parts = []
    if full_message:
        parts.append(types.Part(text=full_message))
    if parts:
        contents.append(types.Content(role="user", parts=parts))

    # CHANGED: structured output constraints
    resp = client.models.generate_content(
        model=default_model,
        config=types.GenerateContentConfig(
            system_instruction=system_msg,
            response_mime_type="application/json",   # enforce JSON-only
            response_schema=schema                   # enforce exact schema
        ),
        contents=contents,
    )

    # Prefer direct JSON text from the model; fallback to extractor if needed
    raw_text = getattr(resp, "text", "") or ""
    if not raw_text and isinstance(resp, list):
        raw_text = "".join([m.text for m in resp if hasattr(m, "text") and m.text])

    questions = []
    if raw_text:
        try:
            questions = json.loads(raw_text)
        except Exception:
            questions = extract_json_from_text(raw_text)
    return questions

result = []
def all():
    for i in range(3):
        r = generate(system_mgs[i], ctxs[i], schemas[i])
        result.append(r)
    flat_results = list(chain.from_iterable(result))
    print(flat_results)
    with open("genrated_questions.json", "w") as outfile:
        json.dump(flat_results, outfile, indent=2)
    return flat_results
    
all()
