"""
chatbot_common.py
------------------
این فایل یک ماژول مشترک (utility module) است که تمام توابع، تنظیمات و بارگذاری‌های داده‌ی تکراری را در یک جا جمع کرده تا در نوت‌بوک‌های مختلف پروژه‌ی چت‌بات بانکی تکرار نشوند.کار اصلی‌اش این است که:

کلاینت مدل زبانی (Claude یا OpenAI) را آماده کند
فایل‌های مربوط به intent و slot را بارگذاری کند
خروجی‌های JSON مدل را به‌درستی پارس کند
و برای ارزیابی، داده‌های train/validation را در دسترس قرار دهد.
"""

import os
import re
import json
from getpass import getpass
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------
# 1) Paths — edit these if your folder layout is different (e.g. Colab
#    + Google Drive). Everything else in the project reads from here.
# ---------------------------------------------------------------------
DATA_DIR = "./data"
TRAIN_DIR = os.path.join(DATA_DIR, "train")            
VAL_DIR = os.path.join(DATA_DIR, "validation")          
INTENT_MAPPING_CSV = os.path.join(DATA_DIR, "Intent_and_Slot_Mapping_Table-2.csv")  
RAW_SLOTS_TXT = os.path.join(DATA_DIR, "slots.txt")      

INTENTS_JSON = os.path.join(DATA_DIR, "intents.json")
SLOTS_JSON = os.path.join(DATA_DIR, "slots.json")
SCHEMA_JSON = os.path.join(DATA_DIR, "intent_slot_schema.json")
FEWSHOT_JSON = os.path.join(DATA_DIR, "few_shot_examples.json")

# ---------------------------------------------------------------------
# 2) LLM client. Switch LLM_PROVIDER to "openai" to use GPT instead of
#    Claude — the rest of the pipeline (prompts, JSON parsing, dialogue
#    manager) does not need to change.
# ---------------------------------------------------------------------
LLM_PROVIDER = "anthropic"          
LLM_MODEL_ANTHROPIC = "claude-sonnet-4-5"
LLM_MODEL_OPENAI = "gpt-4o-mini"

_client = None


def get_llm_client():
    """Create (once) and return the LLM client, asking for an API key
    interactively if it is not already in the environment."""
    global _client
    if _client is not None:
        return _client

    if LLM_PROVIDER == "anthropic":
        import anthropic
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            api_key = getpass("Enter your ANTHROPIC_API_KEY: ")
            os.environ["ANTHROPIC_API_KEY"] = api_key
        _client = anthropic.Anthropic(api_key=api_key)

    elif LLM_PROVIDER == "openai":
        import openai
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            api_key = getpass("Enter your OPENAI_API_KEY: ")
            os.environ["OPENAI_API_KEY"] = api_key
        _client = openai.OpenAI(api_key=api_key)

    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {LLM_PROVIDER}")

    return _client


def call_llm(system_prompt: str, user_prompt: str, max_tokens: int = 1024,
             temperature: float = 0.0) -> str:
    """Single unified call. Returns the raw text of the model's reply."""
    client = get_llm_client()

    if LLM_PROVIDER == "anthropic":
        resp = client.messages.create(
            model=LLM_MODEL_ANTHROPIC,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return "".join(block.text for block in resp.content if block.type == "text")

    else:  # openai
        resp = client.chat.completions.create(
            model=LLM_MODEL_OPENAI,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return resp.choices[0].message.content


def extract_json(text: str) -> dict:
    """Pull the first {...} JSON object out of an LLM response and parse it.
    Tolerates ```json fences and stray text around the object."""
    cleaned = text.strip()
    cleaned = re.sub(r"^```(json)?", "", cleaned.strip(), flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"```$", "", cleaned.strip()).strip()
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in LLM response:\n{text}")
    return json.loads(match.group(0))


# ---------------------------------------------------------------------
# 3) Loading helpers for the artifacts notebook 00 produces
# ---------------------------------------------------------------------
def load_json_file(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_intents(path: str = INTENTS_JSON) -> List[str]:
    return load_json_file(path)


def load_slots(path: str = SLOTS_JSON) -> Dict[str, List[str]]:
    return load_json_file(path)


def load_schema(path: str = SCHEMA_JSON) -> Dict[str, List[dict]]:
    return load_json_file(path)


def load_fewshot(path: str = FEWSHOT_JSON) -> Dict[str, List[dict]]:
    if os.path.exists(path):
        return load_json_file(path)
    return {}


def canonical_slot_name(bio_tag: str) -> str:
    """'b-cardـnumber' / 'i-cardـnumber' -> 'card_number'
    (also copes with the one 'issuance card' entry that has a literal space)"""
    name = bio_tag.split("-", 1)[1] if "-" in bio_tag else bio_tag
    name = name.replace("ـ", "_").replace(" ", "_")
    return name


def load_raw_examples(folder_path: str) -> List[dict]:
    """Load the original {input_text, intent_id, slots:[...]} json files,
    if the user has them available locally."""
    data = []
    if not os.path.isdir(folder_path):
        return data
    for fn in sorted(os.listdir(folder_path)):
        if fn.endswith(".json"):
            with open(os.path.join(folder_path, fn), encoding="utf-8") as f:
                data.append(json.load(f))
    return data


def bio_slots_to_dict(input_text: str, bio_labels: List[str]) -> Dict[str, str]:
    """Turn a per-word BIO label sequence (as stored in the original dataset)
    into a {slot_name: value} dict, merging consecutive B-/I- words."""
    words = input_text.split()
    out: Dict[str, List[str]] = {}
    current_slot = None
    for word, label in zip(words, bio_labels):
        if label == "o":
            current_slot = None
            continue
        prefix, _, rest = label.partition("-")
        slot = canonical_slot_name(label)
        if prefix == "b" or slot != current_slot:
            out.setdefault(slot, [])
            out[slot].append(word)
            current_slot = slot
        elif prefix == "i" and slot == current_slot:
            out[slot][-1] += " " + word
    return {k: v[0] if len(v) == 1 else " / ".join(v) for k, v in out.items()}
