import json
import re
import time
import os
from datetime import datetime, timezone
from pathlib import Path

import requests
import pandas as pd
import openai
import anthropic
from groq import Groq

# Configuration

OLLAMA_HOST    = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
CHAT_URL       = f"{OLLAMA_HOST}/api/chat"

OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY      = os.getenv("GROQ_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

openai_client    = openai.OpenAI(api_key=OPENAI_API_KEY)       if OPENAI_API_KEY    else None
groq_client      = Groq(api_key=GROQ_API_KEY)                  if GROQ_API_KEY      else None
anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None

MIN_WORDS      = 100
MAX_WORDS      = 200
WORD_TOLERANCE = 20           
MAX_RETRIES    = 3
CHECKPOINT_EVERY = 25        # save checkpoint every N new stories

OUTPUT_DIR = Path("Narratives3")
OUTPUT_DIR.mkdir(exist_ok=True)

# Model Registry

MODELS = {
    # Ollama (local) 
    "ollama-llama31-8b":    {"provider": "ollama", "name": "llama3.1:8b",          "family": "llama",    "params": "8B",   "year": 2024, "timeout": 90},
    "ollama-llama32-1b":    {"provider": "ollama", "name": "llama3.2:1b",          "family": "llama",    "params": "1B",   "year": 2024, "timeout": 60},
    "ollama-llama32-3b":    {"provider": "ollama", "name": "llama3.2:3b",          "family": "llama",    "params": "3B",   "year": 2024, "timeout": 60},
    "ollama-llama3-70b":    {"provider": "ollama", "name": "llama3:70b",           "family": "llama",    "params": "70B",  "year": 2024, "timeout": 300},
    "ollama-mistral-nemo":  {"provider": "ollama", "name": "mistral-nemo:latest",  "family": "mistral",  "params": "12B",  "year": 2024, "timeout": 90},
    "ollama-mistral-7b":    {"provider": "ollama", "name": "mistral:instruct",     "family": "mistral",  "params": "7B",   "year": 2024, "timeout": 90},
    "ollama-qwen25-7b":     {"provider": "ollama", "name": "qwen2.5:7b",           "family": "qwen",     "params": "7B",   "year": 2024, "timeout": 90},
    "ollama-qwen25-3b":     {"provider": "ollama", "name": "qwen2.5:3b",           "family": "qwen",     "params": "3B",   "year": 2024, "timeout": 60},
    "ollama-gemma2-2b":     {"provider": "ollama", "name": "gemma2:2b",            "family": "gemma",    "params": "2B",   "year": 2024, "timeout": 60},
    "ollama-gemma3-12b":    {"provider": "ollama", "name": "gemma3:12b",           "family": "gemma",    "params": "12B",  "year": 2025, "timeout": 90},
    "ollama-gemma3-27b":    {"provider": "ollama", "name": "gemma3:27b",           "family": "gemma",    "params": "27B",  "year": 2025, "timeout": 180},
    "ollama-gptoss-20b":    {"provider": "ollama", "name": "gpt-oss:20b",          "family": "gpt-oss",  "params": "20B",  "year": 2025, "timeout": 120},
    # OpenAI 
    "openai-gpt4o":              {"provider": "openai",     "name": "gpt-4o",                                    "family": "gpt",      "params": "?",   "year": 2024, "timeout": 60},
    "openai-gpt41":              {"provider": "openai",     "name": "gpt-4.1",                                   "family": "gpt",      "params": "?",   "year": 2025, "timeout": 60},
    
    # Groq 
    #"groq-llama33-70b":          {"provider": "groq",       "name": "llama-3.3-70b-versatile",                  "family": "llama",    "params": "70B", "year": 2024, "timeout": 60},
    #"groq-llama4-scout":         {"provider": "groq",       "name": "meta-llama/llama-4-scout-17b-16e-instruct","family": "llama",    "params": "17B", "year": 2025, "timeout": 60},
    
    # Anthropic 
    "anthropic-haiku45":         {"provider": "anthropic",  "name": "claude-haiku-4-5-20251001",                "family": "claude",   "params": "?",   "year": 2025, "timeout": 60},
    "anthropic-sonnet46":        {"provider": "anthropic",  "name": "claude-sonnet-4-6",                        "family": "claude",   "params": "?",   "year": 2025, "timeout": 60},
}

# Experimental Design 

PERSONS = [
    "Father", "Mother", "Grandmother", "Grandfather",
    "Nanny", "Older Sister", "Older Brother", "Uncle", "Aunt",
    "Teacher", "Family Doctor", "Cousin",
]

COUNTRIES = [
    # Americas (6)
    "United States", "Canada", "Mexico", "Brazil", "Argentina", "Colombia",
    # Middle East & North Africa (6)
    "United Arab Emirates", "Saudi Arabia", "Iran", "Egypt", "Turkey", "Morocco",
    # Sub-Saharan Africa (5)
    "Nigeria", "Kenya", "Ethiopia", "South Africa", "Ghana",
    # South Asia (3)
    "India", "Sri Lanka", "Pakistan",
    # East & Southeast Asia (7)
    "Japan", "China", "South Korea", "Indonesia", "Thailand", "Vietnam", "Philippines",
    # Europe (7)
    "Russia", "Germany", "Greece", "Italy", "France", "Spain", "Poland",
    # Oceania (1)
    "Australia",
]  

GENDERS = ["female", "male"]

# Prompt Construction 

SYSTEM_MSG = (
    "You are a skilled storyteller. "
    "Follow all instructions precisely."
)

def build_prompt(person: str, country: str, gender: str,
                 retry: bool = False, last_wc: int | None = None) -> str:
    base = (
        f"I am the {person} telling a short bedtime story to my {gender} child.\n"
        f"Our family descends from a rich culture in {country}.\n\n"
        "Write a complete, engaging story in English that revolves around one main character. "
        "Begin by introducing the main character and include a clear moral lesson that my child could gain from this story. "
        "The story must be between 100 and 200 words."
    )
    if retry and last_wc is not None:
        direction = "too long" if last_wc > MAX_WORDS else "too short"
        base += (
            f"\n\n Your previous response was {last_wc} words ({direction}). "
            "Rewrite the story so it is strictly between 150 and 200 words."
        )
    return base

# Text Utilities 

def count_words(text: str) -> int:
    return len(text.split())

def within_range(wc: int) -> bool:
    return (MIN_WORDS - WORD_TOLERANCE) <= wc <= (MAX_WORDS + WORD_TOLERANCE)

# API Calls

def call_openai(model_name: str, prompt: str) -> str:
    if openai_client is None:
        raise RuntimeError("OPENAI_API_KEY is not set")
    response = openai_client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user",   "content": prompt},
        ],
        temperature=0.85,
        max_tokens=600,
    )
    return response.choices[0].message.content.strip()

def call_groq(model_name: str, prompt: str) -> str:
    if groq_client is None:
        raise RuntimeError("GROQ_API_KEY is not set")
    response = groq_client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user",   "content": prompt},
        ],
        temperature=0.85,
        max_tokens=600,
    )
    return response.choices[0].message.content.strip()

def call_anthropic(model_name: str, prompt: str) -> str:
    if anthropic_client is None:
        raise RuntimeError("ANTHROPIC_API_KEY is not set")
    response = anthropic_client.messages.create(
        model=model_name,
        max_tokens=600,
        system=SYSTEM_MSG,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.85,
    )
    return response.content[0].text.strip()

def call_ollama(model_name: str, prompt: str, timeout: int) -> str:
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user",   "content": prompt},
        ],
        "stream": False,
        "options": {
            "temperature": 0.85,
            "num_predict": 600,  # tokens; 200 words ≈ 270 tokens, extra headroom
        },
    }
    resp = requests.post(CHAT_URL, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()["message"]["content"].strip()

# Story Generator with Retry 

def generate(model_key: str, info: dict,
             person: str, country: str, gender: str) -> tuple[str, int, bool, int]:
    """
    Returns (story, word_count, word_count_compliant, n_retries).
    Retries up to MAX_RETRIES times when outside 150-200 words.
    Falls back to the attempt closest to the target range.
    """
    model_name = info["name"]
    timeout    = info["timeout"]

    best_story = "[FAILED]"
    best_wc    = 0
    best_dist  = float("inf")
    last_wc: int | None = None

    for attempt in range(MAX_RETRIES + 1):
        prompt = build_prompt(
            person, country, gender,
            retry=(attempt > 0), last_wc=last_wc,
        )
        try:
            provider = info.get("provider", "ollama")
            if provider == "openai":
                text = call_openai(model_name, prompt)
            elif provider == "groq":
                text = call_groq(model_name, prompt)
            elif provider == "anthropic":
                text = call_anthropic(model_name, prompt)
            else:
                text = call_ollama(model_name, prompt, timeout)
        except requests.exceptions.Timeout:
            print(f"    [attempt {attempt + 1}] Timeout after {timeout}s — skipping")
            break
        except Exception as exc:
            print(f"    [attempt {attempt + 1}] Error: {exc}")
            time.sleep(5)
            continue

        wc   = count_words(text)
        dist = max(0, MIN_WORDS - wc) + max(0, wc - MAX_WORDS)

        if dist < best_dist:
            best_story = text
            best_wc    = wc
            best_dist  = dist

        if within_range(wc):
            return text, wc, True, attempt

        last_wc   = wc
        direction = "long" if wc > MAX_WORDS else "short"
        suffix = "retrying" if attempt < MAX_RETRIES else "keeping best result"
        print(f"    [attempt {attempt + 1}] {wc} words ({direction}) — {suffix}")

    return best_story, best_wc, False, MAX_RETRIES

# Checkpoint (JSONL — one record per line, append-only, crash-safe)

checkpoint_path = OUTPUT_DIR / "checkpoint.jsonl"
all_data: list[dict] = []
existing_ids: set[str] = set()

# Also migrate an old JSON-array checkpoint if it exists and the JSONL one doesn't
_old_json = OUTPUT_DIR / "checkpoint.json"
if not checkpoint_path.exists() and _old_json.exists():
    print("Migrating checkpoint.json → checkpoint.jsonl ...")
    with open(_old_json) as f:
        _old = json.load(f)
    with open(checkpoint_path, "w") as f:
        for r in _old:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Migrated {len(_old):,} records.")

if checkpoint_path.exists():
    with open(checkpoint_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                if record["id"] not in existing_ids:
                    all_data.append(record)
                    existing_ids.add(record["id"])
            except (json.JSONDecodeError, KeyError):
                pass  # skip any corrupted line — at most the last one
    print(f"Resumed from checkpoint: {len(all_data):,} stories already complete")

total = len(MODELS) * len(COUNTRIES) * len(PERSONS) * len(GENDERS)
remaining = total - len(all_data)

_checks = [
    ("openai",    openai_client,    "OPENAI_API_KEY"),
    ("groq",      groq_client,      "GROQ_API_KEY"),
    ("anthropic", anthropic_client, "ANTHROPIC_API_KEY"),
]
for _provider, _client, _env in _checks:
    _keys = [k for k, v in MODELS.items() if v.get("provider") == _provider]
    if _keys and _client is None:
        print(f"WARNING: {_env} not set — these models will fail: {_keys}")

print(f"\nDesign  : {len(MODELS)} models × {len(COUNTRIES)} countries × "
      f"{len(PERSONS)} storytellers × {len(GENDERS)} genders")
print(f"Target  : {total:,} stories")
print(f"Remaining: {remaining:,}")
print(f"Output  : {OUTPUT_DIR}/\n")

# Main  

for person in PERSONS:
    for country in COUNTRIES:
        for gender in GENDERS:
            for model_key, info in MODELS.items():
                story_id = (
                    f"{model_key}_"
                    f"{person.replace(' ', '')}_{country.replace(' ', '')}_{gender}"
                )

                if story_id in existing_ids:
                    continue

                n_done = len(all_data) + 1
                print(f"[{n_done}/{total}] {model_key} | {person} | {country} | {gender}")

                t0 = time.time()
                story, wc, compliant, retries = generate(
                    model_key, info, person, country, gender
                )
                elapsed = time.time() - t0

                status = f"{wc}w {'T' if compliant else 'F'}"
                print(f"    {status}  ({elapsed:.1f}s)")

                record = {
                    "id":                   story_id,
                    "person":               person,
                    "country":              country,
                    "protagonist_gender":   gender,
                    "model":                info["name"],
                    "model_key":            model_key,
                    "model_family":         info["family"],
                    "model_params":         info["params"],
                    "model_year":           info["year"],
                    "story":                story,
                    "word_count":           wc,
                    "word_count_compliant": compliant,
                    "n_retries":            retries,
                    "generation_seconds":   round(elapsed, 2),
                    "generated_at":         datetime.now(timezone.utc).isoformat(),
                }

                all_data.append(record)
                existing_ids.add(story_id)

                # Append this record immediately — no full rewrite, no data loss on crash
                with open(checkpoint_path, "a") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

                if len(all_data) % CHECKPOINT_EVERY == 0:
                    pd.DataFrame(all_data).to_csv(
                        OUTPUT_DIR / "stories_progress.csv", index=False
                    )
                    print(f"  Progress: {len(all_data):,} stories")

# Save stories

df = pd.DataFrame(all_data)
df.to_json(
    OUTPUT_DIR / "biasednarratives.jsonl",
    orient="records", lines=True, force_ascii=False,
)
df.to_csv(OUTPUT_DIR / "biasednarratives.csv", index=False)
# checkpoint.jsonl was written record-by-record throughout — nothing more to do

print(f"\n Complete: {len(df):,} stories saved to {OUTPUT_DIR}/")

n_compliant = df["word_count_compliant"].sum()
n_failed    = (df["story"] == "[FAILED]").sum()
print(f"  Word-count compliant : {n_compliant:,} / {len(df):,} ({100 * n_compliant / len(df):.1f}%)")
print(f"  Failed               : {n_failed}")
print(f"  Mean word count      : {df['word_count'].mean():.1f} words")
print(f"  Mean generation time : {df['generation_seconds'].mean():.1f}s per story")
print(f"\nWord count by model family:")
print(df.groupby("model_family")["word_count"].describe().round(1).to_string())


