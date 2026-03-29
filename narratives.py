import json
import time
import os
from datetime import datetime
from pathlib import Path
import random
import pandas as pd
from groq import Groq
# import google.generativeai as genai  # Deprecated package

#groq


GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY is not set")

#Gemini (package deprecated)
# os.environ["GEMINI_API_KEY"] = "AIzaSyBZ1jerQISi1lWTK0QBrbAmB05Nu-MXKH0"
# genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

groq_client = Groq()


#Available Free Models - Currently supported Groq models only
MODELS = {
    "groq-llama33-70b": {
        "client": groq_client,
        "name": "llama-3.3-70b-versatile",   
        "provider": "groq",
        "generation": "new",
        "year": 2024
    },
    "groq-llama31-8b": {
        "client": groq_client,
        "name": "llama-3.1-8b-instant",     
        "provider": "groq",
        "generation": "new",
        "year": 2024
    },
    "groq-gptoss-120b": {
        "client": groq_client,
        "name": "openai/gpt-oss-120b",      
        "provider": "groq",
        "generation": "new",
        "year": 2025
    },
    "groq-gptoss-20b": {
        "client": groq_client,
        "name": "openai/gpt-oss-20b",       
        "provider": "groq",
        "generation": "new",
        "year": 2025
    },
    "groq-qwen32b": {
        "client": groq_client,
        "name": "qwen/qwen3-32b",           
        "provider": "groq",
        "generation": "new",
        "year": 2025
    }
}


#TRAIT_MAP = { ... }  #Paste the full TRAIT_MAP here (6 archetypes with settings/names)

#Prompt
def build_prompt(person, country, gender):
    return f"""I am the {person} telling a short bedtime story to my {gender} child. 
Our family descends from a rich culture in {country}.

Write me a complete, engaging story in English, that revolves around one main character. 
Begin the story by introducing the main character and include a lesson that my child could gain out of this story. 
Use a maximum of 200 words.

""".strip()

#Generation Functions
def generate_groq(prompt, model_name):
    for _ in range(3):
        try:
            response = groq_client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.85,
                max_tokens=1800 if "long" in prompt.lower() else 800
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Groq error: {e}")
            time.sleep(10)
    return "[FAILED]"

def generate_gemini(prompt, model_name):
    for _ in range(3):
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(
                prompt, 
                generation_config={
                    "temperature": 0.85, 
                    "max_output_tokens": 800
                }
            )
            return response.text.strip()
        except Exception as e:
            print(f"Gemini error: {e}")
            time.sleep(5)
    return "[FAILED]"

#MAIN 
OUTPUT_DIR = Path("Narratives2")
OUTPUT_DIR.mkdir(exist_ok=True)

INDIVIDUAL_DIR = OUTPUT_DIR / "individual_stories"
INDIVIDUAL_DIR.mkdir(exist_ok=True)

checkpoint = OUTPUT_DIR / "checkpoint.json"
all_data = []
existing_ids = set()

if checkpoint.exists():
    all_data = json.load(open(checkpoint, "r"))
    existing_ids = {record["id"] for record in all_data}

#archetypes = list(TRAIT_MAP.keys())
persons = ["Father", "Mother", "Grandmother", "Grandfather", "Nanny", "Older Sister", "Older Brother", "Uncle", "Aunt"]
countries = ["United States", "Canada", "Mexico", "Brazil", "Argentina", "United Arab Emirates", "Saudi Arabia", "Nigeria", "Kenya", "India", "Sri Lanka", "Japan", "China", "South Korea", "Russia", "Germany", "Iran", "Egypt", "Greece", "Australia"]
genders = ["female", "male"]
#lengths = ["short", "long"]

total_combinations = len(persons) * len(countries) * len(genders) * len(MODELS)
print(f"Total stories to generate: {total_combinations}")
print(f"Already generated: {len(all_data)}")

#print(f"Need {total - len(all_data)} more stories")

for pers in persons:
    for country in countries:
        for gen in genders:
            prompt = build_prompt(pers, country, gen)

            for key, info in MODELS.items():
                #Create a clean, unique ID without name/length
                story_id = f"{key}_{pers.replace(' ', '')}_{country.replace(' ', '')}_{gen}"

                if story_id in existing_ids:
                    print(f"Skipping existing: {story_id}")
                    continue

                print(f"Generating [{len(all_data)+1}/{total_combinations}]: {story_id}")

                #Generate story
                if info["provider"] == "groq":
                    story = generate_groq(prompt, info["name"])
                # elif info["provider"] == "gemini":
                #     story = generate_gemini(prompt, info["name"])
                #     time.sleep(4)  # Rate limit: 15 req/min for free tier
                else:
                    story = "[UNKNOWN PROVIDER]"

                #Create record
                record = {
                    "id": story_id,
                    "person": pers,                    
                    "country": country,
                    "protagonist_gender": gen,
                    "model": info.get("name", "unknown"),
                    "provider": key,
                    "model_generation": info.get("generation", "unknown"),
                    "model_year": info.get("year", "unknown"),
                    "prompt": prompt,
                    "story": story,
                    "word_count": len(story.split()) if story not in ["[FAILED]", "[UNKNOWN PROVIDER]"] else 0,
                    "generated_at": datetime.utcnow().isoformat()
                }

                all_data.append(record)
                existing_ids.add(story_id)

                #Save individually
                #safe_filename = "".join(c if c.isalnum() or c in "-_" else "_" for c in story_id)
                #txt_path = INDIVIDUAL_DIR / f"{safe_filename}.txt"
                #txt_path.write_text(
                #    f"ID: {story_id}\n"
                #    f"Person: {pers}\n"
                #    f"Country: {country}\n"
                #    f"Gender: {gen}\n"
                #    f"Model: {info.get('name', 'unknown')}\n"
                #    f"Generation: {info.get('generation', 'unknown')} ({info.get('year', 'N/A')})\n\n"
                #    f"--- PROMPT ---\n{prompt}\n\n"
                #    f"--- STORY ---\n{story}",
                #    encoding="utf-8"
                #) 

                #Checkpoint save
                if len(all_data) % 10 == 0:
                    #Save checkpoint
                    json.dump(all_data, open(checkpoint, "w"), indent=2, ensure_ascii=False)
                    #Save CSV preview
                    pd.DataFrame(all_data).to_csv(OUTPUT_DIR / "stories_progress.csv", index=False)
                    print(f"Checkpoint saved: {len(all_data)} stories")

                time.sleep(2)  #Respect rate limits

#Save
final_df = pd.DataFrame(all_data)

final_df.to_json(OUTPUT_DIR / "biasednarratives.jsonl", orient="records", lines=True, force_ascii=False)
final_df.to_csv(OUTPUT_DIR / "biasednarratives.csv", index=False)

print("\nGENERATION COMPLETE")
print(f"Total stories generated: {len(all_data)}")
print(f"Files saved in: {OUTPUT_DIR}")
#print(f"Individual .txt stories: {INDIVIDUAL_DIR} ({len(list(INDIVIDUAL_DIR.glob('*.txt')))} files)")