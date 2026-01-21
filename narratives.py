import json
import time
import os
from datetime import datetime
from pathlib import Path
import random
import pandas as pd
from groq import Groq
#import google.generativeai as genai  
#import google.genai as genai

#groq
os.environ["GROQ_API_KEY"] = "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"

#Gemini
'''os.environ["GEMINI_API_KEY"] = "AIzaSyBZ1jerQISi1lWTK0QBrbAmB05Nu-MXKH0"
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))'''

groq_client = Groq()


#Availabe Free Models
MODELS = {
    "groq-llama33-70b": {
        "client": groq_client,
        "name": "llama-3.3-70b-versatile",   
        "provider": "groq"
    },
    "groq-llama8b": {
        "client": groq_client,
        "name": "llama-3.1-8b-instant",     
        "provider": "groq"
    },
    "groq-gptoss-120b": {
        "client": groq_client,
        "name": "openai/gpt-oss-120b",      
        "provider": "groq"
    },
    "groq-gptoss-20b": {
        "client": groq_client,
        "name": "openai/gpt-oss-20b",       
        "provider": "groq"
    },
    "groq-qwen32b": {
        "client": groq_client,
        "name": "qwen/qwen3-32b",           
        "provider": "groq"
    }
}


#TRAIT_MAP = { ... }  #Paste the full TRAIT_MAP here (6 archetypes with settings/names)

#Prompt
def build_prompt(person, culture, gender):
    return f"""I am the {person} telling a short bedtime story to my {gender} child. 
Our family descends from a rich {culture} culture.

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

'''def generate_gemini(prompt):
    try:
    OPTION 1
        model = MODELS["gemini-flash"]["model"]
        response = model.generate_content(prompt, generation_config={"temperature": 0.85, "max_output_tokens": 1800})
        return response.text.strip()

        OPTION 2
        model = genai.GenerativeModel('gemini-1.5-flash-latest')  
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Gemini error: {e}")
        return "[FAILED]"
'''

#MAIN 
OUTPUT_DIR = Path("Narratives")
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
cultures = ["European", "Sri Lankan", "Japanese", "Persian", "Kenyan"]
genders = ["female", "male"]
#lengths = ["short", "long"]

total_combinations = len(persons) * len(cultures) * len(genders) * len(MODELS)
print(f"Total stories to generate: {total_combinations}")
print(f"Already generated: {len(all_data)}")

#print(f"Need {total - len(all_data)} more stories")

for pers in persons:
    for cult in cultures:
        for gen in genders:
            prompt = build_prompt(pers, cult, gen)

            for key, info in MODELS.items():
                #Create a clean, unique ID without name/length
                story_id = f"{key}_{pers.replace(' ', '')}_{cult.replace(' ', '')}_{gen}"

                if story_id in existing_ids:
                    print(f"Skipping existing: {story_id}")
                    continue

                print(f"Generating [{len(all_data)+1}/{total_combinations}]: {story_id}")

                #Generate story
                if info["provider"] == "groq":
                    story = generate_groq(prompt, info["name"])
                elif info["provider"] == "gemini":
                    story = generate_gemini(prompt)
                else:
                    story = "[UNKNOWN PROVIDER]"

                #Create record
                record = {
                    "id": story_id,
                    "person": pers,                    
                    "culture": cult,
                    "protagonist_gender": gen,
                    "model": info.get("name", "unknown"),
                    "provider": key,
                    "prompt": prompt,
                    "story": story,
                    "word_count": len(story.split()) if story not in ["[FAILED]", "[UNKNOWN PROVIDER]"] else 0,
                    "generated_at": datetime.utcnow().isoformat()
                }

                all_data.append(record)
                existing_ids.add(story_id)

                #Save individually
                safe_filename = "".join(c if c.isalnum() or c in "-_" else "_" for c in story_id)
                txt_path = INDIVIDUAL_DIR / f"{safe_filename}.txt"
                txt_path.write_text(
                    f"ID: {story_id}\n"
                    f"Person/Archetype: {pers}\n"
                    f"Culture: {cult}\n"
                    f"Gender: {gen}\n"
                    f"Model: {info.get('name', 'unknown')}\n\n"
                    f"--- PROMPT ---\n{prompt}\n\n"
                    f"--- STORY ---\n{story}",
                    encoding="utf-8"
                )

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

final_df.to_json(OUTPUT_DIR / "biasednarratives_free.jsonl", orient="records", lines=True, force_ascii=False)
final_df.to_csv(OUTPUT_DIR / "biasednarratives_free.csv", index=False)

print("\n=== GENERATION COMPLETE ===")
print(f"Total stories generated: {len(all_data)}")
print(f"Files saved in: {OUTPUT_DIR}")
print(f"Individual .txt stories: {INDIVIDUAL_DIR} ({len(list(INDIVIDUAL_DIR.glob('*.txt')))} files)")