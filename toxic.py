#Toxicity Detection
#Uses Detoxify (Hugging Face) for toxicity scores (hate, obscene, etc.)

from detoxify import Detoxify
import pandas as pd
from pathlib import Path


INPUT_CSV = Path("Narratives/clean_stories_for_analysis.csv")
OUTPUT_CSV = Path("Narratives/with_toxicity.csv")

#Load model (offline - small multilingual)
model = Detoxify('multilingual')  # or 'unbiased' for English


df = pd.read_csv(INPUT_CSV)

def compute_toxicity(text):
    results = model.predict(text)
    return results['toxicity']  # Main score (0-1, higher = more toxic)

print("Computing toxicity...")
df['toxicity'] = df['story'].apply(compute_toxicity)

#Summary
print("\nToxicity Summary (0-1, higher = more toxic):")
print(df['toxicity'].describe())

#Flag high toxicity
high_tox = df[df['toxicity'] > 0.5]  #Threshold based on papers
print(f"High toxicity stories: {len(high_tox)}")

#Save
df.to_csv(OUTPUT_CSV, index=False)
print(f"Saved to {OUTPUT_CSV}")