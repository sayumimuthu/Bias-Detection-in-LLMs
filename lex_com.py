#Computes AoA (Average Age of Acquisition) and FKRE (Flesch-Kincaid Reading Ease)
#Higher AoA = more complex (adult-level words) and FKRE 0-100 (higher = easier)

import pandas as pd
from pathlib import Path
import textstat
import nltk
from nltk.corpus import cmudict

#Load Kuperman dataset 
kuperman_path = "aoa_words.xlsx"  
aoa_df = pd.read_excel(kuperman_path)
#Keep only Word and AoA_rating columns, make lowercase
aoa_dict = dict(zip(aoa_df['Word'].str.lower(), aoa_df['AoA_Kup_lem']))

#Download if needed
nltk.download('cmudict', quiet=True)

#Load CMUdict for syllables (used by textstat)
pronunciations = cmudict.dict()


INPUT_CSV = Path("Narratives/clean_stories_for_analysis.csv")
OUTPUT_CSV = Path("Narratives/with_complexity.csv")

#Load data
df = pd.read_csv(INPUT_CSV)

def compute_aoa(text):
    if not text:
        return 0.0
    
    words = text.lower().split()  
    aoa_scores = []
    
    for word in words:
        #Clean word of punctuation
        word_clean = word.strip(".,!?;:'\"")
        if word_clean in aoa_dict:
            aoa_scores.append(aoa_dict[word_clean])
    
    if aoa_scores:
        return round(sum(aoa_scores) / len(aoa_scores), 2)  #mean AoA in years
    return 0.0  #no known words

'''def compute_aoa(text):
    words = text.split()
    aoa_scores = []
    for word in words:
        # Simple AoA proxy: textstat uses Flesch-Kincaid base, but for true AoA, use average word length as proxy (or load Kuperman dataset)
        # Note: For accurate AoA, download Kuperman 2012 dataset CSV and average per word
        aoa_scores.append(textstat.lexicon_count(word, remove_punctuation=True) + len(word) / 10)  # Placeholder - replace with real AoA lookup
    return sum(aoa_scores) / len(aoa_scores) if aoa_scores else 0.0'''

def compute_fkscore(text):
    return textstat.flesch_reading_ease(text)

print("Computing complexity...")
df['aoa'] = df['story'].apply(compute_aoa)
df['fkre'] = df['story'].apply(compute_fkscore)

#Summary
print("\nAoA Summary (higher = more complex):")
print(df['aoa'].describe())

print("\nFKRE Summary (higher = easier):")
print(df['fkre'].describe())

#Save
df.to_csv(OUTPUT_CSV, index=False)
print(f"Saved to {OUTPUT_CSV}")