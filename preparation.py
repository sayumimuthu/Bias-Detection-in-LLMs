#Prepares clean DataFrame for bias analysis

import pandas as pd
import json
import os
from pathlib import Path
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
import string
import re
from datetime import datetime

import spacy
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])  # fast mode

#Download necessary NLTK data 
try:
    #Test if punkt_tab is available
    word_tokenize("This is a test.")
except LookupError as e:
    print("Downloading missing NLTK resources...")
    nltk.download('punkt_tab', quiet=False)
    nltk.download('punkt', quiet=False)  #for safety
    print("Resources downloaded successfully.")


INPUT_DIR = Path("Narratives")          
INPUT_FILE_JSONL = INPUT_DIR / "biasednarratives_free.jsonl"  
INPUT_FILE_CSV = INPUT_DIR / "biasednarratives_free.csv"      #backup option

OUTPUT_CLEAN_CSV = INPUT_DIR / "clean_stories_for_analysis.csv"
MIN_WORD_COUNT = 100          #minimum length to keep 
REMOVE_PATTERNS = ["[FAILED]", "[GENERATION FAILED]", "[UNKNOWN PROVIDER]"]

#Load data
print("Loading generated stories...")

if INPUT_FILE_JSONL.exists():
    df = pd.read_json(INPUT_FILE_JSONL, lines=True)
elif INPUT_FILE_CSV.exists():
    df = pd.read_csv(INPUT_FILE_CSV)
else:
    raise FileNotFoundError("No input file found! Check JSONL or CSV path.")

print(f"Original rows: {len(df)}")

#FILTER FAILED GENERATIONS    
#Remove rows with failed markers
mask_failed = df['story'].isin(REMOVE_PATTERNS) | \
              df['story'].str.contains(r'\[FAILED\]', na=False, regex=True)

#Remove very short / empty stories
mask_short = df['word_count'] < MIN_WORD_COUNT

#Remove NaN or empty strings
mask_empty = df['story'].isna() | (df['story'].str.strip() == "")

#Combine filters
df_clean = df[~(mask_failed | mask_short | mask_empty)].copy()

print(f"After filtering failed/short/empty: {len(df_clean)} rows kept "
      f"({len(df) - len(df_clean)} removed)")

#TOKENIZATION  
'''def clean_and_tokenize(text):
    if not isinstance(text, str) or not text.strip():
        return []
    
    #Basic cleaning
    text = re.sub(r'\s+', ' ', text.strip())                #normalize whitespace
    text = text.translate(str.maketrans('', '', string.punctuation))  #remove punctuation
    text = text.lower()                                     #lowercase for consistency
    
    #Tokenize
    tokens = word_tokenize(text)
    
    #Remove very short tokens (noise like 'a', 'the' can stay, but filter garbage)
    tokens = [t for t in tokens if len(t) > 1]
    
    return tokens'''

def clean_and_tokenize(text):
    if not isinstance(text, str) or not text.strip():
        return []
    
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc 
              if not token.is_punct and not token.is_stop and len(token.lemma_) > 1]
    return tokens

print("Tokenizing stories...")
df_clean['tokens'] = df_clean['story'].apply(clean_and_tokenize)
df_clean['token_count'] = df_clean['tokens'].apply(len)

# Optional: also add sentence tokens if needed later (e.g., for sentiment per sentence)
# df_clean['sentences'] = df_clean['story'].apply(sent_tokenize)

#FINAL CLEAN DATAFRAME
#Select and rename useful columns  
keep_columns = [
    'id', 'person', 'culture', 'protagonist_gender',
    'model', 'provider', 'prompt', 'story',
    'word_count', 'token_count', 'generated_at',
    'tokens'  #list of words
]

#If some columns are missing, ignore them
available_cols = [col for col in keep_columns if col in df_clean.columns]
df_final = df_clean[available_cols].copy()

#Add timestamp of cleaning
df_final['cleaned_at'] = datetime.utcnow().isoformat()

print("\nClean DataFrame summary:")
print(df_final.info())
print("\nSample:")
print(df_final.head(3))

#Save  
df_final.to_csv(OUTPUT_CLEAN_CSV, index=False)
print(f"\nClean dataset saved to: {OUTPUT_CLEAN_CSV}")
print(f"Ready rows for bias analysis: {len(df_final)}")