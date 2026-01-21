#Computes GAR (Gender Attribute Ratio) = appearance_words / (agency_words + 1) per story
#Higher GAR → more appearance focus (often bias toward female protagonists)

import pandas as pd
from pathlib import Path
import re
import matplotlib.pyplot as plt
import seaborn as sns


INPUT_CSV = Path("Narratives/clean_stories_for_analysis.csv")
OUTPUT_CSV = Path("Narratives/with_gar.csv")

#Define word lists
APPEARANCE_WORDS = {
    'beautiful', 'beauty', 'pretty', 'cute', 'lovely', 'gorgeous', 'attractive', 'charming',
    'elegant', 'delicate', 'graceful', 'grace', 'hair', 'eye', 'eyes', 'smile', 'dress',
    'skirt', 'princess', 'shiny', 'small', 'tiny', 'fragile', 'soft', 'gentle', 'fair'
}

AGENCY_WORDS = {
    'brave', 'bravery', 'strong', 'strength', 'smart', 'clever', 'intelligent', 'wise',
    'determined', 'determination', 'leader', 'lead', 'fight', 'win', 'save', 'protect',
    'decide', 'decision', 'build', 'create', 'explore', 'adventure', 'courage', 'powerful',
    'independent', 'bold', 'fearless'
}

#Load and compute GAR
print("Loading cleaned data...")
df = pd.read_csv(INPUT_CSV)

if 'tokens' not in df.columns:
    raise ValueError("DataFrame must have 'tokens' column (list of words)")

def compute_gar(tokens):
    if not isinstance(tokens, list) or not tokens:
        return 0.0
    
    tokens_lower = [t.lower() for t in tokens]
    
    app_count = sum(1 for w in tokens_lower if w in APPEARANCE_WORDS)
    agency_count = sum(1 for w in tokens_lower if w in AGENCY_WORDS)
    
    #Avoid division by zero
    gar = app_count / (agency_count + 1)
    #print(f"Story ID: {row.name} | appearance: {app_count}, agency: {agency_count}, GAR: {gar}")
    return round(gar, 4)

print("Computing GAR...")
df['gar'] = df['tokens'].apply(compute_gar)

#Optional: also save raw counts
df['appearance_count'] = df['tokens'].apply(
    lambda tks: sum(1 for w in [x.lower() for x in tks] if w in APPEARANCE_WORDS)
)
df['agency_count'] = df['tokens'].apply(
    lambda tks: sum(1 for w in [x.lower() for x in tks] if w in AGENCY_WORDS)
)

#Summary and Visualization
print("\nGAR Summary by Gender:")
gar_by_gender = df.groupby('protagonist_gender')['gar'].agg(['mean', 'count', 'std']).round(4)
print(gar_by_gender)

print("Global appearance words found:", df['appearance_count'].sum())
print("Global agency words found:", df['agency_count'].sum())

plt.figure(figsize=(8, 5))
sns.boxplot(x='protagonist_gender', y='gar', data=df)
plt.title('Gendered Attribute Ratio (GAR) by Protagonist Gender')
plt.ylabel('GAR (higher = more appearance focus)')
plt.xlabel('Protagonist Gender')
plt.tight_layout()
plt.savefig(INPUT_CSV.parent / 'gar_by_gender_boxplot.png')
plt.show()

#Save
df.to_csv(OUTPUT_CSV, index=False)
print(f"\nData with GAR saved to: {OUTPUT_CSV}")
print(f"Rows processed: {len(df)}")