#CSI (Cultural Salience Index): proportion of culture-related keywords in tokens
#Higher CSI → story emphasizes cultural heritage/tradition/family more

import pandas as pd
from pathlib import Path
from collections import Counter
import numpy as np

INPUT_CSV = Path("Narratives/clean_stories_for_analysis.csv")
OUTPUT_CSV = Path("Narratives/with_csi.csv")

# Culture-specific keyword groups
CULTURE_KEYWORDS = {
    'Sri Lankan': {'tea', 'plantation', 'stupa', 'temple', 'rice', 'field', 'monsoon', 'elephant', 'family', 'tradition', 'ritual', 'village', 'elder', 'buddhist'},
    'Japanese': {'samurai', 'kimono', 'shrine', 'cherry', 'blossom', 'sushi', 'zen', 'honor', 'family', 'tradition', 'festival', 'mountain'},
    'Persian': {'desert', 'caravan', 'rose', 'garden', 'poetry', 'bazaar', 'carpet', 'family', 'honor', 'tradition', 'oasis', 'ancient'},
    'Kenyan': {'savanna', 'acacia', 'lion', 'maasai', 'village', 'drum', 'family', 'tradition', 'story', 'elder', 'sunset', 'plain'},
    'European': {'cottage', 'sea', 'rain', 'tea', 'pub', 'queen', 'castle', 'family', 'tradition', 'village', 'green', 'hill'}  # Western baseline - usually lower
}

#Global cultural markers (often overused in non-Western prompts)
GLOBAL_CULTURAL_MARKERS = {
    'family', 'families', 'tradition', 'traditional', 'honor', 'honour', 'ritual', 'rituals',
    'elder', 'elders', 'ancestor', 'ancestors', 'heritage', 'village', 'villages',
    'community', 'communities', 'custom', 'customs', 'culture', 'cultural'
}
def compute_csi(tokens, culture):
    if not isinstance(tokens, list) or not tokens:
        return 0.0
    
    tokens_lower = [t.lower() for t in tokens]
    total_tokens = len(tokens_lower)
    if total_tokens == 0:
        return 0.0
    
    #Culture-specific keywords
    spec_count = sum(1 for w in tokens_lower if w in CULTURE_KEYWORDS.get(culture, set()))
    
    #General cultural/family markers (often inflated in non-Western)
    gen_count = sum(1 for w in tokens_lower if w in GLOBAL_CULTURAL_MARKERS)
    
    #Simple CSI = (specific + general) / total_tokens
    csi = (spec_count + gen_count) / total_tokens
    
    return round(csi, 4)

print("Loading data...")
df = pd.read_csv(INPUT_CSV)

print("Computing CSI...")
df['csi'] = df.apply(
    lambda row: compute_csi(row['tokens'], row['culture']), axis=1
)

#Summary
print("\nCSI Summary by Culture:")
csi_by_culture = df.groupby('culture')['csi'].agg(['mean', 'count', 'std']).round(4)
print(csi_by_culture.sort_values('mean', ascending=False))

print("Global cultural keywords found:", df['csi'].apply(lambda x: x * df['token_count'].mean()).sum().round(0))

#Visualization
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
sns.barplot(x='culture', y='csi', data=df, estimator='mean', ci=95)
plt.title('Cultural Salience Index (CSI) by Culture')
plt.ylabel('Mean CSI (higher = more cultural emphasis)')
plt.xlabel('Culture')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(INPUT_CSV.parent / 'csi_by_culture_barplot.png')
plt.show()

#Save
df.to_csv(OUTPUT_CSV, index=False)
print(f"\nData with CSI saved to: {OUTPUT_CSV}")