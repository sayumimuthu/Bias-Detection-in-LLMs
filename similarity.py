#semantic_diversity_by_groups
#Computes average pairwise semantic similarity within groups defined by culture, person, protagonist_gender

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
from pathlib import Path
import itertools
import seaborn as sns
import matplotlib.pyplot as plt
import torch


INPUT_CSV = Path("Narratives/clean_stories_for_analysis.csv")
OUTPUT_CSV = Path("Narratives/with_group_similarity.csv")
OUTPUT_PLOT = Path("Narratives/similarity_heatmap.png")

#Load model 
model = SentenceTransformer('all-MiniLM-L6-v2')


df = pd.read_csv(INPUT_CSV)

#Ensure columns exist (adjust names if different)
group_cols = ['culture', 'person', 'protagonist_gender']
for col in group_cols:
    if col not in df.columns:
        print(f"Warning: Column '{col}' not found. Using available columns.")
        group_cols = [c for c in group_cols if c in df.columns]

print(f"Grouping by: {group_cols}")
print(f"Total stories: {len(df)}")

#Encode stories
print("Encoding all stories...")
embeddings = model.encode(
    df['story'].tolist(),
    batch_size=32,
    show_progress_bar=True,
    convert_to_tensor=True
)

df['embedding'] = list(embeddings.cpu().numpy())  #store for later use 

#Group Level Similarity
similarity_results = []

#All possible combinations of the three variables
for group_values, sub_df in df.groupby(group_cols):
    if len(sub_df) < 2:
        continue  # skip groups with <2 stories (can't compute pairwise sim)

    group_emb = torch.tensor(np.stack(sub_df['embedding'].values))
    
    #Compute pairwise cosine similarity
    sim_matrix = util.cos_sim(group_emb, group_emb)
    #Set diagonal to 0 (ignore self-similarity)
    sim_matrix.fill_diagonal_(0)
    #Average similarity (excluding self)
    avg_sim = sim_matrix.sum() / (len(sub_df) * (len(sub_df) - 1))
    
    similarity_results.append({
        **dict(zip(group_cols, group_values)),
        'group_size': len(sub_df),
        'avg_pairwise_similarity': float(avg_sim),
        'diversity_score': 1 - float(avg_sim)  #higher = more diverse
    })

#Convert to DataFrame
sim_df = pd.DataFrame(similarity_results)

#Overall similarity
overall_sim_matrix = util.cos_sim(embeddings, embeddings)
overall_sim_matrix.fill_diagonal_(0)
overall_avg = overall_sim_matrix.sum() / (len(df) * (len(df) - 1))
print(f"\nOverall average pairwise similarity: {overall_avg:.4f}")
print(f"Overall diversity score (1 - similarity): {1 - overall_avg:.4f}")

#Summary
print("\nTop 10 most similar groups (least diverse):")
print(sim_df.sort_values('avg_pairwise_similarity', ascending=False).head(10))

print("\nTop 10 most diverse groups:")
print(sim_df.sort_values('avg_pairwise_similarity', ascending=True).head(10))

#Grouped means
print("\nMean similarity by culture:")
print(sim_df.groupby('culture')['avg_pairwise_similarity'].mean().sort_values())

print("\nMean similarity by person (parent role):")
print(sim_df.groupby('person')['avg_pairwise_similarity'].mean().sort_values())

print("\nMean similarity by protagonist gender:")
print(sim_df.groupby('protagonist_gender')['avg_pairwise_similarity'].mean().sort_values())

#Visusalization 

if len(group_cols) >= 2:
    pivot = sim_df.pivot_table(
        index=group_cols[0],
        columns=group_cols[1],
        values='avg_pairwise_similarity',
        aggfunc='mean'
    )
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot, annot=True, cmap='YlOrRd', fmt='.3f')
    plt.title('Average Pairwise Semantic Similarity by Culture & Person')
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT)
    plt.show()
    print(f"Heatmap saved: {OUTPUT_PLOT}")

#Save results
sim_df.to_csv(OUTPUT_CSV, index=False)
print(f"\nGroup similarity results saved to: {OUTPUT_CSV}")