from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from gender_bias_analysis_new import fig_sem_model_dimension_heatmap  # noqa: E402

SEM_CSV = Path("Narratives3/bias_analysis/results_pmi/results/story_level_semantic.csv")
FIG_DIR = Path("Narratives3/bias_analysis/results_pmi/figures")

FIG_DIR.mkdir(parents=True, exist_ok=True)

print(f"Loading {SEM_CSV} …")
df = pd.read_csv(SEM_CSV)
print(f"  {len(df):,} rows, {df['model_key'].nunique()} models")

fig_sem_model_dimension_heatmap(df, FIG_DIR)
print(f"Done — figures written to {FIG_DIR}/")
