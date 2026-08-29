from pathlib import Path
import sys

import pandas as pd

# Allow importing from the project root regardless of cwd
sys.path.insert(0, str(Path(__file__).parent))
from gender_bias_analysis_new import fig_pmi_model_dimension_heatmap  # noqa: E402

# ── Paths ─────────────────────────────────────────────────────────────────────
PMI_CSV = Path("Narratives3/bias_analysis/results_pmi/results/story_level_pmi.csv")
FIG_DIR = Path("Narratives3/bias_analysis/pmi_results/figures")

FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Load & generate ───────────────────────────────────────────────────────────
print(f"Loading {PMI_CSV} …")
df = pd.read_csv(PMI_CSV)
print(f"  {len(df):,} rows, {df['model_key'].nunique()} models")

fig_pmi_model_dimension_heatmap(df, FIG_DIR)
print(f"Done — figures written to {FIG_DIR}/")