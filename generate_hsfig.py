from pathlib import Path
from matplotlib.patches import Rectangle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HS_DIR  = Path("Narratives3/hidden_states_by_model")
OUT_DIR = Path("Narratives3/hidden_state_surface_corr")
FIG_DPI = 150

OUT_DIR.mkdir(parents=True, exist_ok=True)


def _gender_axis_projection(hs_layer: np.ndarray,
                             is_female: np.ndarray) -> np.ndarray:
    female_vecs = hs_layer[is_female]
    male_vecs   = hs_layer[~is_female]
    axis = female_vecs.mean(0) - male_vecs.mean(0)
    norm = np.linalg.norm(axis)
    if norm < 1e-9:
        return np.zeros(len(hs_layer))
    axis /= norm
    return hs_layer @ axis


# Collect per-model mean projections 
records = []
for npy_path in sorted(HS_DIR.glob("hidden_states_*.npy")):
    mk        = npy_path.stem.replace("hidden_states_", "")
    meta_path = HS_DIR / f"extractor_meta_{mk}.csv"
    if not meta_path.exists():
        print(f"  [{mk}] skipped — no metadata CSV")
        continue

    hs   = np.load(npy_path)                     # (n_stories, n_layers, hidden_dim)
    meta = pd.read_csv(meta_path)
    print(f"  {mk}: {hs.shape}")

    is_female = (meta["protagonist_gender"] == "female").values
    if is_female.sum() < 2 or (~is_female).sum() < 2:
        print(f"  [{mk}] skipped — too few stories per gender")
        continue

    proj = _gender_axis_projection(hs[:, -1, :], is_female)
    records.append({
        "model_key": mk,
        "f_proj":    float(proj[is_female].mean()),
        "m_proj":    float(proj[~is_female].mean()),
    })

if not records:
    raise RuntimeError(f"No valid models found in {HS_DIR}")

df           = pd.DataFrame(records).sort_values("model_key").reset_index(drop=True)
models       = df["model_key"].tolist()
model_labels = [m.replace("ollama-", "") for m in models]
n_models     = len(models)
fem_vals     = df["f_proj"].values
masc_vals    = df["m_proj"].values

print(f"\n  {n_models} models loaded")

# Colour normalization 
EPS = 1e-9
f_min, f_max = fem_vals.min(),  fem_vals.max()
m_min, m_max = masc_vals.min(), masc_vals.max()

def _nf(v):
    return (v - f_min) / (f_max - f_min + EPS)

def _nm(v):                        # more negative -> 1 (darkest orange)
    return 1.0 - (v - m_min) / (m_max - m_min + EPS)

green_cmap  = plt.cm.Greens
orange_cmap = plt.cm.Oranges
ILO, IHI    = 0.20, 0.82

# Layout 
cw    = 1.55
ch    = 0.46
lm    = 3.00
pad_b = 0.18
fm_h  = 0.36
dim_h = 0.44
pad_t = 0.12

content_h = n_models * ch
fig_h = pad_b + content_h + fm_h + dim_h + pad_t
fig_w = lm + 2 * cw + 0.22

fig, ax = plt.subplots(figsize=(fig_w, fig_h))
ax.set_xlim(0, fig_w)
ax.set_ylim(0, fig_h)
ax.set_aspect("auto")
ax.axis("off")

xf = lm
xm = lm + cw

# Draw cells 
for i, mlabel in enumerate(model_labels):
    y = pad_b + (n_models - 1 - i) * ch

    ax.text(lm - 0.10, y + ch / 2, mlabel,
            ha="right", va="center", fontsize=16, fontweight="bold", color="black")

    fi  = ILO + (IHI - ILO) * _nf(fem_vals[i])
    fc  = green_cmap(fi)
    ax.add_patch(Rectangle((xf, y), cw, ch,
                            facecolor=fc, edgecolor="white", linewidth=0.5))
    ax.text(xf + cw / 2, y + ch / 2 - 0.03,
            f"{fem_vals[i]:.3f}".replace("-", "- "),
            ha="center", va="center", fontsize=16, fontweight="bold",
            color="white" if fi > 0.58 else "black")

    mi  = ILO + (IHI - ILO) * _nm(masc_vals[i])
    mc  = orange_cmap(mi)
    ax.add_patch(Rectangle((xm, y), cw, ch,
                            facecolor=mc, edgecolor="white", linewidth=0.5))
    ax.text(xm + cw / 2, y + ch / 2 - 0.03,
            f"{masc_vals[i]:.3f}".replace("-", "- "),
            ha="center", va="center", fontsize=16, fontweight="bold",
            color="white" if mi > 0.58 else "black")

# F / M labels 
y_fm = pad_b + content_h + 0.04
ax.text(xf + cw / 2, y_fm + fm_h / 2, "F",
        ha="center", va="center", fontsize=16, fontweight="bold", color="black")
ax.text(xm + cw / 2, y_fm + fm_h / 2, "M",
        ha="center", va="center", fontsize=16, fontweight="bold", color="black")

# Dimension header 
y_dim = y_fm + fm_h + 0.04
x_mid = (xf + xm + cw) / 2
ax.text(x_mid, y_dim + dim_h / 2, "Gender Projection",
        ha="center", va="center", fontsize=16, fontweight="bold")
ax.plot([xf, xm + cw], [y_dim, y_dim], color="black", linewidth=0.9)

fig.suptitle(
    "Hidden-State Gender-Axis Projection by Model\n"
    "(F = daughter stories · M = son stories · final transformer layer"
    " · darker = stronger alignment)",
    fontsize=9.5, fontweight="bold",
)
fig.subplots_adjust(top=0.94, bottom=0.02, left=0.02, right=0.98)

stem = OUT_DIR / "fig_hs_projection_heatmap"
for ext in ("png", "pdf"):
    fig.savefig(f"{stem}.{ext}", dpi=FIG_DPI, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved {stem}.png / .pdf")
