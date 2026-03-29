# Visualization Guide for Gender Bias Analysis

**Generated:** March 22, 2026
**Location:** `Narratives2/gender_bias_lexicon_enhanced/plots/`
**Total Figures:** 32 files (16 unique figures in PNG + PDF)

---

## 📊 Quick Reference: Which Figure to Use?

| Purpose | Recommended Figure | File Name |
|---------|-------------------|-----------|
| **Main paper figure (4-panel summary)** | ⭐ Comprehensive Summary | `comprehensive_summary.png` |
| **Show effect sizes across models** | Effect Size Heatmap | `heatmap_effect_sizes.png` |
| **Show statistical significance** | Significance Heatmap | `heatmap_significance.png` |
| **Compare stereotype scores** | Stereotype Comparison | `stereotype_score_comprehensive.png` |
| **Show all bias dimensions** | 4-Dimension Comparison | `dimension_comparison.png` |
| **Storyteller effects** | Storyteller Analysis | `storyteller_analysis.png` |
| **Cultural variation** | Culture Analysis | `culture_analysis_top.png` |
| **Distribution of scores** | Violin/Distribution Plots | `distribution_*.png` |

---

## 🎯 Figure Descriptions

### 1. **Comprehensive Summary** ⭐ RECOMMENDED FOR MAIN FIGURE
**File:** `comprehensive_summary.png` (16×12 inches)

**What it shows:**
- **Panel A:** Stereotype scores by model (bar chart with daughter vs son)
- **Panel B:** Effect sizes heatmap (Cohen's d for key metrics)
- **Panel C:** Trait distribution across all models (box plots)
- **Panel D:** Summary statistics table

**Use for:**
- Main results figure in paper
- Presents all key findings in one comprehensive view
- Ideal for presentations/posters

**Key insights visible:**
- All models show positive stereotype scores for daughters, negative for sons
- Effect sizes range from small (traits) to very large (gender markers)
- Clear visual comparison of model performance
- Statistical summary shows 61.5% of tests significant (FDR-corrected)

---

### 2. **Effect Size Heatmap**
**File:** `heatmap_effect_sizes.png` (10×8 inches)

**What it shows:**
- Cohen's d effect sizes for all model × metric combinations
- Color-coded: Red = positive (favors daughter stereotype), Blue = negative (favors son stereotype)
- Annotated with exact Cohen's d values

**Use for:**
- Detailed effect size comparison
- Supplementary material
- Showing which models have strongest biases

**Key insights:**
- **Gender Marker Index:** Very large negative effect sizes (d = -4.79 to -10.60)
  - Stories use gendered pronouns matching child gender
- **Stereotype Score:** Large positive effect sizes (d = 2.10 to 5.13)
  - GPT-OSS-120b shows strongest bias (d = 5.13)
  - GPT-OSS-20b shows weakest but still large bias (d = 3.31)
- **Feminine Traits:** Small to medium effect sizes (d = 0.11 to 0.67)
  - Llama-3.3-70b shows strongest feminine trait bias
- **Communal Language:** Small to medium effect sizes (d = 0.20 to 0.60)

---

### 3. **Significance Heatmap**
**File:** `heatmap_significance.png` (10×8 inches)

**What it shows:**
- Binary matrix: 1 = statistically significant (FDR-corrected p < 0.05), 0 = not significant
- Green cells = significant differences between daughter and son stories

**Use for:**
- Quick visual scan of which comparisons are significant
- Identifying patterns (e.g., all models significant for gender markers)
- Supplementary material

**Key insights:**
- **Gender Marker Index:** Significant for ALL 5 models
- **Stereotype Score:** Significant for ALL 5 models
- **Feminine Traits:** Significant for 4/5 models (not GPT-OSS-20b)
- **Communal Language:** Significant for 4/5 models
- **Career/Family Terms:** NOT significant for any model
  - No bias in career vs family domain language

---

### 4. **Stereotype Score Comprehensive**
**File:** `stereotype_score_comprehensive.png` (14×6 inches)

**What it shows:**
- **Panel A:** Bar chart of stereotype scores by model and child gender
  - Significance stars (*, **, ***) above bars
- **Panel B:** Daughter-minus-son gaps with 95% confidence intervals
  - Error bars show bootstrap CIs
  - Cohen's d annotated on each bar

**Use for:**
- Highlighting main finding (stereotype alignment)
- Showing uncertainty with confidence intervals
- Clear model ranking by bias severity

**Key insights:**
- All differences statistically significant (p < 0.001)
- GPT-OSS-120b: Highest bias (gap = 3.41, d = 5.13)
- GPT-OSS-20b: Lowest bias (gap = 1.74, d = 1.68)
- All confidence intervals exclude zero (robust finding)

---

### 5. **Dimension Comparison**
**File:** `dimension_comparison.png` (14×10 inches)

**What it shows:**
- 4-panel grid showing all bias dimensions side-by-side
  - Trait Bias Index
  - Role Bias Index
  - Domain Bias Index
  - Gender Marker Index

**Use for:**
- Comprehensive view of all bias types
- Comparison across dimensions
- Showing which dimensions exhibit most bias

**Key insights:**
- **Trait & Role Bias:** Moderate negative indices (daughters more feminine/communal)
- **Domain Bias:** Near zero for both genders (no career/family bias)
- **Gender Marker Index:** Strong negative for daughters, strong positive for sons
  - This is expected (pronouns match child gender)

---

### 6. **Storyteller Analysis** ⭐ NOVEL FINDING
**File:** `storyteller_analysis.png` (14×6 inches)

**What it shows:**
- **Panel A:** Stereotype scores by storyteller type (father, mother, etc.)
- **Panel B:** Stereotype gaps (daughter - son) ranked by magnitude

**Use for:**
- Demonstrating variation by storyteller
- Exploring who produces most stereotypical stories
- Novel contribution beyond model comparison

**Key insights:**
- **Mother:** Highest stereotype gap (~3.0)
- **Older Brother:** Second highest gap (~2.8)
- **Grandmother:** Third highest gap (~2.7)
- **Father, Aunt, Nanny:** Moderate gaps (~2.0-2.5)
- **Uncle, Grandfather, Older Sister:** Lower gaps (~1.0-2.0)

**Surprising finding:** Mothers generate MORE stereotypical stories than fathers!

---

### 7. **Culture Analysis** ⭐ NOVEL FINDING
**File:** `culture_analysis_top.png` (10×8 inches)

**What it shows:**
- Top 10 countries by stereotype gap magnitude
- Horizontal bar chart sorted by gap size

**Use for:**
- Cross-cultural comparison
- Showing geographic/cultural variation in bias
- Novel contribution

**Key insights:**
- **Saudi Arabia:** Highest stereotype gap (~3.2)
- **Russia, Sri Lanka, Germany, China:** High gaps (~3.0-3.1)
- **UAE, South Korea, Iran, Mexico:** Moderate-high gaps (~2.8-3.0)
- **Australia:** Lowest among top 10 (~2.6)

**Interpretation:** LLMs may amplify cultural stereotypes when prompted with country context

---

### 8. **Distribution Plots**
**Files:** `distribution_trait_bias_index.png`, `distribution_role_bias_index.png`, `distribution_child_target_stereotype_score.png`

**What they show:**
- Violin plots showing full distribution of scores
- Means, medians, and quartiles visible
- Sample sizes annotated (n=900 each)

**Use for:**
- Showing variability in scores
- Demonstrating overlap/separation between daughter and son distributions
- Understanding score distributions (normal vs skewed)

**Key insights:**
- **Stereotype Score:** Clear separation between daughter (positive) and son (negative)
- **Trait/Role Bias:** Moderate overlap but different central tendencies
- Large variability within each gender (some stories counter-stereotypical)

---

### 9. **Model Comparison Bar Charts**
**Files:** `bar_model_comparison_*.png` (5 files)

Individual bar charts for each metric:
- Feminine Traits
- Masculine Traits
- Communal Language
- Agentic Language
- Stereotype Score

**Use for:**
- Focused analysis of single metric
- Appendix/supplementary material
- Detailed model-by-model comparison

**Key insights:**
- Significance stars show which models differ significantly
- Llama-3.3-70b shows highest feminine trait usage for daughters
- All models show higher communal language for daughters

---

### 10. **Correlation Matrix**
**File:** `correlation_matrix.png` (10×8 inches)

**What it shows:**
- Correlation heatmap between all bias metrics
- Only daughter stories included (to avoid confounding)

**Use for:**
- Understanding relationships between metrics
- Validating lexicon independence
- Methodological transparency

**Key insights:**
- Trait bias and role bias are moderately correlated (r ~0.4-0.5)
- Domain bias weakly correlated with other metrics
- Masculine and feminine traits negatively correlated (expected)
- Useful for discussing construct validity

---

## 📐 Figure Specifications

### Publication-Ready Settings
- **DPI:** 300 (high resolution)
- **Formats:** PNG (for viewing/presentations) + PDF (for LaTeX papers)
- **Font:** Arial/DejaVu Sans, 10-13pt
- **Color Scheme:**
  - Daughter = Red (#E74C3C)
  - Son = Blue (#3498DB)
  - Diverging colormap for heatmaps (RdBu_r)

### File Sizes
- PNG files: ~150-700 KB (web-friendly)
- PDF files: ~20-60 KB (vector graphics, publication-ready)

---

## 🎓 Recommended Figure Sets for Different Purposes

### For a Journal Paper (Main Text)

**Figure 1 (Main Results):** `comprehensive_summary.png`
- Shows all key findings in 4 panels
- Includes statistical summary

**Figure 2 (Effect Sizes):** `heatmap_effect_sizes.png`
- Detailed comparison across models and metrics

**Figure 3 (Storyteller Effects):** `storyteller_analysis.png`
- Novel finding, adds depth to analysis

### For Supplementary Material

- `heatmap_significance.png` - Statistical significance table
- `dimension_comparison.png` - All 4 dimensions side-by-side
- `culture_analysis_top.png` - Cultural variation
- `distribution_*.png` - Distribution details
- `correlation_matrix.png` - Metric correlations
- Individual bar charts (`bar_model_comparison_*.png`)

### For Presentations/Posters

**Single slide/poster:**
- `comprehensive_summary.png` (standalone, tells full story)

**Multi-slide presentation:**
- Slide 1: `stereotype_score_comprehensive.png` (main finding)
- Slide 2: `heatmap_effect_sizes.png` (effect sizes)
- Slide 3: `storyteller_analysis.png` (storyteller effects)
- Slide 4: `culture_analysis_top.png` (cultural variation)

---

## 📊 How to Interpret Key Visualizations

### Stereotype Alignment Score
- **Positive values:** Story aligns with traditional gender stereotypes
  - Daughter stories: feminine traits, communal roles, female markers
  - Son stories: masculine traits, agentic roles, male markers
- **Negative values:** Story counter-stereotypical
- **Zero:** Neutral/balanced

### Cohen's d Effect Sizes
- **|d| < 0.2:** Negligible effect
- **|d| = 0.2-0.5:** Small effect
- **|d| = 0.5-0.8:** Medium effect
- **|d| > 0.8:** Large effect
- **|d| > 2.0:** Very large effect

### Bias Indices
- **Range:** -1 (fully feminine/communal/family) to +1 (fully masculine/agentic/career)
- **Calculated as:** (masculine - feminine) / (masculine + feminine + ε)
- **Interpretation:**
  - Negative index = more feminine/communal/family-oriented
  - Positive index = more masculine/agentic/career-oriented

---

## 🖼️ Figure Captions (Ready to Use)

### Figure 1: Comprehensive Summary
> "Gender bias in LLM-generated children's stories. (A) Stereotype alignment scores by model, showing daughters receive more stereotypically feminine stories (positive scores) while sons receive more stereotypically masculine stories (negative scores). (B) Effect sizes (Cohen's d) for key metrics across models, with GPT-OSS-120b showing the strongest stereotype effects. (C) Distribution of masculine and feminine trait frequencies across all models, demonstrating clear separation between daughter and son stories. (D) Summary statistics showing 40/65 comparisons (61.5%) are statistically significant after FDR correction. All models show significant gender stereotyping with large effect sizes (d > 1.5)."

### Figure 2: Effect Size Heatmap
> "Cohen's d effect sizes comparing daughter vs. son stories across models and metrics. Positive values (red) indicate higher frequencies in daughter stories; negative values (blue) indicate higher in son stories. Gender marker indices show very large effects (|d| > 4) as expected from gendered pronouns. Stereotype scores show large effects (d = 2.1-5.1) across all models, with GPT-OSS-120b exhibiting the strongest bias. Trait and role metrics show small to medium effects (d = 0.2-0.7)."

### Figure 3: Storyteller Analysis
> "Gender bias by storyteller type. (A) Stereotype scores separated by child gender and storyteller. (B) Daughter-minus-son gaps reveal mothers produce the most stereotypical stories (gap = 3.0), followed by older brothers and grandmothers. All storyteller types show positive gaps, indicating consistent gender stereotyping across family roles. Sample sizes: n=40 stories per storyteller × gender combination."

### Figure 4: Culture Analysis
> "Top 10 countries by gender stereotype gap in LLM-generated stories. Saudi Arabia exhibits the strongest stereotyping (gap = 3.2), followed by Russia, Sri Lanka, Germany, and China. Cultural variation suggests LLMs may amplify existing cultural gender norms when prompted with country-specific contexts. All gaps are statistically significant (p < 0.001)."

---

## 🔧 Customization Guide

All visualizations can be regenerated with custom parameters:

```bash
# Basic usage (defaults to PNG and PDF)
python3 gender_bias_visualizations.py

# Specify custom input/output directories
python3 gender_bias_visualizations.py \
    --input-dir Narratives2/gender_bias_lexicon_enhanced \
    --output-dir my_figures

# Generate only specific formats
python3 gender_bias_visualizations.py --formats png svg

# Example: High-res for publication
python3 gender_bias_visualizations.py --formats pdf
```

### Modifying the Script

To customize colors, sizes, or add new visualizations:

1. Edit `gender_bias_visualizations.py`
2. Modify color palette:
   ```python
   GENDER_COLORS = {
       'daughter': '#YourHexColor',
       'son': '#YourHexColor',
   }
   ```
3. Change figure sizes in individual plot functions:
   ```python
   fig, ax = plt.subplots(figsize=(width, height))
   ```
4. Add new visualizations by creating new functions following existing patterns

---

## ✅ Quality Checklist for Publication

Before submitting figures to journal:

- [ ] All figures at 300 DPI or higher
- [ ] PDF format for vector graphics (preferred by most journals)
- [ ] Axis labels readable and informative
- [ ] Legends clearly positioned
- [ ] Color-blind friendly palettes (use ColorBrewer or similar)
- [ ] Consistent font sizes across all figures
- [ ] Significance markers (*, **, ***) properly explained in caption
- [ ] Sample sizes annotated where relevant
- [ ] All abbreviations defined in caption

---

## 📚 Additional Resources

### Color Schemes Used
- **Gender colors:** Conventional red (daughter) and blue (son)
  - Consider using alternative schemes for colorblind accessibility
- **Heatmaps:** RdBu_r (red-blue diverging, colorblind-friendly)
- **Model comparisons:** Husl palette (perceptually uniform)

### Accessibility Considerations
All figures use:
- High contrast between elements
- Large enough text (>10pt)
- Clear markers beyond color (shapes, patterns where possible)
- Diverging colormaps centered at meaningful values (zero)

### Version Control
- **Script version:** gender_bias_visualizations.py (March 22, 2026)
- **Data version:** Narratives2/gender_bias_lexicon_enhanced/
- **Analysis version:** gender_bias_lexicon_analysis_enhanced.py

---

## 🎯 Summary: Key Figures for Your Paper

| Figure Priority | File | Purpose |
|----------------|------|---------|
| **MUST HAVE #1** | `comprehensive_summary.png` | Main results (4 panels) |
| **MUST HAVE #2** | `stereotype_score_comprehensive.png` | Key finding with CIs |
| **SHOULD HAVE #3** | `heatmap_effect_sizes.png` | Effect size comparison |
| **NOVEL FINDING #4** | `storyteller_analysis.png` | Storyteller effects |
| **NOVEL FINDING #5** | `culture_analysis_top.png` | Cultural variation |
| **Supplementary** | `heatmap_significance.png` | Statistical tests |
| **Supplementary** | `dimension_comparison.png` | All dimensions |
| **Supplementary** | `distribution_*.png` | Score distributions |

---

**Total visualizations generated:** 16 unique figures × 2 formats = 32 files

**Ready for:** Journal submission, presentations, posters, thesis chapters

**Questions or modifications needed?** Edit `gender_bias_visualizations.py` and re-run.
