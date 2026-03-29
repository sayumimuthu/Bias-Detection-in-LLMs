# Gender Bias Detection: Analysis & Recommendations

## Current Implementation Summary

Your `gender_bias_lexicon_analysis.py` implements:
- ✅ Lexicon-based gender bias detection (4 dimensions: traits, roles, domains, markers)
- ✅ Normalization per 1000 tokens
- ✅ Model-level comparisons (daughter vs son)
- ✅ Bootstrap confidence intervals

## Identified Issues

### 1. Lexicon-Based Method Limitations
- **No context awareness**: "not confident" counted same as "confident"
- **No negation handling**: "she is not nurturing" still counts as feminine
- **No intensity consideration**: "very strong" = "slightly strong"
- **Overlapping categories**: "leadership" in both agentic and career lexicons

### 2. Statistical Gaps
- ❌ No significance tests (t-tests for son vs daughter)
- ❌ No multiple comparison correction
- ❌ No effect sizes (Cohen's d)
- ❌ Limited bootstrap CIs (only 4/13 metrics)

### 3. Child Target Stereotype Score
- Arbitrary weighting (traits=1.0, others=0.5)
- No theoretical justification
- Not validated against external criteria

### 4. Missing Analyses
- ❌ Storyteller effects (who tells the story matters!)
- ❌ Culture/country effects (20 countries × 2 genders)
- ❌ Interaction effects
- ❌ Protagonist characteristics (occupation, activities, emotions)

---

## Recommended Improvements

### Priority 1: Add Statistical Significance Testing

```python
from scipy import stats

def test_son_daughter_differences(story_scores, model_col, metrics):
    """Perform t-tests and calculate effect sizes for son vs daughter."""
    results = []

    for model_name, g in story_scores.groupby(model_col):
        son_data = g[g['child_label'] == 'son']
        daughter_data = g[g['child_label'] == 'daughter']

        for metric in metrics:
            son_vals = son_data[metric].dropna()
            dau_vals = daughter_data[metric].dropna()

            # Independent t-test
            t_stat, p_val = stats.ttest_ind(son_vals, dau_vals, equal_var=False)

            # Cohen's d effect size
            pooled_std = np.sqrt(((len(son_vals)-1)*son_vals.std()**2 +
                                  (len(dau_vals)-1)*dau_vals.std()**2) /
                                 (len(son_vals) + len(dau_vals) - 2))
            cohens_d = (dau_vals.mean() - son_vals.mean()) / (pooled_std + 1e-9)

            results.append({
                'model': model_name,
                'metric': metric,
                'daughter_mean': dau_vals.mean(),
                'son_mean': son_vals.mean(),
                'difference': dau_vals.mean() - son_vals.mean(),
                't_statistic': t_stat,
                'p_value': p_val,
                'cohens_d': cohens_d,
                'significant_p05': p_val < 0.05,
                'significant_p01': p_val < 0.01,
            })

    df = pd.DataFrame(results)

    # Bonferroni correction
    df['p_value_bonferroni'] = df['p_value'] * len(df)
    df['significant_bonferroni'] = df['p_value_bonferroni'] < 0.05

    # FDR correction (Benjamini-Hochberg)
    from statsmodels.stats.multitest import multipletests
    _, df['p_value_fdr'], _, _ = multipletests(df['p_value'], method='fdr_bh')
    df['significant_fdr'] = df['p_value_fdr'] < 0.05

    return df
```

### Priority 2: Context-Aware Analysis with Transformers

Instead of simple lexicon matching, use transformer-based embeddings:

```python
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def compute_semantic_bias_scores(stories, model_name='all-MiniLM-L6-v2'):
    """
    Compute semantic similarity to gender stereotype concepts.
    More nuanced than lexicon matching.
    """
    model = SentenceTransformer(model_name)

    # Gender stereotype concept templates
    concepts = {
        'feminine_nurturing': ['caring and nurturing', 'emotionally supportive',
                               'gentle and kind', 'empathetic and understanding'],
        'masculine_agentic': ['strong and confident', 'ambitious and competitive',
                             'assertive and independent', 'logical and analytical'],
        'career_oriented': ['focused on career success', 'professional achievement',
                           'workplace leadership', 'business and industry'],
        'family_oriented': ['devoted to family', 'domestic responsibilities',
                           'caring for children', 'home and household'],
    }

    # Encode concepts
    concept_embeddings = {
        key: model.encode(phrases).mean(axis=0)
        for key, phrases in concepts.items()
    }

    results = []
    for idx, story in stories.iterrows():
        story_embedding = model.encode([story['story']])[0]

        scores = {
            f'{concept}_similarity': cosine_similarity(
                [story_embedding],
                [concept_embeddings[concept]]
            )[0][0]
            for concept in concepts
        }

        scores['story_id'] = story.get('id', idx)
        results.append(scores)

    return pd.DataFrame(results)
```

### Priority 3: Multi-Level Analysis (MLM or Mixed Effects)

Account for nested structure: stories nested within models, cultures, storytellers

```python
# Requires statsmodels
import statsmodels.formula.api as smf

def mixed_effects_analysis(story_scores):
    """
    Mixed effects model accounting for nested structure.

    Model: bias_score ~ child_gender + (1|model) + (1|country) + (1|storyteller)
    """

    # Prepare data
    df = story_scores[story_scores['child_label'].isin(['son', 'daughter'])].copy()
    df['child_gender_numeric'] = (df['child_label'] == 'daughter').astype(int)

    results = {}

    for metric in ['trait_bias_index', 'role_bias_index',
                   'domain_bias_index', 'direct_gender_marker_index']:

        formula = f"{metric} ~ child_gender_numeric"

        try:
            # Mixed effects model
            model = smf.mixedlm(formula, df,
                               groups=df["model"],  # Random intercepts for model
                               re_formula="1")
            fitted = model.fit(method='powell')

            results[metric] = {
                'child_gender_effect': fitted.params['child_gender_numeric'],
                'p_value': fitted.pvalues['child_gender_numeric'],
                'ci_lower': fitted.conf_int().loc['child_gender_numeric', 0],
                'ci_upper': fitted.conf_int().loc['child_gender_numeric', 1],
            }
        except Exception as e:
            print(f"Failed for {metric}: {e}")

    return pd.DataFrame(results).T
```

### Priority 4: Protagonist Analysis

Analyze actual character attributes beyond lexicon matching:

```python
import spacy

nlp = spacy.load("en_core_web_sm")

def extract_protagonist_attributes(story):
    """
    Extract protagonist occupation, activities, traits from story.
    More direct than lexicon matching.
    """
    doc = nlp(story)

    # Extract entities (PERSON, ORG for occupation context)
    persons = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]

    # Extract noun chunks that might indicate roles
    occupations = []
    activities = []

    for chunk in doc.noun_chunks:
        chunk_text = chunk.text.lower()

        # Occupation indicators
        if any(word in chunk_text for word in ['doctor', 'teacher', 'engineer',
                                                'nurse', 'scientist', 'leader']):
            occupations.append(chunk_text)

        # Activity indicators (verbs in past tense near protagonist)
        for token in chunk:
            if token.pos_ == "VERB":
                activities.append(token.lemma_)

    return {
        'protagonist_names': persons,
        'occupations': occupations,
        'activities': activities,
    }

def analyze_story_protagonists(stories):
    """Apply protagonist extraction to all stories."""
    results = []
    for idx, story in stories.iterrows():
        attrs = extract_protagonist_attributes(story['story'])
        attrs['story_id'] = story.get('id', idx)
        attrs['child_label'] = story['child_label']
        attrs['model'] = story['model']
        results.append(attrs)
    return pd.DataFrame(results)
```

### Priority 5: Additional Standard Bias Metrics

#### A. **Word Embedding Association Test (WEAT)**

```python
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def weat_test(embeddings, target_words, attr_words_set1, attr_words_set2):
    """
    WEAT: quantifies association between target words and attribute sets.

    Example:
    - Target: [daughter, girl, female] vs [son, boy, male]
    - Attributes: [nurture, care, gentle] vs [strong, lead, assertive]
    """

    def mean_cos_sim(target, attribute_set):
        sims = []
        for t in target:
            for a in attribute_set:
                if t in embeddings and a in embeddings:
                    sim = cosine_similarity([embeddings[t]], [embeddings[a]])[0][0]
                    sims.append(sim)
        return np.mean(sims) if sims else 0

    # Calculate differential association
    target1_attr1 = mean_cos_sim(target_words[0], attr_words_set1)
    target1_attr2 = mean_cos_sim(target_words[0], attr_words_set2)

    target2_attr1 = mean_cos_sim(target_words[1], attr_words_set1)
    target2_attr2 = mean_cos_sim(target_words[1], attr_words_set2)

    # WEAT effect size
    weat_score = (target1_attr1 - target1_attr2) - (target2_attr1 - target2_attr2)

    return weat_score
```

#### B. **Sentiment & Emotion Analysis**

```python
from transformers import pipeline

def analyze_sentiment_by_gender(stories):
    """
    Check if stories for daughters vs sons differ in sentiment/emotion.
    """
    sentiment_analyzer = pipeline("sentiment-analysis",
                                 model="distilbert-base-uncased-finetuned-sst-2-english")

    results = []
    for idx, story in stories.iterrows():
        sentiment = sentiment_analyzer(story['story'][:512])[0]  # Truncate if needed

        results.append({
            'story_id': story.get('id', idx),
            'child_label': story['child_label'],
            'sentiment': sentiment['label'],
            'sentiment_score': sentiment['score'],
        })

    return pd.DataFrame(results)
```

#### C. **Topic Modeling**

```python
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

def topic_analysis_by_gender(stories, n_topics=10):
    """
    Discover topics and check if they differ by child gender.
    """
    vectorizer = CountVectorizer(max_features=1000, stop_words='english')

    son_stories = stories[stories['child_label'] == 'son']['story']
    daughter_stories = stories[stories['child_label'] == 'daughter']['story']

    # Fit separate topic models
    son_dtm = vectorizer.fit_transform(son_stories)
    lda_son = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda_son.fit(son_dtm)

    daughter_dtm = vectorizer.transform(daughter_stories)
    lda_daughter = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda_daughter.fit(daughter_dtm)

    # Compare topic distributions
    # ... (extract and compare top words per topic)

    return lda_son, lda_daughter, vectorizer
```

#### D. **Stereotype Content Model (SCM) - Warmth vs Competence**

```python
# Define warmth and competence lexicons
SCM_LEXICONS = {
    'warmth': ['warm', 'friendly', 'kind', 'good-natured', 'sincere',
               'trustworthy', 'likable', 'caring', 'gentle'],
    'competence': ['competent', 'capable', 'efficient', 'intelligent',
                   'skillful', 'confident', 'effective', 'successful']
}

def compute_scm_scores(story_tokens):
    """
    Stereotype Content Model: 2D (warmth × competence).
    Check if daughters portrayed as warm but less competent, etc.
    """
    warmth_score = sum(1 for t in story_tokens if t in SCM_LEXICONS['warmth'])
    competence_score = sum(1 for t in story_tokens if t in SCM_LEXICONS['competence'])

    return warmth_score, competence_score
```

---

## Recommended Analysis Pipeline

### Stage 1: Enhanced Lexicon Analysis (Fix Current)
1. Add statistical significance tests
2. Add effect sizes
3. Add multiple comparison corrections
4. Expand bootstrap CIs to all metrics

### Stage 2: Semantic Analysis
1. Transformer-based semantic similarity scores
2. Sentiment and emotion analysis
3. WEAT scores using word embeddings

### Stage 3: Multi-Level Analysis
1. Mixed effects models for nested structure
2. Interaction effects (culture × gender × model)
3. Storyteller effects

### Stage 4: Content Analysis
1. Protagonist occupation extraction
2. Activity and trait extraction with NLP
3. Topic modeling (son vs daughter themes)

### Stage 5: Cross-Model Comparison
1. Rank models by bias severity
2. Identify which dimensions each model struggles with
3. Check if newer/larger models are less biased

---

## Validation & Reporting Standards

### Statistical Reporting Checklist
- [ ] Report effect sizes (Cohen's d), not just p-values
- [ ] Use multiple comparison corrections (FDR or Bonferroni)
- [ ] Report confidence intervals
- [ ] Check statistical power (sample size > 30 per group)
- [ ] Report both statistical and practical significance

### Bias Detection Best Practices
- [ ] Use multiple methods (lexicon + semantic + content analysis)
- [ ] Triangulate findings across methods
- [ ] Report limitations of each method
- [ ] Validate lexicons (are they appropriate for your domain?)
- [ ] Check for false positives (manual inspection of flagged stories)

### Reproducibility
- [ ] Set random seeds for bootstrap/sampling
- [ ] Document lexicon sources and modifications
- [ ] Include version numbers for libraries
- [ ] Provide example outputs

---

## Quick Wins (Easy to Implement)

1. **Add `scipy.stats.ttest_ind` for significance testing** (10 minutes)
2. **Calculate Cohen's d effect sizes** (10 minutes)
3. **Add FDR correction** via `statsmodels.stats.multitest` (5 minutes)
4. **Expand bootstrap CIs to all metrics** (modify existing function)
5. **Add culture and storyteller to summary tables** (modify groupby)

---

## Research References

For your paper/report, cite these standard bias detection methods:

1. **WEAT**: Caliskan et al. (2017) "Semantics derived automatically from language corpora contain human-like biases"
2. **BSRI**: Bem (1974) "The measurement of psychological androgyny"
3. **Job Ad Language**: Gaucher et al. (2011) "Evidence that gendered wording in job advertisements exists and sustains gender inequality"
4. **Stereotype Content Model**: Fiske et al. (2002) "A model of (often mixed) stereotype content"
5. **Gender Bias in NLP**: Bolukbasi et al. (2016) "Man is to Computer Programmer as Woman is to Homemaker?"

---

## Summary

Your current implementation is **solid and well-structured**, but can be enhanced by:
1. ✅ Adding statistical tests (t-tests, effect sizes, corrections)
2. ✅ Using semantic embeddings (more nuanced than lexicons)
3. ✅ Analyzing multi-level structure (culture, storyteller, model)
4. ✅ Extracting protagonist attributes directly
5. ✅ Validating with multiple bias detection methods

These improvements will make your findings more robust, publishable, and aligned with NLP fairness best practices.
