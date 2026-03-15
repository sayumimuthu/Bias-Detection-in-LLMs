# Protagonist Attribute Extraction - Methodology Comparison

## ❌ OLD APPROACH: protagonist_attributes.py (LLM-based)
**Problem**: Uses LLM to analyze LLM-generated content → Circular bias!

### Issues:
1. **Bias amplification**: Detection LLM has its own biases
2. **Inconsistency**: LLM outputs vary even with low temperature
3. **Not reproducible**: Different API calls = different results
4. **Expensive**: API costs for 422 stories
5. **Slow**: ~7-10 minutes with rate limiting
6. **Black box**: Can't explain WHY a trait was detected

---

## ✅ NEW APPROACH: protagonist_attributes_nlp.py (Rule-based NLP)
**Solution**: Objective computational linguistics without LLM inference

### Methods Used:

#### 1. **Lexicon-Based Trait Detection**
- Predefined dictionaries of masculine/feminine/neutral coded words
- Based on research literature (Williams & Best, 1990; Gaucher et al., 2011)
- Counts exact matches in text
- **Transparent**: You can see exactly which words triggered detection

#### 2. **Dependency Parsing for Agency**
- Uses spaCy to analyze sentence structure
- Active vs passive voice ratio
- Subject-verb relationships
- Counts protagonist as sentence subject
- **Objective metric**: 0-1 agency score

#### 3. **Verb-Based Activity Detection**
- Extracts all verbs using POS tagging
- Categorizes by activity type (adventure, domestic, etc.)
- Lemmatization handles verb forms
- **Linguistic basis**: Verbs indicate actions

#### 4. **Named Entity Recognition (NER)**
- Identifies persons and their roles
- Occupation keyword matching
- Context window analysis
- **Pre-trained models**: spaCy's research-validated NER

#### 5. **Sentiment Analysis**
- TextBlob polarity scores
- Trajectory analysis (beginning → end)
- Outcome detection from story endings
- **Established methods**: Standard NLP technique

#### 6. **Pronoun Analysis**
- Regex-based pronoun counting
- Male/female/neutral pronoun ratios
- Indicates character gender representation
- **Quantitative**: Hard counts

#### 7. **Relationship Focus Score**
- Social pronoun frequency (we, us, they)
- Relationship keyword matching
- Dialogue detection (quoted speech)
- **Normalized metrics**: Per-word basis

---

## Metrics Extracted (NLP-based)

### Trait Metrics:
- `masculine_traits` - Count of masculine-coded words
- `feminine_traits` - Count of feminine-coded words  
- `neutral_traits` - Count of neutral trait words
- `traits_found_*` - Specific words found (transparency)

### Activity Metrics:
- `activity_adventure` - Adventure verb count
- `activity_domestic` - Domestic verb count
- `activity_intellectual` - Intellectual verb count
- `activity_social` - Social verb count
- `activity_creative` - Creative verb count

### Agency & Power:
- `agency_score` - 0-1 scale (active voice, subject position)
- `agency_level` - Categorical: low/medium/high

### Relationship:
- `relationship_score` - 0-1 scale (social focus)
- `relationship_level` - Categorical: low/medium/high

### Occupation:
- `occupation_category` - professional/leadership/service/trade/creative/none
- `occupation_name` - Specific occupation found

### Outcome:
- `outcome` - success/failure/mixed/neutral
- `sentiment_trajectory` - Change from beginning to end

### Pronoun Analysis:
- `male_pronouns` - Count of he/him/his
- `female_pronouns` - Count of she/her/hers
- `neutral_pronouns` - Count of they/them/their

### Composite Scores:
- `stereotype_score` - Combined stereotype alignment (-3 to +3)
- `trait_gender_bias` - masculine_traits - feminine_traits
- `activity_gender_bias` - adventure - domestic
- `pronoun_gender_bias` - male_pronouns - female_pronouns

---

## Advantages of NLP Approach

✅ **Reproducible**: Same input = same output always  
✅ **Transparent**: Can inspect exactly why a score was assigned  
✅ **Fast**: Processes 422 stories in ~1-2 minutes  
✅ **No API costs**: All local computation  
✅ **Objective**: Based on linguistic research, not LLM opinions  
✅ **Explainable**: Can show which words/patterns drove results  
✅ **Research-validated**: Methods from published NLP papers  
✅ **No circular bias**: Detection tools don't have content biases  

---

## Research Foundations

### Gender-Coded Language:
- Gaucher, D., Friesen, J., & Kay, A. C. (2011). Evidence that gendered wording in job advertisements exists and sustains gender inequality. *Journal of Personality and Social Psychology*

### Agency Analysis:
- Prewitt-Freilino, J. L., et al. (2012). The gendering of language: A comparison of gender equality in countries with gendered, natural gender, and genderless languages. *Sex Roles*

### Stereotype Detection:
- Williams, J. E., & Best, D. L. (1990). *Measuring sex stereotypes: A multination study*. Sage Publications

---

## Usage

### Install dependencies:
```bash
bash install_nlp_dependencies.sh
```

### Run extraction:
```bash
python protagonist_attributes_nlp.py
```

### Visualize results:
```bash
python protagonist_attributes_viz.py
```

The visualization script works with both approaches - it just reads the CSV!
