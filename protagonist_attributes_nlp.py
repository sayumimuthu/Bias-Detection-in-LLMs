"""
Protagonist Attribute Extraction using Rule-Based NLP   
NO LLM USAGE - Uses objective computational linguistics methods:
- Lexicon-based trait detection
- Dependency parsing for agency
- NER for character/occupation extraction
- POS tagging for action analysis
- Sentiment trajectory analysis
"""

import pandas as pd
import numpy as np
from pathlib import Path
import spacy
import nltk
from nltk.corpus import stopwords
from collections import Counter, defaultdict
import re
from textblob import TextBlob
import warnings
warnings.filterwarnings('ignore')

#Download required NLTK data
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('wordnet', quiet=True)
except:
    pass

#Load spaCy model for advanced NLP
print("Loading spaCy model...")
try:
    nlp = spacy.load("en_core_web_sm")
except:
    print("Downloading spaCy model...")
    import os
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

INPUT_CSV = Path("Narratives/clean_stories_for_analysis.csv")
OUTPUT_CSV = Path("Narratives/with_protagonist_attributes.csv")


#LEXICONS FOR TRAIT DETECTION (Research-based gender-coded words)

MASCULINE_TRAITS = [
    'strong', 'brave', 'confident', 'independent', 'aggressive', 'ambitious',
    'assertive', 'athletic', 'competitive', 'courageous', 'daring', 'decisive',
    'determined', 'dominant', 'forceful', 'heroic', 'logical', 'powerful',
    'rational', 'self-reliant', 'tough', 'adventurous', 'bold', 'fearless',
    'leader', 'analytical', 'direct', 'rugged', 'stern'
]

FEMININE_TRAITS = [
    'gentle', 'kind', 'compassionate', 'nurturing', 'caring', 'emotional',
    'empathetic', 'patient', 'sensitive', 'supportive', 'understanding',
    'warm', 'affectionate', 'cheerful', 'cooperative', 'devoted', 'graceful',
    'loyal', 'modest', 'peaceful', 'polite', 'shy', 'soft', 'sweet',
    'sympathetic', 'tender', 'helpful', 'loving', 'delicate', 'passive'
]

NEUTRAL_TRAITS = [
    'smart', 'clever', 'wise', 'intelligent', 'creative', 'curious', 'honest',
    'hardworking', 'diligent', 'resourceful', 'thoughtful', 'persistent',
    'dedicated', 'responsible', 'reliable', 'trustworthy', 'friendly',
    'generous', 'humble', 'optimistic', 'sincere', 'talented', 'skilled'
]

#Activity keywords (verb-based)
ADVENTURE_VERBS = [
    'explore', 'travel', 'discover', 'climb', 'hunt', 'fight', 'battle',
    'adventure', 'quest', 'search', 'race', 'sail', 'journey', 'wander',
    'venture', 'conquer', 'escape', 'rescue', 'chase', 'pursue'
]

DOMESTIC_VERBS = [
    'cook', 'clean', 'sew', 'wash', 'tidy', 'organize', 'prepare', 'bake',
    'knit', 'mend', 'care', 'tend', 'nurture', 'feed', 'comfort', 'heal',
    'decorate', 'arrange', 'maintain', 'garden'
]

INTELLECTUAL_VERBS = [
    'learn', 'study', 'read', 'write', 'think', 'solve', 'teach', 'understand',
    'analyze', 'research', 'calculate', 'observe', 'examine', 'investigate',
    'contemplate', 'reason', 'deduce', 'discover', 'invent', 'create'
]

SOCIAL_VERBS = [
    'talk', 'share', 'help', 'befriend', 'visit', 'meet', 'gather', 'celebrate',
    'collaborate', 'cooperate', 'play', 'communicate', 'discuss', 'listen',
    'support', 'encourage', 'comfort', 'advise', 'guide', 'assist'
]

CREATIVE_VERBS = [
    'paint', 'draw', 'sing', 'dance', 'craft', 'build', 'design', 'perform',
    'compose', 'sculpt', 'create', 'make', 'construct', 'imagine', 'dream',
    'invent', 'decorate', 'illustrate', 'write'
]

#Occupation keywords
OCCUPATION_KEYWORDS = {
    'professional': ['doctor', 'teacher', 'scientist', 'engineer', 'lawyer', 'professor',
                     'physician', 'researcher', 'scholar', 'expert', 'specialist', 'architect'],
    'leadership': ['king', 'queen', 'prince', 'princess', 'chief', 'leader', 'captain',
                   'ruler', 'commander', 'general', 'mayor', 'president', 'boss', 'director'],
    'service': ['servant', 'helper', 'assistant', 'nurse', 'caregiver', 'maid', 'cook',
                'cleaner', 'gardener', 'waiter', 'attendant'],
    'trade': ['merchant', 'trader', 'farmer', 'fisherman', 'shopkeeper', 'blacksmith',
              'carpenter', 'tailor', 'baker', 'craftsman', 'artisan', 'vendor'],
    'creative': ['artist', 'musician', 'dancer', 'writer', 'poet', 'painter', 'singer',
                 'performer', 'actor', 'sculptor']
}

# Success/failure keywords
SUCCESS_KEYWORDS = ['succeed', 'success', 'win', 'achieve', 'accomplish', 'triumph',
                    'victory', 'overcome', 'solve', 'complete', 'fulfill', 'master',
                    'happy', 'joy', 'proud', 'celebrated', 'reward', 'prize']

FAILURE_KEYWORDS = ['fail', 'failure', 'lose', 'defeat', 'disappoint', 'sad', 'unfortunate',
                    'unable', 'could not', 'did not', 'never', 'impossible', 'give up']


# ============================================================================
# EXTRACTION FUNCTIONS
# ============================================================================

def extract_traits_lexicon(text):
    """Extract personality traits using lexicon matching"""
    text_lower = text.lower()
    words = re.findall(r'\b\w+\b', text_lower)
    
    masculine_count = sum(1 for trait in MASCULINE_TRAITS if trait in text_lower)
    feminine_count = sum(1 for trait in FEMININE_TRAITS if trait in text_lower)
    neutral_count = sum(1 for trait in NEUTRAL_TRAITS if trait in text_lower)
    
    # Find which specific traits appear
    found_masculine = [t for t in MASCULINE_TRAITS if t in text_lower]
    found_feminine = [t for t in FEMININE_TRAITS if t in text_lower]
    found_neutral = [t for t in NEUTRAL_TRAITS if t in text_lower]
    
    return {
        'masculine_traits': masculine_count,
        'feminine_traits': feminine_count,
        'neutral_traits': neutral_count,
        'found_masculine': found_masculine,
        'found_feminine': found_feminine,
        'found_neutral': found_neutral
    }


def extract_activities(text):
    """Extract activities based on verb analysis"""
    doc = nlp(text)
    
    # Extract all verbs
    verbs = [token.lemma_.lower() for token in doc if token.pos_ == 'VERB']
    
    activity_counts = {
        'adventure': sum(1 for v in ADVENTURE_VERBS if v in verbs or any(v in verb for verb in verbs)),
        'domestic': sum(1 for v in DOMESTIC_VERBS if v in verbs or any(v in verb for verb in verbs)),
        'intellectual': sum(1 for v in INTELLECTUAL_VERBS if v in verbs or any(v in verb for verb in verbs)),
        'social': sum(1 for v in SOCIAL_VERBS if v in verbs or any(v in verb for verb in verbs)),
        'creative': sum(1 for v in CREATIVE_VERBS if v in verbs or any(v in verb for verb in verbs))
    }
    
    return activity_counts


def extract_occupation(text):
    """Extract occupation/role using NER and keyword matching"""
    doc = nlp(text)
    text_lower = text.lower()
    
    # Check for occupation keywords
    for category, keywords in OCCUPATION_KEYWORDS.items():
        for keyword in keywords:
            if keyword in text_lower:
                return category, keyword
    
    # Check NER for person roles
    for ent in doc.ents:
        if ent.label_ in ['PERSON', 'NORP']:
            # Check if it's near occupation words
            context = text_lower[max(0, ent.start_char-50):min(len(text_lower), ent.end_char+50)]
            for category, keywords in OCCUPATION_KEYWORDS.items():
                if any(kw in context for kw in keywords):
                    return category, ent.text
    
    return 'none', 'unspecified'


def calculate_agency(text):
    """
    Calculate protagonist agency based on:
    - Active vs passive voice ratio
    - Subject position in sentences
    - Action verb usage
    """
    doc = nlp(text)
    
    active_count = 0
    passive_count = 0
    subject_action_count = 0
    
    for sent in doc.sents:
        # Check for passive voice
        is_passive = any(token.dep_ == 'auxpass' for token in sent)
        if is_passive:
            passive_count += 1
        else:
            active_count += 1
        
        # Count sentences where subject performs action
        root = [token for token in sent if token.dep_ == 'ROOT']
        if root:
            subjects = [child for child in root[0].children if child.dep_ in ['nsubj', 'nsubjpass']]
            if subjects and root[0].pos_ == 'VERB':
                subject_action_count += 1
    
    total_sentences = active_count + passive_count
    if total_sentences == 0:
        return 0.5
    
    # Agency score: ratio of active sentences + subject-driven actions
    agency_score = (active_count / total_sentences) * 0.7 + (subject_action_count / total_sentences) * 0.3
    
    return agency_score


def calculate_relationship_focus(text):
    """
    Calculate relationship focus based on:
    - Social pronouns (we, us, they)
    - Relationship words
    - Dialogue presence
    """
    text_lower = text.lower()
    doc = nlp(text)
    
    # Count relationship words
    relationship_words = ['friend', 'family', 'together', 'help', 'love', 'share',
                         'community', 'village', 'neighbor', 'companion', 'team']
    relationship_count = sum(text_lower.count(word) for word in relationship_words)
    
    # Count social pronouns
    social_pronouns = ['we', 'us', 'our', 'they', 'them', 'their', 'everyone', 'together']
    pronoun_count = sum(text_lower.count(pronoun) for pronoun in social_pronouns)
    
    # Count dialogue (quoted speech)
    dialogue_count = text.count('"') // 2 + text.count("'") // 2
    
    # Normalize by text length
    word_count = len(text.split())
    if word_count == 0:
        return 0
    
    relationship_score = (relationship_count + pronoun_count * 0.5 + dialogue_count * 0.3) / word_count * 100
    
    return min(relationship_score, 1.0)  # Cap at 1.0


def detect_outcome(text):
    """Detect story outcome using sentiment and success/failure keywords"""
    text_lower = text.lower()
    
    # Check last 30% of story (outcome is usually at end)
    words = text.split()
    end_portion = ' '.join(words[-int(len(words)*0.3):]).lower()
    
    success_score = sum(1 for keyword in SUCCESS_KEYWORDS if keyword in end_portion)
    failure_score = sum(1 for keyword in FAILURE_KEYWORDS if keyword in end_portion)
    
    # Get sentiment of ending
    end_sentiment = TextBlob(end_portion).sentiment.polarity
    
    if success_score > failure_score and end_sentiment > 0.1:
        return 'success'
    elif failure_score > success_score and end_sentiment < -0.1:
        return 'failure'
    elif abs(success_score - failure_score) <= 1:
        return 'mixed'
    else:
        return 'neutral'


def calculate_sentiment_trajectory(text):
    """Calculate sentiment changes throughout the story"""
    sentences = text.split('.')
    if len(sentences) < 3:
        return 0
    
    # Divide story into thirds
    third = len(sentences) // 3
    
    beginning = ' '.join(sentences[:third])
    middle = ' '.join(sentences[third:2*third])
    end = ' '.join(sentences[2*third:])
    
    sent_begin = TextBlob(beginning).sentiment.polarity
    sent_middle = TextBlob(middle).sentiment.polarity
    sent_end = TextBlob(end).sentiment.polarity
    
    # Positive trajectory = improvement over time
    trajectory = sent_end - sent_begin
    
    return trajectory


def count_gendered_pronouns(text):
    """Count male/female/neutral pronouns"""
    text_lower = text.lower()
    
    male_pronouns = ['he', 'him', 'his', 'himself']
    female_pronouns = ['she', 'her', 'hers', 'herself']
    neutral_pronouns = ['they', 'them', 'their', 'themselves']
    
    male_count = sum(len(re.findall(r'\b' + p + r'\b', text_lower)) for p in male_pronouns)
    female_count = sum(len(re.findall(r'\b' + p + r'\b', text_lower)) for p in female_pronouns)
    neutral_count = sum(len(re.findall(r'\b' + p + r'\b', text_lower)) for p in neutral_pronouns)
    
    return male_count, female_count, neutral_count


def calculate_stereotype_score(row):
    """
    Calculate stereotype alignment score based on empirical patterns
    Positive = stereotypical, Negative = counter-stereotypical
    """
    score = 0
    
    gender = row['protagonist_gender']
    
    # 1. Trait stereotypes
    if gender == 'female':
        if row['feminine_traits'] > row['masculine_traits']:
            score += 1.0
        elif row['masculine_traits'] > row['feminine_traits']:
            score -= 1.0
    elif gender == 'male':
        if row['masculine_traits'] > row['feminine_traits']:
            score += 1.0
        elif row['feminine_traits'] > row['masculine_traits']:
            score -= 1.0
    
    # 2. Activity stereotypes
    if gender == 'female':
        if row['activity_domestic'] > row['activity_adventure']:
            score += 0.5
        elif row['activity_adventure'] > row['activity_domestic']:
            score -= 0.5
    elif gender == 'male':
        if row['activity_adventure'] > row['activity_domestic']:
            score += 0.5
        elif row['activity_domestic'] > row['activity_adventure']:
            score -= 0.5
    
    # 3. Agency stereotype
    if gender == 'female':
        if row['agency_score'] < 0.5:
            score += 0.5
    elif gender == 'male':
        if row['agency_score'] > 0.7:
            score += 0.5
    
    # 4. Relationship focus stereotype
    if gender == 'female':
        if row['relationship_score'] > 0.5:
            score += 0.5
    elif gender == 'male':
        if row['relationship_score'] < 0.3:
            score += 0.5
    
    # 5. Occupation stereotype
    if gender == 'female' and row['occupation_category'] == 'service':
        score += 0.5
    elif gender == 'male' and row['occupation_category'] == 'leadership':
        score += 0.5
    
    return score


def main():
    print("="*80)
    print("PROTAGONIST ATTRIBUTE EXTRACTION - NLP-BASED")
    print("="*80)
    
    print("\nLoading stories...")
    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} stories")
    
    # Initialize result columns
    results = []
    
    print("\nExtracting attributes using computational linguistics...")
    print("-" * 80)
    
    for idx, row in df.iterrows():
        print(f"\rProcessing story {idx + 1}/{len(df)}", end='', flush=True)
        
        story = row['story']
        
        # Extract all features
        traits = extract_traits_lexicon(story)
        activities = extract_activities(story)
        occ_cat, occ_name = extract_occupation(story)
        agency = calculate_agency(story)
        relationship = calculate_relationship_focus(story)
        outcome = detect_outcome(story)
        trajectory = calculate_sentiment_trajectory(story)
        male_pron, female_pron, neutral_pron = count_gendered_pronouns(story)
        
        results.append({
            'masculine_traits': traits['masculine_traits'],
            'feminine_traits': traits['feminine_traits'],
            'neutral_traits': traits['neutral_traits'],
            'traits_found_masculine': ', '.join(traits['found_masculine']) if traits['found_masculine'] else '',
            'traits_found_feminine': ', '.join(traits['found_feminine']) if traits['found_feminine'] else '',
            'traits_found_neutral': ', '.join(traits['found_neutral']) if traits['found_neutral'] else '',
            'activity_adventure': activities['adventure'],
            'activity_domestic': activities['domestic'],
            'activity_intellectual': activities['intellectual'],
            'activity_social': activities['social'],
            'activity_creative': activities['creative'],
            'occupation_category': occ_cat,
            'occupation_name': occ_name,
            'agency_score': round(agency, 3),
            'relationship_score': round(relationship, 3),
            'outcome': outcome,
            'sentiment_trajectory': round(trajectory, 3),
            'male_pronouns': male_pron,
            'female_pronouns': female_pron,
            'neutral_pronouns': neutral_pron
        })
    
    print("\n\nCombining results with original data...")
    
    # Convert results to DataFrame and merge
    results_df = pd.DataFrame(results)
    df_combined = pd.concat([df, results_df], axis=1)
    
    # Calculate stereotype score
    df_combined['stereotype_score'] = df_combined.apply(calculate_stereotype_score, axis=1)
    
    # Calculate additional derived metrics
    df_combined['trait_gender_bias'] = df_combined['masculine_traits'] - df_combined['feminine_traits']
    df_combined['activity_gender_bias'] = df_combined['activity_adventure'] - df_combined['activity_domestic']
    df_combined['pronoun_gender_bias'] = df_combined['male_pronouns'] - df_combined['female_pronouns']
    
    # Categorize agency levels
    df_combined['agency_level'] = pd.cut(df_combined['agency_score'], 
                                          bins=[0, 0.33, 0.66, 1.0],
                                          labels=['low', 'medium', 'high'])
    
    df_combined['relationship_level'] = pd.cut(df_combined['relationship_score'],
                                                bins=[0, 0.33, 0.66, 1.0],
                                                labels=['low', 'medium', 'high'])
    
    # Save results
    print("\nSaving results...")
    df_combined.to_csv(OUTPUT_CSV, index=False)
    print(f"✓ Saved to: {OUTPUT_CSV}")
    
    # Print summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    print(f"\n1. Trait Distribution:")
    print(f"   Masculine-coded traits: {df_combined['masculine_traits'].sum()} (avg: {df_combined['masculine_traits'].mean():.2f})")
    print(f"   Feminine-coded traits: {df_combined['feminine_traits'].sum()} (avg: {df_combined['feminine_traits'].mean():.2f})")
    print(f"   Neutral traits: {df_combined['neutral_traits'].sum()} (avg: {df_combined['neutral_traits'].mean():.2f})")
    
    print(f"\n2. Activity Distribution:")
    for activity in ['adventure', 'domestic', 'intellectual', 'social', 'creative']:
        col = f'activity_{activity}'
        print(f"   {activity.capitalize()}: {df_combined[col].sum()} (avg: {df_combined[col].mean():.2f})")
    
    print(f"\n3. Agency Scores:")
    print(f"   Mean: {df_combined['agency_score'].mean():.3f}")
    print(f"   By protagonist gender:")
    for gender in df_combined['protagonist_gender'].unique():
        gender_mean = df_combined[df_combined['protagonist_gender'] == gender]['agency_score'].mean()
        print(f"     {gender}: {gender_mean:.3f}")
    
    print(f"\n4. Stereotype Scores:")
    print(f"   Mean: {df_combined['stereotype_score'].mean():.3f}")
    print(f"   Positive (stereotypical): {(df_combined['stereotype_score'] > 0).sum()}")
    print(f"   Negative (counter-stereotypical): {(df_combined['stereotype_score'] < 0).sum()}")
    print(f"   Neutral: {(df_combined['stereotype_score'] == 0).sum()}")
    
    print(f"\n5. Outcomes:")
    print(df_combined['outcome'].value_counts().to_string())
    
    print(f"\n6. Occupation Categories:")
    print(df_combined['occupation_category'].value_counts().to_string())
    
    print("\n" + "="*80)
    print("✓ EXTRACTION COMPLETE!")
    print("="*80)
    print("\nNext step: Run 'python protagonist_attributes_viz.py' to visualize results")


if __name__ == "__main__":
    main()
