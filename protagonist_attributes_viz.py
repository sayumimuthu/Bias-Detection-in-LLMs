"""
Visualization and Statistical Analysis for Protagonist Attributes (Phase 1.2)
Creates comprehensive visualizations and performs statistical tests
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from scipy.stats import chi2_contingency, kruskal, mannwhitneyu
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10

INPUT_CSV = Path("Narratives/with_protagonist_attributes.csv")
OUTPUT_DIR = Path("Narratives/visualizations")
OUTPUT_DIR.mkdir(exist_ok=True)


def load_data():
    """Load the data with protagonist attributes"""
    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} stories with protagonist attributes")
    return df


def plot_trait_distribution(df):
    """Plot distribution of masculine/feminine/neutral traits by gender"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Stacked bar chart by protagonist gender
    trait_by_gender = df.groupby('protagonist_gender')[['masculine_traits', 'feminine_traits', 'neutral_traits']].mean()
    
    ax = axes[0, 0]
    trait_by_gender.plot(kind='bar', stacked=False, ax=ax, color=['#3498db', '#e74c3c', '#95a5a6'])
    ax.set_title('Average Trait Distribution by Protagonist Gender', fontsize=14, fontweight='bold')
    ax.set_xlabel('Protagonist Gender')
    ax.set_ylabel('Average Count')
    ax.legend(title='Trait Type', labels=['Masculine-coded', 'Feminine-coded', 'Neutral'])
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    
    # 2. Trait distribution by culture
    trait_by_culture = df.groupby('culture')[['masculine_traits', 'feminine_traits', 'neutral_traits']].mean()
    
    ax = axes[0, 1]
    trait_by_culture.plot(kind='bar', stacked=False, ax=ax, color=['#3498db', '#e74c3c', '#95a5a6'])
    ax.set_title('Average Trait Distribution by Culture', fontsize=14, fontweight='bold')
    ax.set_xlabel('Culture')
    ax.set_ylabel('Average Count')
    ax.legend(title='Trait Type', labels=['Masculine-coded', 'Feminine-coded', 'Neutral'])
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    
    # 3. Trait distribution by story-teller
    trait_by_person = df.groupby('person')[['masculine_traits', 'feminine_traits', 'neutral_traits']].mean()
    
    ax = axes[1, 0]
    trait_by_person.plot(kind='bar', stacked=False, ax=ax, color=['#3498db', '#e74c3c', '#95a5a6'])
    ax.set_title('Average Trait Distribution by Story-teller', fontsize=14, fontweight='bold')
    ax.set_xlabel('Story-teller')
    ax.set_ylabel('Average Count')
    ax.legend(title='Trait Type', labels=['Masculine-coded', 'Feminine-coded', 'Neutral'])
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    
    # 4. Interaction: Gender x Culture heatmap for masculine traits
    ax = axes[1, 1]
    pivot_masc = df.pivot_table(values='masculine_traits', index='culture', columns='protagonist_gender', aggfunc='mean')
    sns.heatmap(pivot_masc, annot=True, fmt='.2f', cmap='Blues', ax=ax, cbar_kws={'label': 'Avg Masculine Traits'})
    ax.set_title('Masculine Traits: Culture × Gender Interaction', fontsize=14, fontweight='bold')
    ax.set_xlabel('Protagonist Gender')
    ax.set_ylabel('Culture')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'trait_distribution.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {OUTPUT_DIR / 'trait_distribution.png'}")
    plt.close()


def plot_activity_distribution(df):
    """Plot distribution of activities across groups"""
    activity_cols = ['activity_adventure', 'activity_domestic', 'activity_intellectual', 
                     'activity_social', 'activity_creative']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Activity distribution by gender
    ax = axes[0, 0]
    activity_by_gender = df.groupby('protagonist_gender')[activity_cols].mean()
    activity_by_gender.plot(kind='bar', ax=ax, colormap='Set2')
    ax.set_title('Average Activity Distribution by Protagonist Gender', fontsize=14, fontweight='bold')
    ax.set_xlabel('Protagonist Gender')
    ax.set_ylabel('Average Count')
    ax.legend(title='Activity Type', labels=['Adventure', 'Domestic', 'Intellectual', 'Social', 'Creative'], 
              bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    
    # 2. Activity distribution by culture
    ax = axes[0, 1]
    activity_by_culture = df.groupby('culture')[activity_cols].mean()
    activity_by_culture.plot(kind='bar', ax=ax, colormap='Set2')
    ax.set_title('Average Activity Distribution by Culture', fontsize=14, fontweight='bold')
    ax.set_xlabel('Culture')
    ax.set_ylabel('Average Count')
    ax.legend(title='Activity Type', labels=['Adventure', 'Domestic', 'Intellectual', 'Social', 'Creative'], 
              bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    
    # 3. Adventure vs Domestic by gender (key stereotype indicator)
    ax = axes[1, 0]
    adv_dom = df.groupby('protagonist_gender')[['activity_adventure', 'activity_domestic']].mean()
    x = np.arange(len(adv_dom.index))
    width = 0.35
    ax.bar(x - width/2, adv_dom['activity_adventure'], width, label='Adventure', color='#e67e22')
    ax.bar(x + width/2, adv_dom['activity_domestic'], width, label='Domestic', color='#9b59b6')
    ax.set_title('Adventure vs Domestic Activities by Gender', fontsize=14, fontweight='bold')
    ax.set_xlabel('Protagonist Gender')
    ax.set_ylabel('Average Count')
    ax.set_xticks(x)
    ax.set_xticklabels(adv_dom.index)
    ax.legend()
    
    # 4. Heatmap: Activity types by culture
    ax = axes[1, 1]
    activity_by_culture_T = activity_by_culture.T
    activity_by_culture_T.index = ['Adventure', 'Domestic', 'Intellectual', 'Social', 'Creative']
    sns.heatmap(activity_by_culture_T, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax, 
                cbar_kws={'label': 'Avg Count'})
    ax.set_title('Activity Type Heatmap by Culture', fontsize=14, fontweight='bold')
    ax.set_xlabel('Culture')
    ax.set_ylabel('Activity Type')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'activity_distribution.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {OUTPUT_DIR / 'activity_distribution.png'}")
    plt.close()


def plot_occupation_distribution(df):
    """Plot occupation category distribution"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Occupation by gender
    ax = axes[0, 0]
    occ_gender = pd.crosstab(df['protagonist_gender'], df['occupation_category'], normalize='index') * 100
    occ_gender.plot(kind='bar', stacked=True, ax=ax, colormap='Spectral')
    ax.set_title('Occupation Categories by Protagonist Gender (%)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Protagonist Gender')
    ax.set_ylabel('Percentage')
    ax.legend(title='Occupation', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    
    # 2. Occupation by culture
    ax = axes[0, 1]
    occ_culture = pd.crosstab(df['culture'], df['occupation_category'], normalize='index') * 100
    occ_culture.plot(kind='bar', stacked=True, ax=ax, colormap='Spectral')
    ax.set_title('Occupation Categories by Culture (%)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Culture')
    ax.set_ylabel('Percentage')
    ax.legend(title='Occupation', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    
    # 3. Key occupations (leadership, professional, service) by gender
    ax = axes[1, 0]
    key_occs = df[df['occupation_category'].isin(['leadership', 'professional', 'service'])]
    if len(key_occs) > 0:
        key_occ_counts = pd.crosstab(key_occs['protagonist_gender'], key_occs['occupation_category'])
        key_occ_counts.plot(kind='bar', ax=ax, color=['#e74c3c', '#3498db', '#2ecc71'])
        ax.set_title('Key Occupations (Leadership/Professional/Service) by Gender', fontsize=14, fontweight='bold')
        ax.set_xlabel('Protagonist Gender')
        ax.set_ylabel('Count')
        ax.legend(title='Occupation')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    
    # 4. Occupation counts overall
    ax = axes[1, 1]
    occ_counts = df['occupation_category'].value_counts()
    occ_counts.plot(kind='barh', ax=ax, color='skyblue')
    ax.set_title('Overall Occupation Category Distribution', fontsize=14, fontweight='bold')
    ax.set_xlabel('Count')
    ax.set_ylabel('Occupation Category')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'occupation_distribution.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {OUTPUT_DIR / 'occupation_distribution.png'}")
    plt.close()


def plot_agency_and_outcomes(df):
    """Plot agency levels and outcomes"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Agency by gender
    ax = axes[0, 0]
    agency_gender = pd.crosstab(df['protagonist_gender'], df['agency_level'], normalize='index') * 100
    agency_gender = agency_gender[['high', 'medium', 'low']] if all(x in agency_gender.columns for x in ['high', 'medium', 'low']) else agency_gender
    agency_gender.plot(kind='bar', ax=ax, color=['#27ae60', '#f39c12', '#e74c3c'])
    ax.set_title('Agency Level by Protagonist Gender (%)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Protagonist Gender')
    ax.set_ylabel('Percentage')
    ax.legend(title='Agency Level')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    
    # 2. Agency by culture
    ax = axes[0, 1]
    agency_culture = pd.crosstab(df['culture'], df['agency_level'], normalize='index') * 100
    agency_culture = agency_culture[['high', 'medium', 'low']] if all(x in agency_culture.columns for x in ['high', 'medium', 'low']) else agency_culture
    agency_culture.plot(kind='bar', ax=ax, color=['#27ae60', '#f39c12', '#e74c3c'])
    ax.set_title('Agency Level by Culture (%)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Culture')
    ax.set_ylabel('Percentage')
    ax.legend(title='Agency Level')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    
    # 3. Outcome by gender
    ax = axes[0, 2]
    outcome_gender = pd.crosstab(df['protagonist_gender'], df['outcome'], normalize='index') * 100
    outcome_gender.plot(kind='bar', ax=ax, colormap='RdYlGn')
    ax.set_title('Story Outcome by Protagonist Gender (%)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Protagonist Gender')
    ax.set_ylabel('Percentage')
    ax.legend(title='Outcome')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    
    # 4. Sentiment trajectory by gender
    ax = axes[1, 0]
    df.boxplot(column='sentiment_trajectory', by='protagonist_gender', ax=ax)
    ax.set_title('Sentiment Trajectory by Protagonist Gender', fontsize=14, fontweight='bold')
    ax.set_xlabel('Protagonist Gender')
    ax.set_ylabel('Sentiment Change (Beginning to End)')
    plt.sca(ax)
    plt.xticks(rotation=0)
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='No change')
    
    # 5. Relationship focus by gender
    ax = axes[1, 1]
    rel_gender = pd.crosstab(df['protagonist_gender'], df['relationship_level'], normalize='index') * 100
    rel_gender = rel_gender[['high', 'medium', 'low']] if all(x in rel_gender.columns for x in ['high', 'medium', 'low']) else rel_gender
    rel_gender.plot(kind='bar', ax=ax, color=['#9b59b6', '#3498db', '#95a5a6'])
    ax.set_title('Relationship Focus by Protagonist Gender (%)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Protagonist Gender')
    ax.set_ylabel('Percentage')
    ax.legend(title='Relationship Focus')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    
    # 6. Agency vs Relationship scatter (using continuous scores)
    ax = axes[1, 2]
    
    for gender in df['protagonist_gender'].unique():
        gender_data = df[df['protagonist_gender'] == gender]
        ax.scatter(gender_data['agency_score'], gender_data['relationship_score'], 
                  alpha=0.6, label=gender, s=50)
    
    ax.set_title('Agency vs Relationship Focus by Gender', fontsize=14, fontweight='bold')
    ax.set_xlabel('Agency Score (0=low, 1=high)')
    ax.set_ylabel('Relationship Score (0=low, 1=high)')
    ax.legend(title='Protagonist Gender')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'agency_outcomes.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {OUTPUT_DIR / 'agency_outcomes.png'}")
    plt.close()


def plot_stereotype_scores(df):
    """Plot stereotype alignment scores"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Stereotype score distribution by gender
    ax = axes[0, 0]
    df.boxplot(column='stereotype_score', by='protagonist_gender', ax=ax)
    ax.set_title('Stereotype Score Distribution by Protagonist Gender', fontsize=14, fontweight='bold')
    ax.set_xlabel('Protagonist Gender')
    ax.set_ylabel('Stereotype Score (+ = stereotypical, - = counter-stereotypical)')
    plt.sca(ax)
    plt.xticks(rotation=0)
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Neutral')
    
    # 2. Stereotype score by culture
    ax = axes[0, 1]
    df.boxplot(column='stereotype_score', by='culture', ax=ax)
    ax.set_title('Stereotype Score Distribution by Culture', fontsize=14, fontweight='bold')
    ax.set_xlabel('Culture')
    ax.set_ylabel('Stereotype Score')
    plt.sca(ax)
    plt.xticks(rotation=45, ha='right')
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    # 3. Stereotype score by story-teller
    ax = axes[1, 0]
    df.boxplot(column='stereotype_score', by='person', ax=ax)
    ax.set_title('Stereotype Score Distribution by Story-teller', fontsize=14, fontweight='bold')
    ax.set_xlabel('Story-teller')
    ax.set_ylabel('Stereotype Score')
    plt.sca(ax)
    plt.xticks(rotation=45, ha='right')
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    # 4. Heatmap: Average stereotype score by culture x gender
    ax = axes[1, 1]
    pivot_stereo = df.pivot_table(values='stereotype_score', index='culture', 
                                   columns='protagonist_gender', aggfunc='mean')
    sns.heatmap(pivot_stereo, annot=True, fmt='.2f', cmap='RdBu_r', center=0, ax=ax,
                cbar_kws={'label': 'Avg Stereotype Score'})
    ax.set_title('Stereotype Score: Culture × Gender Interaction', fontsize=14, fontweight='bold')
    ax.set_xlabel('Protagonist Gender')
    ax.set_ylabel('Culture')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'stereotype_scores.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {OUTPUT_DIR / 'stereotype_scores.png'}")
    plt.close()


def perform_statistical_tests(df):
    """Perform statistical tests for bias detection"""
    
    print("\n" + "="*80)
    print("STATISTICAL TESTS FOR BIAS DETECTION")
    print("="*80)
    
    results = []
    
    # 1. Test for trait differences by gender
    print("\n1. TRAIT DIFFERENCES BY PROTAGONIST GENDER")
    print("-" * 60)
    
    for trait_type in ['masculine_traits', 'feminine_traits', 'neutral_traits']:
        male_data = df[df['protagonist_gender'] == 'male'][trait_type].dropna()
        female_data = df[df['protagonist_gender'] == 'female'][trait_type].dropna()
        
        if len(male_data) > 0 and len(female_data) > 0:
            # Mann-Whitney U test (non-parametric)
            stat, p_value = mannwhitneyu(male_data, female_data, alternative='two-sided')
            
            mean_male = male_data.mean()
            mean_female = female_data.mean()
            
            print(f"\n{trait_type.replace('_', ' ').title()}:")
            print(f"  Male protagonists: Mean = {mean_male:.2f} (SD = {male_data.std():.2f})")
            print(f"  Female protagonists: Mean = {mean_female:.2f} (SD = {female_data.std():.2f})")
            print(f"  Mann-Whitney U = {stat:.2f}, p-value = {p_value:.4f}")
            
            if p_value < 0.05:
                print(f"  *** SIGNIFICANT DIFFERENCE (p < 0.05) ***")
            
            results.append({
                'test': f'{trait_type}_by_gender',
                'statistic': stat,
                'p_value': p_value,
                'significant': p_value < 0.05
            })
    
    # 2. Test for activity differences by gender
    print("\n\n2. ACTIVITY DIFFERENCES BY PROTAGONIST GENDER")
    print("-" * 60)
    
    activity_cols = ['activity_adventure', 'activity_domestic', 'activity_intellectual', 
                     'activity_social', 'activity_creative']
    
    for activity in activity_cols:
        male_data = df[df['protagonist_gender'] == 'male'][activity].dropna()
        female_data = df[df['protagonist_gender'] == 'female'][activity].dropna()
        
        if len(male_data) > 0 and len(female_data) > 0:
            stat, p_value = mannwhitneyu(male_data, female_data, alternative='two-sided')
            
            mean_male = male_data.mean()
            mean_female = female_data.mean()
            
            print(f"\n{activity.replace('_', ' ').title()}:")
            print(f"  Male protagonists: Mean = {mean_male:.2f}")
            print(f"  Female protagonists: Mean = {mean_female:.2f}")
            print(f"  Mann-Whitney U = {stat:.2f}, p-value = {p_value:.4f}")
            
            if p_value < 0.05:
                print(f"  *** SIGNIFICANT DIFFERENCE (p < 0.05) ***")
            
            results.append({
                'test': f'{activity}_by_gender',
                'statistic': stat,
                'p_value': p_value,
                'significant': p_value < 0.05
            })
    
    # 3. Test stereotype score differences
    print("\n\n3. STEREOTYPE SCORE DIFFERENCES")
    print("-" * 60)
    
    # By gender
    male_stereo = df[df['protagonist_gender'] == 'male']['stereotype_score'].dropna()
    female_stereo = df[df['protagonist_gender'] == 'female']['stereotype_score'].dropna()
    
    if len(male_stereo) > 0 and len(female_stereo) > 0:
        stat, p_value = mannwhitneyu(male_stereo, female_stereo, alternative='two-sided')
        
        print(f"\nBy Protagonist Gender:")
        print(f"  Male protagonists: Mean = {male_stereo.mean():.2f} (SD = {male_stereo.std():.2f})")
        print(f"  Female protagonists: Mean = {female_stereo.mean():.2f} (SD = {female_stereo.std():.2f})")
        print(f"  Mann-Whitney U = {stat:.2f}, p-value = {p_value:.4f}")
        
        if p_value < 0.05:
            print(f"  *** SIGNIFICANT DIFFERENCE (p < 0.05) ***")
    
    # By culture (Kruskal-Wallis for multiple groups)
    cultures = df['culture'].unique()
    culture_groups = [df[df['culture'] == culture]['stereotype_score'].dropna() for culture in cultures]
    
    if all(len(g) > 0 for g in culture_groups):
        stat, p_value = kruskal(*culture_groups)
        
        print(f"\nBy Culture (Kruskal-Wallis test):")
        print(f"  H-statistic = {stat:.2f}, p-value = {p_value:.4f}")
        
        if p_value < 0.05:
            print(f"  *** SIGNIFICANT DIFFERENCE across cultures (p < 0.05) ***")
            print(f"\n  Culture-specific means:")
            for culture in cultures:
                culture_mean = df[df['culture'] == culture]['stereotype_score'].mean()
                print(f"    {culture}: {culture_mean:.2f}")
    
    # 4. Chi-square tests for categorical variables
    print("\n\n4. CATEGORICAL ASSOCIATIONS (CHI-SQUARE TESTS)")
    print("-" * 60)
    
    # Agency by gender
    contingency = pd.crosstab(df['protagonist_gender'], df['agency_level'])
    chi2, p_value, dof, expected = chi2_contingency(contingency)
    
    print(f"\nAgency Level × Protagonist Gender:")
    print(f"  Chi-square = {chi2:.2f}, p-value = {p_value:.4f}, df = {dof}")
    if p_value < 0.05:
        print(f"  *** SIGNIFICANT ASSOCIATION (p < 0.05) ***")
        print("\n  Contingency table:")
        print(contingency)
    
    # Occupation by gender
    contingency = pd.crosstab(df['protagonist_gender'], df['occupation_category'])
    chi2, p_value, dof, expected = chi2_contingency(contingency)
    
    print(f"\nOccupation Category × Protagonist Gender:")
    print(f"  Chi-square = {chi2:.2f}, p-value = {p_value:.4f}, df = {dof}")
    if p_value < 0.05:
        print(f"  *** SIGNIFICANT ASSOCIATION (p < 0.05) ***")
        print("\n  Contingency table:")
        print(contingency)
    
    # Save test results
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_DIR / 'statistical_test_results.csv', index=False)
    print(f"\n✓ Statistical test results saved to: {OUTPUT_DIR / 'statistical_test_results.csv'}")
    
    return results_df


def create_summary_report(df):
    """Create a text summary report"""
    
    report_path = OUTPUT_DIR / 'protagonist_attributes_report.txt'
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("PROTAGONIST ATTRIBUTE ANALYSIS REPORT (Phase 1.2)\n")
        f.write("Bias Detection in LLM-Generated Children's Stories\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Total Stories Analyzed: {len(df)}\n")
        f.write(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Group distributions
        f.write("DATASET COMPOSITION\n")
        f.write("-" * 60 + "\n")
        f.write(f"\nCultures: {df['culture'].value_counts().to_dict()}\n")
        f.write(f"Protagonist Genders: {df['protagonist_gender'].value_counts().to_dict()}\n")
        f.write(f"Story-tellers: {df['person'].value_counts().to_dict()}\n\n")
        
        # Key findings
        f.write("\nKEY FINDINGS\n")
        f.write("-" * 60 + "\n")
        
        # 1. Stereotype scores
        f.write("\n1. Stereotype Alignment Scores:\n")
        male_stereo = df[df['protagonist_gender'] == 'male']['stereotype_score'].mean()
        female_stereo = df[df['protagonist_gender'] == 'female']['stereotype_score'].mean()
        f.write(f"   - Male protagonists: {male_stereo:.2f}\n")
        f.write(f"   - Female protagonists: {female_stereo:.2f}\n")
        f.write(f"   - Interpretation: Positive = stereotypical, Negative = counter-stereotypical\n")
        
        # 2. Trait patterns
        f.write("\n2. Personality Trait Patterns:\n")
        male_masc = df[df['protagonist_gender'] == 'male']['masculine_traits'].mean()
        male_fem = df[df['protagonist_gender'] == 'male']['feminine_traits'].mean()
        female_masc = df[df['protagonist_gender'] == 'female']['masculine_traits'].mean()
        female_fem = df[df['protagonist_gender'] == 'female']['feminine_traits'].mean()
        
        f.write(f"   - Male protagonists: {male_masc:.2f} masculine-coded, {male_fem:.2f} feminine-coded\n")
        f.write(f"   - Female protagonists: {female_masc:.2f} masculine-coded, {female_fem:.2f} feminine-coded\n")
        
        # 3. Activity patterns
        f.write("\n3. Activity Patterns:\n")
        male_adv = df[df['protagonist_gender'] == 'male']['activity_adventure'].mean()
        male_dom = df[df['protagonist_gender'] == 'male']['activity_domestic'].mean()
        female_adv = df[df['protagonist_gender'] == 'female']['activity_adventure'].mean()
        female_dom = df[df['protagonist_gender'] == 'female']['activity_domestic'].mean()
        
        f.write(f"   - Male protagonists: {male_adv:.2f} adventure, {male_dom:.2f} domestic\n")
        f.write(f"   - Female protagonists: {female_adv:.2f} adventure, {female_dom:.2f} domestic\n")
        
        # 4. Agency
        f.write("\n4. Agency Levels:\n")
        agency_by_gender = pd.crosstab(df['protagonist_gender'], df['agency_level'], normalize='index') * 100
        for gender in ['male', 'female']:
            if gender in agency_by_gender.index:
                if 'high' in agency_by_gender.columns:
                    high_agency = agency_by_gender.loc[gender, 'high']
                    f.write(f"   {gender.capitalize()} protagonists with high agency: {high_agency:.1f}%\n")
        
        # Average agency scores
        male_agency = df[df['protagonist_gender'] == 'male']['agency_score'].mean()
        female_agency = df[df['protagonist_gender'] == 'female']['agency_score'].mean()
        f.write(f"   Male protagonist average agency score: {male_agency:.3f}\n")
        f.write(f"   Female protagonist average agency score: {female_agency:.3f}\n")
        
        # 5. Occupations
        f.write("\n5. Occupation Patterns:\n")
        occ_by_gender = pd.crosstab(df['protagonist_gender'], df['occupation_category'])
        f.write(f"   Most common occupation categories:\n")
        for gender in ['male', 'female']:
            if gender in occ_by_gender.index:
                top_occ = occ_by_gender.loc[gender].nlargest(3)
                f.write(f"   - {gender.capitalize()}: {', '.join([f'{k} ({v})' for k, v in top_occ.items()])}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*80 + "\n")
    
    print(f"\n✓ Summary report saved to: {report_path}")


def main():
    print("Loading protagonist attribute data...")
    df = load_data()
    
    print("\nGenerating visualizations...")
    print("-" * 60)
    
    plot_trait_distribution(df)
    plot_activity_distribution(df)
    plot_occupation_distribution(df)
    plot_agency_and_outcomes(df)
    plot_stereotype_scores(df)
    
    print("\nPerforming statistical tests...")
    print("-" * 60)
    perform_statistical_tests(df)
    
    print("\nGenerating summary report...")
    print("-" * 60)
    create_summary_report(df)
    
    print("\n" + "="*80)
    print("✓ ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nAll outputs saved to: {OUTPUT_DIR}/")
    print("\nGenerated files:")
    print(f"  1. trait_distribution.png - Personality trait analysis")
    print(f"  2. activity_distribution.png - Activity pattern analysis")
    print(f"  3. occupation_distribution.png - Occupation category analysis")
    print(f"  4. agency_outcomes.png - Agency and outcome analysis")
    print(f"  5. stereotype_scores.png - Stereotype alignment analysis")
    print(f"  6. statistical_test_results.csv - Statistical test results")
    print(f"  7. protagonist_attributes_report.txt - Comprehensive text report")


if __name__ == "__main__":
    main()
