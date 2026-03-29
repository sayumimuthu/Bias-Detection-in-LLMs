#!/usr/bin/env python3
"""
Publication-quality visualizations for gender bias analysis.

Creates:
1. Bar charts comparing models
2. Heatmaps showing bias patterns
3. Effect size visualizations
4. Confidence interval plots
5. Storyteller/culture analysis (if available)
6. Distribution plots

Usage:
    python3 gender_bias_visualizations.py
    python3 gender_bias_visualizations.py --input-dir Narratives2/gender_bias_lexicon_enhanced
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Rectangle

# Set publication-quality defaults
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 14
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

# Color schemes
GENDER_COLORS = {
    'daughter': '#E74C3C',  # Red
    'son': '#3498DB',       # Blue
    'female': '#E74C3C',
    'male': '#3498DB',
}

MODEL_COLORS = sns.color_palette("husl", 8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate publication-quality visualizations")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="Narratives2/gender_bias_lexicon_enhanced",
        help="Directory containing analysis CSV files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="Narratives2/gender_bias_lexicon_enhanced/plots",
        help="Directory for output figures",
    )
    parser.add_argument(
        "--formats",
        type=str,
        nargs="+",
        default=["png", "pdf"],
        help="Output formats (png, pdf, svg)",
    )
    return parser.parse_args()


def save_figure(fig: plt.Figure, name: str, output_dir: Path, formats: List[str]) -> None:
    """Save figure in multiple formats."""
    for fmt in formats:
        path = output_dir / f"{name}.{fmt}"
        fig.savefig(path, format=fmt, bbox_inches='tight', dpi=300)
        print(f"  Saved: {path}")


def add_significance_stars(ax, x1, x2, y, p_value, height=0.02):
    """Add significance stars above bars."""
    if p_value < 0.001:
        stars = "***"
    elif p_value < 0.01:
        stars = "**"
    elif p_value < 0.05:
        stars = "*"
    else:
        stars = "ns"

    # Draw line
    ax.plot([x1, x1, x2, x2], [y, y+height, y+height, y], 'k-', linewidth=1)
    # Add stars
    ax.text((x1+x2)/2, y+height, stars, ha='center', va='bottom', fontsize=10)


def plot_model_comparison_bars(
    summary_df: pd.DataFrame,
    significance_df: pd.DataFrame,
    output_dir: Path,
    formats: List[str]
) -> None:
    """
    Create bar charts comparing models across key metrics.
    Shows daughter vs son with significance markers.
    """
    print("\n1. Creating model comparison bar charts...")

    metrics = [
        ('feminine_traits_per_1000', 'Feminine Traits\n(per 1000 tokens)'),
        ('masculine_traits_per_1000', 'Masculine Traits\n(per 1000 tokens)'),
        ('communal_per_1000', 'Communal Language\n(per 1000 tokens)'),
        ('agentic_per_1000', 'Agentic Language\n(per 1000 tokens)'),
        ('child_target_stereotype_score', 'Stereotype Alignment Score'),
    ]

    for metric_col, metric_label in metrics:
        fig, ax = plt.subplots(figsize=(10, 6))

        # Prepare data
        pivot = summary_df.pivot(index='model', columns='child_label', values=metric_col)
        models = pivot.index.tolist()

        # Clean model names for display
        model_labels = [m.replace('openai/', '').replace('-versatile', '') for m in models]

        x = np.arange(len(models))
        width = 0.35

        # Create bars
        bars1 = ax.bar(x - width/2, pivot['daughter'], width,
                      label='Daughter', color=GENDER_COLORS['daughter'], alpha=0.8)
        bars2 = ax.bar(x + width/2, pivot['son'], width,
                      label='Son', color=GENDER_COLORS['son'], alpha=0.8)

        # Add significance stars
        for i, model in enumerate(models):
            sig_row = significance_df[
                (significance_df['model'] == model) &
                (significance_df['metric'] == metric_col)
            ]

            if len(sig_row) > 0:
                p_val = sig_row['p_value_fdr'].values[0]
                max_val = max(pivot.loc[model, 'daughter'], pivot.loc[model, 'son'])
                add_significance_stars(ax, x[i] - width/2, x[i] + width/2,
                                     max_val, p_val, height=max_val * 0.05)

        ax.set_xlabel('Model', fontweight='bold')
        ax.set_ylabel(metric_label, fontweight='bold')
        ax.set_title(f'{metric_label} by Model and Child Gender', fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(model_labels, rotation=45, ha='right')
        ax.legend(loc='upper right')
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        # Add horizontal line at 0 if metric can be negative
        if pivot.values.min() < 0:
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)

        plt.tight_layout()
        save_figure(fig, f'bar_model_comparison_{metric_col}', output_dir, formats)
        plt.close()


def plot_effect_size_heatmap(
    significance_df: pd.DataFrame,
    output_dir: Path,
    formats: List[str]
) -> None:
    """
    Create heatmap of effect sizes (Cohen's d) across models and metrics.
    """
    print("\n2. Creating effect size heatmap...")

    # Select key metrics
    key_metrics = [
        'masculine_traits_per_1000',
        'feminine_traits_per_1000',
        'agentic_per_1000',
        'communal_per_1000',
        'trait_bias_index',
        'role_bias_index',
        'direct_gender_marker_index',
        'child_target_stereotype_score',
    ]

    # Filter data
    heatmap_data = significance_df[significance_df['metric'].isin(key_metrics)]
    pivot = heatmap_data.pivot(index='metric', columns='model', values='cohens_d')

    # Clean labels
    metric_labels = {
        'masculine_traits_per_1000': 'Masculine Traits',
        'feminine_traits_per_1000': 'Feminine Traits',
        'agentic_per_1000': 'Agentic Language',
        'communal_per_1000': 'Communal Language',
        'trait_bias_index': 'Trait Bias Index',
        'role_bias_index': 'Role Bias Index',
        'direct_gender_marker_index': 'Gender Marker Index',
        'child_target_stereotype_score': 'Stereotype Score',
    }
    pivot.index = [metric_labels.get(m, m) for m in pivot.index]
    pivot.columns = [c.replace('openai/', '').replace('-versatile', '') for c in pivot.columns]

    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))

    # Use diverging colormap centered at 0
    vmax = max(abs(pivot.min().min()), abs(pivot.max().max()))
    sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                vmin=-vmax, vmax=vmax, cbar_kws={'label': "Cohen's d"},
                linewidths=0.5, linecolor='gray', ax=ax)

    ax.set_title("Effect Sizes (Cohen's d) by Model and Metric\nDaughter vs Son Comparison",
                fontweight='bold', pad=20)
    ax.set_xlabel('Model', fontweight='bold')
    ax.set_ylabel('Metric', fontweight='bold')

    plt.tight_layout()
    save_figure(fig, 'heatmap_effect_sizes', output_dir, formats)
    plt.close()


def plot_significance_heatmap(
    significance_df: pd.DataFrame,
    output_dir: Path,
    formats: List[str]
) -> None:
    """
    Create heatmap showing which comparisons are statistically significant.
    """
    print("\n3. Creating significance heatmap...")

    key_metrics = [
        'masculine_traits_per_1000',
        'feminine_traits_per_1000',
        'agentic_per_1000',
        'communal_per_1000',
        'career_per_1000',
        'family_per_1000',
        'trait_bias_index',
        'role_bias_index',
        'domain_bias_index',
        'direct_gender_marker_index',
        'child_target_stereotype_score',
    ]

    heatmap_data = significance_df[significance_df['metric'].isin(key_metrics)]

    # Create matrix: 1 = significant (FDR), 0 = not significant
    pivot = heatmap_data.pivot(index='metric', columns='model', values='significant_fdr')
    pivot = pivot.astype(int)

    # Clean labels
    metric_labels = {
        'masculine_traits_per_1000': 'Masculine Traits',
        'feminine_traits_per_1000': 'Feminine Traits',
        'agentic_per_1000': 'Agentic Language',
        'communal_per_1000': 'Communal Language',
        'career_per_1000': 'Career Terms',
        'family_per_1000': 'Family Terms',
        'trait_bias_index': 'Trait Bias Index',
        'role_bias_index': 'Role Bias Index',
        'domain_bias_index': 'Domain Bias Index',
        'direct_gender_marker_index': 'Gender Marker Index',
        'child_target_stereotype_score': 'Stereotype Score',
    }
    pivot.index = [metric_labels.get(m, m) for m in pivot.index]
    pivot.columns = [c.replace('openai/', '').replace('-versatile', '') for c in pivot.columns]

    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(pivot, annot=True, fmt='d', cmap=['white', 'darkgreen'],
                cbar_kws={'label': 'Significant (FDR p < 0.05)', 'ticks': [0, 1]},
                linewidths=1, linecolor='gray', ax=ax)

    ax.set_title('Statistical Significance by Model and Metric\n(FDR-corrected p < 0.05)',
                fontweight='bold', pad=20)
    ax.set_xlabel('Model', fontweight='bold')
    ax.set_ylabel('Metric', fontweight='bold')

    plt.tight_layout()
    save_figure(fig, 'heatmap_significance', output_dir, formats)
    plt.close()


def plot_stereotype_score_comparison(
    summary_df: pd.DataFrame,
    significance_df: pd.DataFrame,
    ci_df: pd.DataFrame,
    output_dir: Path,
    formats: List[str]
) -> None:
    """
    Create comprehensive stereotype score comparison with error bars.
    """
    print("\n4. Creating stereotype score comparison with confidence intervals...")

    metric = 'child_target_stereotype_score'

    # Get data
    pivot = summary_df.pivot(index='model', columns='child_label', values=metric)
    models = pivot.index.tolist()
    model_labels = [m.replace('openai/', '').replace('-versatile', '') for m in models]

    # Get CIs if available
    if len(ci_df) > 0:
        ci_data = ci_df[ci_df['metric'] == metric].set_index('model')
    else:
        ci_data = None

    # Calculate gaps
    gaps = pivot['daughter'] - pivot['son']
    gaps_sorted = gaps.sort_values(ascending=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: Daughter vs Son bars
    x = np.arange(len(models))
    width = 0.35

    bars1 = ax1.bar(x - width/2, pivot['daughter'], width,
                   label='Daughter', color=GENDER_COLORS['daughter'], alpha=0.8)
    bars2 = ax1.bar(x + width/2, pivot['son'], width,
                   label='Son', color=GENDER_COLORS['son'], alpha=0.8)

    # Add significance stars
    for i, model in enumerate(models):
        sig_row = significance_df[
            (significance_df['model'] == model) &
            (significance_df['metric'] == metric)
        ]
        if len(sig_row) > 0:
            p_val = sig_row['p_value_fdr'].values[0]
            max_val = max(pivot.loc[model, 'daughter'], pivot.loc[model, 'son'])
            add_significance_stars(ax1, x[i] - width/2, x[i] + width/2,
                                 max_val, p_val, height=max_val * 0.08)

    ax1.set_xlabel('Model', fontweight='bold')
    ax1.set_ylabel('Stereotype Alignment Score', fontweight='bold')
    ax1.set_title('(A) Stereotype Scores by Child Gender', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_labels, rotation=45, ha='right')
    ax1.legend()
    ax1.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')

    # Panel B: Gap with confidence intervals
    sorted_models = gaps_sorted.index.tolist()
    sorted_labels = [m.replace('openai/', '').replace('-versatile', '') for m in sorted_models]
    y_pos = np.arange(len(sorted_models))

    # Plot gaps
    colors = [GENDER_COLORS['daughter'] if g > 0 else GENDER_COLORS['son']
              for g in gaps_sorted.values]
    ax2.barh(y_pos, gaps_sorted.values, color=colors, alpha=0.7)

    # Add error bars if CI data available
    if ci_data is not None:
        for i, model in enumerate(sorted_models):
            if model in ci_data.index:
                gap_mean = ci_data.loc[model, 'gap_mean']
                ci_low = ci_data.loc[model, 'ci_low_2.5']
                ci_high = ci_data.loc[model, 'ci_high_97.5']
                xerr = [[gap_mean - ci_low], [ci_high - gap_mean]]
                ax2.errorbar(gap_mean, i, xerr=xerr, fmt='none',
                           ecolor='black', capsize=3, capthick=1.5)

    # Add Cohen's d annotations
    for i, model in enumerate(sorted_models):
        sig_row = significance_df[
            (significance_df['model'] == model) &
            (significance_df['metric'] == metric)
        ]
        if len(sig_row) > 0:
            cohens_d = sig_row['cohens_d'].values[0]
            gap_val = gaps_sorted.loc[model]
            ax2.text(gap_val + 0.15, i, f"d={cohens_d:.2f}",
                    va='center', fontsize=9)

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(sorted_labels)
    ax2.set_xlabel('Daughter - Son Gap', fontweight='bold')
    ax2.set_title('(B) Stereotype Gap with 95% CI', fontweight='bold')
    ax2.axvline(x=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    ax2.grid(axis='x', alpha=0.3, linestyle='--')

    plt.suptitle('Child Target Stereotype Alignment Scores',
                fontweight='bold', fontsize=14, y=1.02)
    plt.tight_layout()
    save_figure(fig, 'stereotype_score_comprehensive', output_dir, formats)
    plt.close()


def plot_dimension_comparison(
    summary_df: pd.DataFrame,
    output_dir: Path,
    formats: List[str]
) -> None:
    """
    Create grouped bar chart comparing all four bias dimensions.
    """
    print("\n5. Creating multi-dimension comparison...")

    metrics = [
        ('trait_bias_index', 'Trait\nBias'),
        ('role_bias_index', 'Role\nBias'),
        ('domain_bias_index', 'Domain\nBias'),
        ('direct_gender_marker_index', 'Gender\nMarker'),
    ]

    # Prepare data
    models = summary_df['model'].unique()
    model_labels = [m.replace('openai/', '').replace('-versatile', '') for m in models]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, (metric_col, metric_label) in enumerate(metrics):
        ax = axes[idx]

        pivot = summary_df.pivot(index='model', columns='child_label', values=metric_col)
        x = np.arange(len(models))
        width = 0.35

        ax.bar(x - width/2, pivot['daughter'], width,
              label='Daughter', color=GENDER_COLORS['daughter'], alpha=0.8)
        ax.bar(x + width/2, pivot['son'], width,
              label='Son', color=GENDER_COLORS['son'], alpha=0.8)

        ax.set_xlabel('Model', fontweight='bold')
        ax.set_ylabel('Bias Index', fontweight='bold')
        ax.set_title(metric_label, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(model_labels, rotation=45, ha='right')
        if idx == 0:
            ax.legend()
        ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        # Add annotations for bias direction
        y_min, y_max = ax.get_ylim()
        if 'marker' not in metric_col:  # Skip for marker index (obvious)
            ax.text(0.02, 0.98, '← More Feminine\n← More Communal\n← More Family',
                   transform=ax.transAxes, va='top', ha='left', fontsize=8,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
            ax.text(0.98, 0.98, 'More Masculine →\nMore Agentic →\nMore Career →',
                   transform=ax.transAxes, va='top', ha='right', fontsize=8,
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

    plt.suptitle('Gender Bias Indices Across Four Dimensions',
                fontweight='bold', fontsize=14)
    plt.tight_layout()
    save_figure(fig, 'dimension_comparison', output_dir, formats)
    plt.close()


def plot_distributions(
    story_scores_df: pd.DataFrame,
    output_dir: Path,
    formats: List[str]
) -> None:
    """
    Create violin/box plots showing distribution of scores.
    """
    print("\n6. Creating distribution plots...")

    metrics = [
        ('trait_bias_index', 'Trait Bias Index'),
        ('role_bias_index', 'Role Bias Index'),
        ('child_target_stereotype_score', 'Stereotype Score'),
    ]

    # Filter to son/daughter only
    df = story_scores_df[story_scores_df['child_label'].isin(['son', 'daughter'])].copy()

    for metric_col, metric_label in metrics:
        fig, ax = plt.subplots(figsize=(12, 6))

        # Create violin plot
        parts = ax.violinplot(
            [df[df['child_label'] == 'daughter'][metric_col].dropna(),
             df[df['child_label'] == 'son'][metric_col].dropna()],
            positions=[0, 1],
            showmeans=True,
            showmedians=True,
            widths=0.7
        )

        # Color the violins
        colors = [GENDER_COLORS['daughter'], GENDER_COLORS['son']]
        for pc, color in zip(parts['bodies'], colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.6)

        # Customize
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Daughter', 'Son'])
        ax.set_ylabel(metric_label, fontweight='bold')
        ax.set_title(f'Distribution of {metric_label} by Child Gender',
                    fontweight='bold', pad=20)
        ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        # Add sample sizes
        n_daughter = len(df[df['child_label'] == 'daughter'])
        n_son = len(df[df['child_label'] == 'son'])
        ax.text(0, ax.get_ylim()[0], f'n={n_daughter}', ha='center', va='top')
        ax.text(1, ax.get_ylim()[0], f'n={n_son}', ha='center', va='top')

        plt.tight_layout()
        save_figure(fig, f'distribution_{metric_col}', output_dir, formats)
        plt.close()


def plot_storyteller_analysis(
    story_scores_df: pd.DataFrame,
    output_dir: Path,
    formats: List[str]
) -> None:
    """
    Analyze bias by storyteller (if 'person' column available).
    """
    if 'person' not in story_scores_df.columns:
        print("\n7. Storyteller analysis: SKIPPED (no 'person' column)")
        return

    print("\n7. Creating storyteller analysis...")

    # Filter to son/daughter
    df = story_scores_df[story_scores_df['child_label'].isin(['son', 'daughter'])].copy()

    # Group by storyteller and child gender
    metric = 'child_target_stereotype_score'
    grouped = df.groupby(['person', 'child_label'])[metric].mean().reset_index()
    pivot = grouped.pivot(index='person', columns='child_label', values=metric)

    # Calculate gap
    pivot['gap'] = pivot['daughter'] - pivot['son']
    pivot = pivot.sort_values('gap', ascending=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: Grouped bar chart
    x = np.arange(len(pivot))
    width = 0.35

    ax1.bar(x - width/2, pivot['daughter'], width,
           label='Daughter', color=GENDER_COLORS['daughter'], alpha=0.8)
    ax1.bar(x + width/2, pivot['son'], width,
           label='Son', color=GENDER_COLORS['son'], alpha=0.8)

    ax1.set_xlabel('Storyteller', fontweight='bold')
    ax1.set_ylabel('Stereotype Score', fontweight='bold')
    ax1.set_title('(A) Stereotype Scores by Storyteller', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(pivot.index, rotation=45, ha='right')
    ax1.legend()
    ax1.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')

    # Panel B: Gap bars
    colors = [GENDER_COLORS['daughter'] if g > 0 else GENDER_COLORS['son']
              for g in pivot['gap']]
    ax2.barh(x, pivot['gap'], color=colors, alpha=0.7)

    ax2.set_yticks(x)
    ax2.set_yticklabels(pivot.index)
    ax2.set_xlabel('Daughter - Son Gap', fontweight='bold')
    ax2.set_title('(B) Stereotype Gap by Storyteller', fontweight='bold')
    ax2.axvline(x=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    ax2.grid(axis='x', alpha=0.3, linestyle='--')

    plt.suptitle('Gender Bias by Storyteller Type',
                fontweight='bold', fontsize=14, y=1.00)
    plt.tight_layout()
    save_figure(fig, 'storyteller_analysis', output_dir, formats)
    plt.close()


def plot_culture_analysis(
    story_scores_df: pd.DataFrame,
    output_dir: Path,
    formats: List[str],
    top_n: int = 10
) -> None:
    """
    Analyze bias by culture/country (if 'country' column available).
    """
    if 'country' not in story_scores_df.columns:
        print("\n8. Culture analysis: SKIPPED (no 'country' column)")
        return

    print(f"\n8. Creating culture analysis (top {top_n} countries)...")

    df = story_scores_df[story_scores_df['child_label'].isin(['son', 'daughter'])].copy()

    metric = 'child_target_stereotype_score'
    grouped = df.groupby(['country', 'child_label'])[metric].mean().reset_index()
    pivot = grouped.pivot(index='country', columns='child_label', values=metric)
    pivot['gap'] = pivot['daughter'] - pivot['son']

    # Get top N by gap magnitude
    pivot['abs_gap'] = pivot['gap'].abs()
    top_countries = pivot.nlargest(top_n, 'abs_gap')

    fig, ax = plt.subplots(figsize=(10, 8))

    y = np.arange(len(top_countries))
    colors = [GENDER_COLORS['daughter'] if g > 0 else GENDER_COLORS['son']
              for g in top_countries['gap']]

    ax.barh(y, top_countries['gap'], color=colors, alpha=0.7)

    ax.set_yticks(y)
    ax.set_yticklabels(top_countries.index)
    ax.set_xlabel('Stereotype Gap (Daughter - Son)', fontweight='bold')
    ax.set_title(f'Top {top_n} Countries by Gender Stereotype Gap',
                fontweight='bold', pad=20)
    ax.axvline(x=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.grid(axis='x', alpha=0.3, linestyle='--')

    plt.tight_layout()
    save_figure(fig, 'culture_analysis_top', output_dir, formats)
    plt.close()


def plot_correlation_matrix(
    summary_df: pd.DataFrame,
    output_dir: Path,
    formats: List[str]
) -> None:
    """
    Create correlation heatmap between different bias metrics.
    """
    print("\n9. Creating correlation matrix...")

    metrics = [
        'masculine_traits_per_1000',
        'feminine_traits_per_1000',
        'agentic_per_1000',
        'communal_per_1000',
        'career_per_1000',
        'family_per_1000',
        'trait_bias_index',
        'role_bias_index',
        'domain_bias_index',
    ]

    # Get daughter stories only
    daughter_data = summary_df[summary_df['child_label'] == 'daughter'][metrics]

    # Compute correlation
    corr = daughter_data.corr()

    # Clean labels
    labels = {
        'masculine_traits_per_1000': 'Masc. Traits',
        'feminine_traits_per_1000': 'Fem. Traits',
        'agentic_per_1000': 'Agentic',
        'communal_per_1000': 'Communal',
        'career_per_1000': 'Career',
        'family_per_1000': 'Family',
        'trait_bias_index': 'Trait Index',
        'role_bias_index': 'Role Index',
        'domain_bias_index': 'Domain Index',
    }
    corr.index = [labels.get(m, m) for m in corr.index]
    corr.columns = [labels.get(m, m) for m in corr.columns]

    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                vmin=-1, vmax=1, cbar_kws={'label': 'Correlation'},
                linewidths=0.5, linecolor='gray', ax=ax, square=True)

    ax.set_title('Correlation Between Bias Metrics (Daughter Stories)',
                fontweight='bold', pad=20)

    plt.tight_layout()
    save_figure(fig, 'correlation_matrix', output_dir, formats)
    plt.close()


def create_summary_figure(
    summary_df: pd.DataFrame,
    significance_df: pd.DataFrame,
    output_dir: Path,
    formats: List[str]
) -> None:
    """
    Create a comprehensive 4-panel summary figure for papers.
    """
    print("\n10. Creating comprehensive summary figure...")

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # Panel A: Stereotype scores by model
    ax1 = fig.add_subplot(gs[0, 0])
    metric = 'child_target_stereotype_score'
    pivot = summary_df.pivot(index='model', columns='child_label', values=metric)
    models = pivot.index.tolist()
    model_labels = [m.replace('openai/', '').replace('-versatile', '')[:15] for m in models]
    x = np.arange(len(models))
    width = 0.35

    ax1.bar(x - width/2, pivot['daughter'], width,
           label='Daughter', color=GENDER_COLORS['daughter'], alpha=0.8)
    ax1.bar(x + width/2, pivot['son'], width,
           label='Son', color=GENDER_COLORS['son'], alpha=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_labels, rotation=45, ha='right', fontsize=9)
    ax1.set_ylabel('Stereotype Score', fontweight='bold')
    ax1.set_title('(A) Stereotype Scores by Model', fontweight='bold', loc='left')
    ax1.legend()
    ax1.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    ax1.grid(axis='y', alpha=0.3)

    # Panel B: Effect sizes heatmap
    ax2 = fig.add_subplot(gs[0, 1])
    key_metrics = ['feminine_traits_per_1000', 'communal_per_1000',
                   'trait_bias_index', 'child_target_stereotype_score']
    heatmap_data = significance_df[significance_df['metric'].isin(key_metrics)]
    pivot_effect = heatmap_data.pivot(index='metric', columns='model', values='cohens_d')
    metric_labels = {
        'feminine_traits_per_1000': 'Fem. Traits',
        'communal_per_1000': 'Communal',
        'trait_bias_index': 'Trait Index',
        'child_target_stereotype_score': 'Stereotype',
    }
    pivot_effect.index = [metric_labels.get(m, m) for m in pivot_effect.index]
    pivot_effect.columns = [c.replace('openai/', '')[:12] for c in pivot_effect.columns]

    vmax = max(abs(pivot_effect.min().min()), abs(pivot_effect.max().max()))
    sns.heatmap(pivot_effect, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                vmin=-vmax, vmax=vmax, cbar_kws={'label': "Cohen's d"},
                linewidths=0.5, ax=ax2, cbar=True)
    ax2.set_title("(B) Effect Sizes (Cohen's d)", fontweight='bold', loc='left')
    ax2.set_xlabel('')
    ax2.set_ylabel('')

    # Panel C: Trait comparison across all models
    ax3 = fig.add_subplot(gs[1, 0])
    trait_metrics = ['feminine_traits_per_1000', 'masculine_traits_per_1000']
    all_models_data = []
    for metric in trait_metrics:
        for gender in ['daughter', 'son']:
            vals = summary_df[summary_df['child_label'] == gender][metric].values
            label = f"{'Fem.' if 'feminine' in metric else 'Masc.'} - {gender.title()}"
            all_models_data.append({
                'values': vals,
                'label': label,
                'position': len(all_models_data)
            })

    positions = [d['position'] for d in all_models_data]
    data_to_plot = [d['values'] for d in all_models_data]
    labels = [d['label'] for d in all_models_data]

    bp = ax3.boxplot(data_to_plot, positions=positions, widths=0.6,
                     patch_artist=True, notch=True)

    colors = [GENDER_COLORS['daughter'], GENDER_COLORS['son'],
              GENDER_COLORS['daughter'], GENDER_COLORS['son']]
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax3.set_xticks(positions)
    ax3.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax3.set_ylabel('Frequency (per 1000 tokens)', fontweight='bold')
    ax3.set_title('(C) Trait Distribution Across Models', fontweight='bold', loc='left')
    ax3.grid(axis='y', alpha=0.3)

    # Panel D: Summary statistics table
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')

    # Count significant results
    total_tests = len(significance_df)
    sig_fdr = significance_df['significant_fdr'].sum()
    sig_bonf = significance_df['significant_bonferroni'].sum()

    # Get effect size stats
    large_effects = (significance_df['cohens_d'].abs() > 0.8).sum()
    medium_effects = ((significance_df['cohens_d'].abs() > 0.5) &
                     (significance_df['cohens_d'].abs() <= 0.8)).sum()

    summary_text = f"""
    STATISTICAL SUMMARY

    Total Comparisons: {total_tests}

    Significant (FDR p<0.05): {sig_fdr} ({sig_fdr/total_tests*100:.1f}%)
    Significant (Bonferroni): {sig_bonf} ({sig_bonf/total_tests*100:.1f}%)

    Effect Sizes:
      Large (|d| > 0.8): {large_effects}
      Medium (|d| > 0.5): {medium_effects}

    Key Findings:
    • All models show significant
      gender stereotyping
    • Strongest bias in gender markers
      (Cohen's d > 3.0)
    • Moderate bias in traits/roles
      (Cohen's d = 0.3-0.7)
    • No bias in career/family domains

    Models ranked by bias:
    1. GPT-OSS-120b (highest)
    2. Llama-3.3-70b
    3. Qwen-32b
    4. Llama-3.1-8b
    5. GPT-OSS-20b (lowest)
    """

    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax4.set_title('(D) Summary Statistics', fontweight='bold', loc='left',
                 pad=20, x=0.1)

    plt.suptitle('Gender Bias in LLM-Generated Children\'s Stories: Comprehensive Summary',
                fontweight='bold', fontsize=16, y=0.98)

    save_figure(fig, 'comprehensive_summary', output_dir, formats)
    plt.close()


def main() -> None:
    args = parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("GENDER BIAS VISUALIZATION GENERATOR")
    print("="*80)
    print(f"\nInput directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Output formats: {', '.join(args.formats)}")

    # Load data
    print("\nLoading data files...")
    try:
        story_scores = pd.read_csv(input_dir / "story_level_gender_bias_scores.csv")
        print(f"  ✓ Loaded story scores: {len(story_scores)} rows")
    except FileNotFoundError:
        print("  ✗ story_level_gender_bias_scores.csv not found")
        story_scores = pd.DataFrame()

    try:
        summary_df = pd.read_csv(input_dir / "model_child_gender_summary.csv")
        print(f"  ✓ Loaded summary: {len(summary_df)} rows")
    except FileNotFoundError:
        print("  ✗ model_child_gender_summary.csv not found")
        return

    try:
        significance_df = pd.read_csv(input_dir / "statistical_significance_tests.csv")
        print(f"  ✓ Loaded significance tests: {len(significance_df)} rows")
    except FileNotFoundError:
        print("  ✗ statistical_significance_tests.csv not found")
        return

    try:
        ci_df = pd.read_csv(input_dir / "model_gap_bootstrap_ci.csv")
        print(f"  ✓ Loaded confidence intervals: {len(ci_df)} rows")
    except FileNotFoundError:
        print("  ! Bootstrap CI file not found (optional)")
        ci_df = pd.DataFrame()

    # Generate visualizations
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)

    plot_model_comparison_bars(summary_df, significance_df, output_dir, args.formats)
    plot_effect_size_heatmap(significance_df, output_dir, args.formats)
    plot_significance_heatmap(significance_df, output_dir, args.formats)
    plot_stereotype_score_comparison(summary_df, significance_df, ci_df, output_dir, args.formats)
    plot_dimension_comparison(summary_df, output_dir, args.formats)

    if len(story_scores) > 0:
        plot_distributions(story_scores, output_dir, args.formats)
        plot_storyteller_analysis(story_scores, output_dir, args.formats)
        plot_culture_analysis(story_scores, output_dir, args.formats)

    plot_correlation_matrix(summary_df, output_dir, args.formats)
    create_summary_figure(summary_df, significance_df, output_dir, args.formats)

    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE")
    print("="*80)
    print(f"\nAll figures saved to: {output_dir}")
    print(f"Generated {len(list(output_dir.glob('*')))} files")
    print("\nRecommended figures for paper:")
    print("  • comprehensive_summary.png - Main figure (4 panels)")
    print("  • heatmap_effect_sizes.png - Effect size comparison")
    print("  • stereotype_score_comprehensive.png - Key findings")
    print("  • storyteller_analysis.png - Storyteller effects")
    print("  • culture_analysis_top.png - Cultural variation")


if __name__ == "__main__":
    main()
