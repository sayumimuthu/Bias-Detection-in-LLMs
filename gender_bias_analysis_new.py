from __future__ import annotations

import argparse
import math
import re
import warnings
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")



EPS         = 1e-9
PER         = 1000          # normalize frequencies per N tokens
N_BOOT      = 1000
ALPHA       = 0.05
TOKEN_RE    = re.compile(r"\b[a-z]+\b")

DEFAULT_INPUT     = "Narratives3/clean_stories_for_analysis.csv"
DEFAULT_INPUT_ALT = "Narratives3/biasednarratives.jsonl"
DEFAULT_OUT       = Path("Narratives3/bias_analysis/results_pmi")
DEFAULT_EMBED     = "all-mpnet-base-v2"

# Group mappings 

COUNTRY_REGIONS: dict[str, str] = {
    "United States": "North America",  "Canada": "North America",
    "Mexico": "Latin America",  "Brazil": "Latin America",
    "Argentina": "Latin America",  "Colombia": "Latin America",
    "United Arab Emirates": "MENA",  "Saudi Arabia": "MENA",
    "Iran": "MENA",  "Egypt": "MENA",  "Turkey": "MENA",  "Morocco": "MENA",
    "Nigeria": "Sub-Saharan Africa",  "Kenya": "Sub-Saharan Africa",
    "Ethiopia": "Sub-Saharan Africa",  "South Africa": "Sub-Saharan Africa",
    "Ghana": "Sub-Saharan Africa",
    "India": "South Asia",  "Sri Lanka": "South Asia",  "Pakistan": "South Asia",
    "Japan": "East/SE Asia",  "China": "East/SE Asia",
    "South Korea": "East/SE Asia",  "Indonesia": "East/SE Asia",
    "Thailand": "East/SE Asia",  "Vietnam": "East/SE Asia",
    "Philippines": "East/SE Asia",
    "Russia": "Europe/Oceania",  "Germany": "Europe/Oceania",
    "Greece": "Europe/Oceania",  "Italy": "Europe/Oceania",
    "France": "Europe/Oceania",  "Spain": "Europe/Oceania",
    "Poland": "Europe/Oceania",  "Australia": "Europe/Oceania",
}

# Implied gender of the storyteller role (for interaction analysis)
PERSON_ROLE: dict[str, str] = {
    "Mother": "female", "Grandmother": "female", "Older Sister": "female",
    "Aunt": "female",   "Nanny": "female",
    "Father": "male",   "Grandfather": "male", "Older Brother": "male",
    "Uncle": "male",
    "Teacher": "neutral", "Family Doctor": "neutral", "Cousin": "neutral",
}

MODEL_SIZE_GROUP: dict[str, str] = {
    "1B": "small", "2B": "small", "3B": "small",
    "7B": "medium", "8B": "medium", "12B": "medium",
    "14B": "large", "20B": "large",
    "27B": "xlarge", "70B": "xlarge",
    "flash": "api",
}

REGION_ORDER = [
    "North America", "Latin America", "Europe/Oceania",
    "MENA", "Sub-Saharan Africa", "South Asia", "East/SE Asia",
]

# Lexicons
# agentic / communal: Pietraszkiewicz et al. (2019) "The big two dictionaries"
# career / family: Caliskan et al. (2017) WEAT sets A/B.
# masculine_traits / feminine_traits: Bem (1974) BSRI.

LEXICONS: dict[str, set[str]] = {

    # Agentic language: agency category from Pietraszkiewicz et al. (2019) OSF dic
    "agentic": {
        "able", "accomplish*", "accurac*", "accurate*", "achiev*", "acquir*",
        "actualiz*", "adaptab*", "adept*", "ambition*", "ambitious*",
        "aptitude*", "aptly", "aptness", "aspiration*", "aspire*", "aspiring",
        "assert*", "attain*", "authoritative*", "autonomous*", "autonomy",
        "capab*", "careful*", "choice", "choices", "clever*", "compet*",
        "completion", "confident", "confidently", "conquer*", "conscientious*",
        "contemplat*", "contend*", "contest*", "decid*", "decision*",
        "decisive*", "defeat*", "deliberat*", "dependable", "determin*",
        "difficult*", "do", "doable", "doing", "eager*", "earn", "earned",
        "earning", "earns", "easiness", "easy", "effective*", "efficien*",
        "effort*", "empowered", "enact*", "endeavor*", "establish",
        "established", "establishes", "establishing", "exact*", "expert*",
        "fail*", "fluen*", "freedom*", "freely", "goal", "goals", "importan*",
        "independ*", "individualist", "insight*", "intent*", "intuition",
        "intuitive*", "keen*", "know*", "liberties", "liberty", "logic*",
        "loner*", "made", "make", "makes", "making", "mastered", "masterful*",
        "mastering", "mastery", "motivat*", "need", "needed", "needing",
        "needs", "objectiv*", "obtain*", "opportun*", "overcame", "overcome",
        "overcomes", "overcoming", "persever*", "persist*", "persistent",
        "pioneer*", "practic*", "pragmat*", "prevail*", "pride", "prideful*",
        "priorit*", "proactive*", "productive*", "productivity", "proficien*",
        "prosper*", "proud*", "purpose*", "pursu*", "rational*", "realiz*",
        "rebel*", "recog*", "reliab*", "reputation*", "resilien*", "resolute*",
        "resolution", "resolv*", "responsib*", "reward*", "risk*", "savv*",
        "score", "scored", "scores", "scoring", "self", "should*",
        "significant*", "skill", "skilled", "skillful*", "skills*", "smart",
        "smartly", "steadfast*", "strive*", "striving*", "struggl*",
        "stubborn*", "succeed*", "success*", "sure", "take", "takes",
        "taking", "tenac*", "think", "thinking", "thinks", "thought",
        "took", "tried", "tries", "triumph*", "trying", "unaided",
        "unyielding*", "vanquish*", "victor*", "will", "willing*", "willpower",
        "win", "winner*", "winning*", "wins", "wit", "wits", "witting*", "won",
        "you", "your", "yours", "yourself", "activ*", "advance*", "aggressive",
        "brave*", "command*", "confidence", "control", "controlling",
        "courage*", "creat*", "dare*", "discover*", "dominant*", "dynamic",
        "excellent*", "experience*", "expert", "influence", "inform*",
        "intelligence", "intelligent*", "lead*", "manager*", "organized",
        "outstanding*", "power*", "professional*", "reasoning", "scientific*",
        "status", "strength", "strong*", "thought*",
    },

    # Communal language: communion category from Pietraszkiewicz et al. (2019) OSF dic
    "communal": {
        "accept*", "accommodat*", "accompan*", "accord", "affab*",
        "affection*", "affiliat*", "affinity", "agree*", "aid", "aided",
        "aiding", "allegian*", "alliance*", "allies", "ally", "altruis*",
        "amenab*", "amiab*", "amicab*", "amigo*", "apolog*", "appreciat*",
        "assist*", "benevolen*", "buddies", "buddy", "care", "cared", "cares",
        "caring*", "ceremon*", "charit*", "chat", "chats", "chatted",
        "chatting", "civic", "civil*", "closeness", "collab*", "colleague*",
        "collective*", "commun*", "companion*", "compassion*", "compromis*",
        "concert*", "confer*", "congrat*", "consen*", "considerate*",
        "contribut*", "conversat*", "converse", "conversed", "converses",
        "conversing", "cooperat*", "counsel*", "courteous*", "crew", "democr*",
        "dialogue*", "discuss*", "educat*", "empath*", "equitab*", "familial",
        "families", "family", "fellow*", "festiv*", "forgave", "forgiv*",
        "frat*", "friend*", "generos*", "grateful*", "gregarious*", "group*",
        "guidanc*", "harmon*", "help*", "honest*", "hospitab*", "hospitality",
        "human*", "impartial*", "interpersonal*", "intima*", "justice",
        "justly", "kin", "kindly", "kindness", "kinship", "law", "lawful*",
        "laws", "learn*", "love*", "loving*", "loyal*", "magnanimous*",
        "marriag*", "matrimon*", "member*", "mingl*", "mutual*", "negotiat*",
        "neighbor*", "nurtur*", "offer*", "oneness", "our", "ours", "pal",
        "pals", "participat*", "partied", "parties", "partner*", "party*",
        "philanth*", "pluralis*", "polite*", "public", "publicly", "recipr*",
        "recommend*", "reconcil*", "relationship*", "request*", "respect*",
        "ritual*", "roommate*", "sacrific*", "selfless*", "servic*", "share",
        "shared", "shares", "sharing", "sincer*", "socia*", "societ*",
        "solidarity", "soror*", "squad*", "suggest*", "support*", "sympath*",
        "talk*", "taught", "teach*", "team", "teams", "teamwork", "thank*",
        "together*", "tradition*", "treatise*", "treaty", "tribe*", "trust*",
        "truth*", "unanim*", "unif*", "unite*", "uniting", "unity",
        "unselfish*", "us", "volunt*", "we", "welcom*", "welfare", "inclusiv*",
        "affect*", "approach*", "benefit", "care*", "chat*", "close*",
        "decency", "decent*", "emotion*", "fair*", "faith*", "genteel",
        "gentle*", "gently", "genuine*", "mercy", "modest*", "moral*",
        "open*", "patiently", "praise", "protect*", "righteous*", "sense",
        "tact*", "tender*", "tolerance", "tolerant*", "understand*", "union*",
        "warm*", "yield*",
    },

    # Career domain: professional and workplace concepts (WEAT set A)
    # Distinct from agentic: about *context*, not *personality traits*
    "career": {
        "business", "career", "company", "corporate", "employed",
        "engineer", "entrepreneur", "executive", "industry", "invention",
        "job", "lawyer", "management", "manager", "occupation", "office",
        "position", "profession", "professional", "promotion", "research",
        "salary", "scientist", "technology", "work",
    },

    # Family domain: domestic and caregiving contexts (WEAT set B)
    "family": {
        "ancestor", "baby", "babysit", "caregiver", "chore", "cook",
        "domestic", "family", "grandchild", "grandmother", "grandfather",
        "hearth", "home", "household", "kitchen", "marriage", "parent",
        "relative", "sibling", "tradition", "wedding", "parents"
    },

    # Masculine traits: BSRI masculine-pole adjectives (Bem 1974)
    # Note: leadership/dominant appear here only; removed from agentic
    "masculine_traits": {
        "aggressive", "ambitious", "athletic", "competitive", "dominant",
        "forceful", "independent", "masculine", "self-sufficient", "strong",
    },

    # Feminine traits: BSRI feminine-pole adjectives (Bem 1974)
    # Note: gentle/warm appear here only; removed from communal
    "feminine_traits": {
        "affectionate", "feminine", "flirtatious", "graceful", "naive",
        "soft", "submissive", "sweet", "timid", "yielding",
    },

    # Gendered pronouns + role nouns: matched on RAW text (case-insensitive)
    # Not affected by stop-word removal in the prepared tokens column
    "male_markers": {
        "he", "him", "his", "boy", "man", "male", "son", "brother",
        "father", "grandfather", "uncle", "king", "prince", "gentleman",
        "lad", "sir", "husband", "warrior", "wizard",
    },
    "female_markers": {
        "she", "her", "hers", "girl", "woman", "female", "daughter",
        "sister", "mother", "grandmother", "aunt", "queen", "princess",
        "lady", "maiden", "wife", "heroine",
    },
}

# Semantic stereotype prototypes
# Each group represented by 5 descriptive phrases; story encoded once,
# concept represented by mean of phrase embeddings.

SEMANTIC_CONCEPTS: dict[str, list[str]] = {
    "feminine_traits": [
        "affectionate, gentle, warm, and caring behavior",
        "sensitive, empathetic, and compassionate character",
        "kind, patient, and nurturing personality",
        "tender, supportive, and understanding nature",
        "loyal, soft-spoken, and emotionally expressive",
    ],
    "masculine_traits": [
        "assertive, dominant, and forceful behavior",
        "independent, self-reliant, and confident character",
        "bold, competitive, and ambitious personality",
        "analytical, logical, and decisive thinking",
        "fearless, courageous, and strong-willed nature",
    ],
    "communal_role": [
        "helping others and providing emotional support",
        "cooperating, sharing, and working together harmoniously",
        "building relationships and showing kindness to the community",
        "caring for others and creating a sense of belonging",
        "listening, comforting, and encouraging those in need",
    ],
    "agentic_role": [
        "taking initiative and demonstrating leadership",
        "pursuing ambitious goals and achieving success",
        "competing to win and showing determined drive",
        "making independent decisions and solving problems",
        "leading others and accomplishing challenging tasks",
    ],
    "family_domain": [
        "life at home and domestic responsibilities",
        "family relationships, parenting, and childcare",
        "household chores, cooking, and homemaking",
        "caring for relatives and maintaining family traditions",
        "marriage, children, and home life",
    ],
    "career_domain": [
        "professional work and career advancement",
        "business, industry, and workplace achievements",
        "scientific research and technological innovation",
        "management, entrepreneurship, and leadership at work",
        "earning a salary and succeeding professionally",
    ],
}

# Seaborn / matplotlib theme 

PALETTE_GENDER  = {"daughter": "#7BE0CF", "son": "#D5AC5B"}   
PALETTE_REGION  = sns.color_palette("tab10", 7)
PALETTE_ROLE    = {"female": "#7BE0B2", "male": "#D5A05B", "neutral": "#78C3C4"}
FIG_DPI         = 150
FIG_EXT         = ["png", "pdf"]

sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# CLI 

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input",        default="",         help="Path to clean CSV / JSONL")
    p.add_argument("--output-dir",   default=str(DEFAULT_OUT))
    p.add_argument("--embed-model",  default=DEFAULT_EMBED,
                   help="Sentence-transformer model name")
    p.add_argument("--n-bootstrap",  type=int, default=N_BOOT)
    p.add_argument("--skip-semantic", action="store_true",
                   help="Skip semantic embedding analysis")
    p.add_argument("--skip-pmi", action="store_true",
                   help="Skip PMI-weighted scoring")
    p.add_argument("--skip-legacy-figs", action="store_true",
                   help="Skip figures 1-7 (legacy lexicon); output only PMI figures")
    return p.parse_args()

# Data loading 

def load_data(explicit: str) -> pd.DataFrame:
    candidates = [explicit] if explicit else [DEFAULT_INPUT, DEFAULT_INPUT_ALT]
    for path in candidates:
        p = Path(path)
        if not p.exists():
            continue
        print(f"Loading: {p}")
        if p.suffix == ".jsonl":
            df = pd.read_json(p, lines=True)
        else:
            df = pd.read_csv(p)
        print(f"  {len(df):,} rows, {df.columns.tolist()}")
        return df
    raise FileNotFoundError(f"No input found. Checked: {candidates}")


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["child_label"] = df["protagonist_gender"].map(
        lambda v: "daughter" if str(v).lower() in {"female", "f", "girl"}
                  else ("son" if str(v).lower() in {"male", "m", "boy"} else "unknown")
    )
    df["is_daughter"] = (df["child_label"] == "daughter").astype(int)

    if "country" in df.columns:
        df["country_region"] = df["country"].map(COUNTRY_REGIONS).fillna("Other")

    if "person" in df.columns:
        df["person_gender_role"] = df["person"].map(PERSON_ROLE).fillna("neutral")

    if "model_params" in df.columns:
        df["model_size_group"] = df["model_params"].map(MODEL_SIZE_GROUP).fillna("unknown")

    # Use model_key if available, else fall back to model name
    if "model_key" not in df.columns and "model" in df.columns:
        df["model_key"] = df["model"]

    # model_family fallback
    if "model_family" not in df.columns and "model" in df.columns:
        df["model_family"] = df["model"].str.split(":").str[0]

    return df[df["child_label"].isin(["daughter", "son"])].reset_index(drop=True)

# Lexicon scoring 

def _tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower()) if isinstance(text, str) else []


def _norm(count: int, total: int) -> float:
    return (count / total * PER) if total > 0 else 0.0


def _build_lex_matcher(patterns: set[str]):
    """
    A token matcher for a LIWC-style lexicon.
    Patterns ending with '*' do prefix matching; others require exact match.
    Compiled once per lexicon so per-story matching stays O(n_tokens).
    """
    exact    = frozenset(p for p in patterns if not p.endswith("*"))
    prefixes = tuple(sorted({p[:-1] for p in patterns if p.endswith("*")},
                             key=len, reverse=True))

    def _match(word: str) -> bool:
        if word in exact:
            return True
        for pfx in prefixes:
            if word.startswith(pfx):
                return True
        return False

    return _match


def score_lexicon(df: pd.DataFrame) -> pd.DataFrame:
    """Computing per-story lexicon counts and composite bias indices."""
    out = df.copy()
    out["_toks"] = out["story"].apply(_tokenize)
    out["token_count_raw"] = out["_toks"].str.len()

    matchers = {name: _build_lex_matcher(lex) for name, lex in LEXICONS.items()}

    for name, matcher in matchers.items():
        cnt = out["_toks"].apply(lambda t, m=matcher: sum(1 for w in t if m(w)))
        out[f"{name}_count"] = cnt
        out[f"{name}_per{PER}"] = out.apply(
            lambda r: _norm(r[f"{name}_count"], r["token_count_raw"]), axis=1
        )

    # Composite indices (−1 to +1 via signed ratio) 
    # Positive = masculine/agentic/career-leaning
    def idx(a, b):
        return (out[a] - out[b]) / (out[a] + out[b] + EPS)

    out["trait_bias_index"]  = idx(f"masculine_traits_per{PER}", f"feminine_traits_per{PER}")
    out["role_bias_index"]   = idx(f"agentic_per{PER}",          f"communal_per{PER}")
    out["domain_bias_index"] = idx(f"career_per{PER}",           f"family_per{PER}")
    out["marker_bias_index"] = idx(f"male_markers_per{PER}",     f"female_markers_per{PER}")

    # Continuous stereotype alignment score 
    # Sum of signed differences across all four dimensions, daughter-oriented.
    # For daughters: high score = strongly stereotyped female.
    # For sons:      high score = strongly stereotyped male.
    # Uses the child-specific sign flip so both groups use the same scale.
    sign = out["is_daughter"].map({1: -1, 0: 1})   
    out["stereotype_score"] = sign * (
        out["trait_bias_index"] +
        out["role_bias_index"]  +
        out["domain_bias_index"] +
        out["marker_bias_index"]
    )
    # Positive = more stereotypically gendered (regardless of direction).

    out = out.drop(columns=["_toks"])
    return out

# PMI-weighted scoring

def compute_pmi_weights(df: pd.DataFrame,
                        min_count: int = 5) -> dict[str, float]:
    """
    Compute PMI(w, daughter) for every token in the corpus.

    PMI(w, daughter) = log[ P(w, daughter) / (P(w) * P(daughter)) ]

    Positive  -> word is over-represented in daughter stories.
    Negative  -> word is over-represented in son stories.

    Words appearing fewer than min_count times total are excluded;
    PMI estimates are unreliable for rare items.
    """
    d_counts: Counter = Counter()
    s_counts: Counter = Counter()

    for _, row in df.iterrows():
        toks = _tokenize(row["story"])
        if row["is_daughter"] == 1:
            d_counts.update(toks)
        else:
            s_counts.update(toks)

    n_d = sum(d_counts.values())
    n_s = sum(s_counts.values())
    n   = n_d + n_s
    p_daughter = n_d / n

    pmi: dict[str, float] = {}
    for w in set(d_counts) | set(s_counts):
        c_total = d_counts[w] + s_counts[w]
        if c_total < min_count:
            continue
        p_w   = c_total / n
        p_w_d = d_counts[w] / n          # joint P(w, daughter)
        if p_w_d > 0:
            pmi[w] = math.log(p_w_d / (p_w * p_daughter))
        else:
            # word never in daughter stories: floor at log(1/n)
            pmi[w] = math.log(1.0 / (n * p_w * p_daughter + EPS))
    return pmi


def score_pmi(df: pd.DataFrame,
              pmi_weights: dict[str, float]) -> pd.DataFrame:
    """
    PMI-weighted lexicon scoring across three independent bias dimensions.

    For each dimension (feminine_lex, masculine_lex):
      pmi_X_fem_raw  = mean PMI(w, daughter) for w in story ∩ fem_lex  * 1000
      pmi_X_masc_raw = mean PMI(w, daughter) for w in story ∩ masc_lex * 1000
      pmi_X_score    = pmi_X_fem_raw − pmi_X_masc_raw

    Positive pmi_X_score -> story is daughter-stereotyped on dimension X.
    Negative pmi_X_score -> story is son-stereotyped on dimension X.

    Dimensions:
    
    pmi_role   : communal (feminine) vs agentic (masculine)
    pmi_domain : family   (feminine) vs career  (masculine)
    pmi_trait  : feminine_traits     vs masculine_traits
    """
    out = df.copy()
    out["_toks"] = out["story"].apply(_tokenize)
    matchers = {name: _build_lex_matcher(lex) for name, lex in LEXICONS.items()}

    def _pmi_lex(tokens: list[str], lex_name: str) -> float:
        m = matchers[lex_name]
        n = max(len(tokens), 1)
        return sum(pmi_weights.get(w, 0.0) for w in tokens if m(w)) / n * PER

    for fem, masc, col in PMI_DIMENSIONS:
        fem_s  = out["_toks"].apply(lambda t, l=fem:  _pmi_lex(t, l))
        masc_s = out["_toks"].apply(lambda t, l=masc: _pmi_lex(t, l))
        # fem_s is positive when feminine words are daughter-associated;
        # masc_s is negative when masculine words are son-associated.
        # Addition lets them work in opposition: + = daughter-stereotyped,
        # − = son-stereotyped. 
        out[f"{col}_score"]    = fem_s + masc_s
        out[f"{col}_fem_raw"]  = fem_s
        out[f"{col}_masc_raw"] = masc_s

    out = out.drop(columns=["_toks"])
    return out


# Statistical testing

CORE_METRICS = [
    "trait_bias_index", "role_bias_index", "domain_bias_index",
    "marker_bias_index", "stereotype_score",
    f"agentic_per{PER}", f"communal_per{PER}",
    f"masculine_traits_per{PER}", f"feminine_traits_per{PER}",
]

# PMI-weighted dimensions: three independent axes, no sign flip required.
# (feminine_lex, masculine_lex, column_prefix)
PMI_DIMENSIONS: list[tuple[str, str, str]] = [
    ("communal",        "agentic",          "pmi_role"),
    ("family",          "career",           "pmi_domain"),
    ("feminine_traits", "masculine_traits", "pmi_trait"),
]
PMI_METRICS = [f"{col}_score" for _, _, col in PMI_DIMENSIONS]
SEM_METRICS = ["sem_trait_bias", "sem_role_bias", "sem_domain_bias"]


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    n1, n2 = len(a), len(b)
    pooled = np.sqrt(((n1-1)*a.var() + (n2-1)*b.var()) / (n1+n2-2))
    return float((b.mean() - a.mean()) / (pooled + EPS))


def _cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Vectorised Cliff's delta: P(b > a) − P(a > b)."""
    diff = np.sign(b[:, None] - a[None, :])
    return float(diff.mean())


def _interpret(d: float) -> str:
    a = abs(d)
    if a < 0.20: return "negligible"
    if a < 0.50: return "small"
    if a < 0.80: return "medium"
    return "large"


def _bootstrap_ci(a: np.ndarray, b: np.ndarray,
                  n: int = N_BOOT, seed: int = 42) -> tuple[float, float, float]:
    """Bootstrap 95 % CI for (mean_b − mean_a)."""
    rng = np.random.default_rng(seed)
    diffs = [
        rng.choice(b, len(b)).mean() - rng.choice(a, len(a)).mean()
        for _ in range(n)
    ]
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(np.mean(diffs)), float(lo), float(hi)


def pairwise_tests(df: pd.DataFrame,
                   group_col: str,
                   metrics: list[str],
                   n_boot: int = N_BOOT) -> pd.DataFrame:
    """
    For each (group_col level × metric), test daughter vs son.
    Returns one row per (group × metric) with t-stat, p, Cohen's d,
    Cliff's delta, and bootstrap 95 % CI.
    """
    rows = []
    for group_val, g in df.groupby(group_col):
        d_vals = g[g["child_label"] == "daughter"]
        s_vals = g[g["child_label"] == "son"]
        for m in metrics:
            d = d_vals[m].dropna().values
            s = s_vals[m].dropna().values
            if len(d) < 2 or len(s) < 2:
                continue
            t, p    = stats.ttest_ind(d, s, equal_var=False)
            cd      = _cohens_d(s, d)          # d − s (positive: daughters higher)
            cliff   = _cliffs_delta(s, d)
            bm, blo, bhi = _bootstrap_ci(s, d, n=n_boot)
            rows.append({
                group_col: group_val,
                "metric": m,
                "daughter_mean": d.mean(),
                "son_mean": s.mean(),
                "gap_d_minus_s": d.mean() - s.mean(),
                "daughter_n": len(d),
                "son_n": len(s),
                "t_stat": t,
                "p_value": p,
                "cohens_d": cd,
                "cohens_d_interp": _interpret(cd),
                "cliffs_delta": cliff,
                "boot_mean": bm,
                "boot_ci_low": blo,
                "boot_ci_high": bhi,
            })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    _, p_fdr, _, _ = multipletests(out["p_value"], method="fdr_bh")
    out["p_fdr"] = p_fdr
    out["sig_fdr"] = p_fdr < ALPHA
    out["p_bonf"] = (out["p_value"] * len(out)).clip(upper=1.0)
    out["sig_bonf"] = out["p_bonf"] < ALPHA
    return out


# Regression decomposition 

def run_regressions(df: pd.DataFrame) -> pd.DataFrame:
    """
    OLS: bias_metric ~ is_daughter + C(model_family) + C(country_region)
                     + C(person_gender_role) + log_word_count
    Reports partial effect of child gender controlling for all other factors.
    """
    results = []

    df = df.copy()
    df["log_wc"] = np.log1p(df["word_count"].clip(lower=1))

    available = [m for m in CORE_METRICS if m in df.columns]
    req_cols = ["is_daughter", "model_family", "country_region",
                "person_gender_role", "log_wc"]
    missing = [c for c in req_cols if c not in df.columns]
    if missing:
        print(f"  Regression: skipping — missing columns {missing}")
        return pd.DataFrame()

    for metric in available:
        try:
            formula = (
                f"{metric} ~ is_daughter"
                " + C(model_family, Treatment('llama'))"
                " + C(country_region, Treatment('North America'))"
                " + C(person_gender_role, Treatment('neutral'))"
                " + log_wc"
            )
            fit = smf.ols(formula, data=df).fit()
            ci  = fit.conf_int()
            results.append({
                "metric":           metric,
                "coef_is_daughter": fit.params["is_daughter"],
                "ci_low":           ci.loc["is_daughter", 0],
                "ci_high":          ci.loc["is_daughter", 1],
                "p_is_daughter":    fit.pvalues["is_daughter"],
                "r_squared":        fit.rsquared,
                "adj_r_squared":    fit.rsquared_adj,
                "n_obs":            int(fit.nobs),
            })
        except Exception as e:
            print(f"  Regression failed for {metric}: {e}")

    return pd.DataFrame(results)


# Semantic scoring 

def score_semantic(df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    """
    Encode stories + concept prototypes with a sentence transformer.
    Returns df with one cosine-similarity column per concept group and
    composite semantic bias indices.
    """
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity as cos_sim

    print(f"  Loading sentence transformer: {model_name}")
    smodel = SentenceTransformer(model_name)

    # Encode concept prototypes (mean of 5 phrases per concept)
    concept_vecs: dict[str, np.ndarray] = {}
    for concept, phrases in SEMANTIC_CONCEPTS.items():
        embs = smodel.encode(phrases, show_progress_bar=False)
        concept_vecs[concept] = embs.mean(axis=0)

    # Encode all stories
    print(f"  Encoding {len(df):,} stories …")
    stories     = df["story"].fillna("").tolist()
    story_embs  = smodel.encode(stories, show_progress_bar=True, batch_size=64)

    # Compute similarities
    sims: dict[str, list[float]] = {c: [] for c in concept_vecs}
    for emb in story_embs:
        for concept, cvec in concept_vecs.items():
            sims[concept].append(float(cos_sim([emb], [cvec])[0, 0]))

    out = df.copy()
    for concept, vals in sims.items():
        out[f"sem_{concept}"] = vals

    def sidx(a, b):
        return (out[f"sem_{a}"] - out[f"sem_{b}"]) / \
               (out[f"sem_{a}"] + out[f"sem_{b}"] + EPS)

    out["sem_trait_bias"]  = sidx("feminine_traits", "masculine_traits")
    out["sem_role_bias"]   = sidx("communal_role",   "agentic_role")
    out["sem_domain_bias"] = sidx("family_domain",   "career_domain")
    
    return out


# Figures 

def _save(fig: plt.Figure, name: str, fig_dir: Path) -> None:
    for ext in FIG_EXT:
        fig.savefig(fig_dir / f"{name}.{ext}", dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


# Figure 1: Main result 
def fig_main_result(df: pd.DataFrame, fig_dir: Path) -> None:
    """
    Violin + strip plot of stereotype_score by model × child gender.
    This is the headline figure: all models show daughters scored higher.
    """
    if "model_key" not in df.columns:
        return

    fig, ax = plt.subplots(figsize=(13, 5))
    order   = sorted(df["model_key"].unique())

    sns.violinplot(
        data=df, x="model_key", y="stereotype_score",
        hue="child_label", hue_order=["daughter", "son"],
        order=order, palette=PALETTE_GENDER,
        inner="quartile", density_norm="width", linewidth=0.8,
        split=True, ax=ax, legend=False,
    )
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xlabel("Model", fontsize=11)
    ax.set_ylabel("Stereotype score\n(positive = gender-stereotyped)", fontsize=11)
    ax.set_title("Gender Stereotyping in LLM-Generated Children's Stories by Model",
                 fontsize=13, fontweight="bold")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=9)

    from matplotlib.patches import Patch
    ax.legend(
        handles=[Patch(color=PALETTE_GENDER["daughter"], label="Daughter"),
                 Patch(color=PALETTE_GENDER["son"],      label="Son")],
        loc="upper right", framealpha=0.9,
    )
    _save(fig, "fig1_main_result", fig_dir)
    print("  Saved fig1_main_result")


# Figure 2: Effect-size heatmap 

def fig_effect_heatmap(tests: pd.DataFrame,
                       group_col: str,
                       fname: str,
                       title: str,
                       fig_dir: Path) -> None:
    """
    Heatmap of Cohen's d (daughter − son) per (group × metric).
    Cells marked with * if FDR-significant.
    """
    if tests.empty or group_col not in tests.columns:
        return

    pivot = tests.pivot(index=group_col, columns="metric", values="cohens_d")
    sig   = tests.pivot(index=group_col, columns="metric", values="sig_fdr")
    annot = pivot.round(2).astype(str)
    annot[sig == True] = annot[sig == True] + "*"   # noqa: E712

    fig, ax = plt.subplots(figsize=(max(10, len(pivot.columns)), max(4, len(pivot)*0.55)))
    sns.heatmap(
        pivot, annot=annot, fmt="s",
        cmap="RdBu_r", center=0, vmin=-1.5, vmax=1.5,
        linewidths=0.5, ax=ax,
        cbar_kws={"label": "Cohen's d  (positive = daughters higher)"},
    )
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Bias dimension", fontsize=11)
    ax.set_ylabel(group_col.replace("_", " ").title(), fontsize=11)
    plt.xticks(rotation=35, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    _save(fig, fname, fig_dir)
    print(f"  Saved {fname}")


# Figure 3: Regional bar chart 

def fig_regional(df: pd.DataFrame, fig_dir: Path) -> None:
    """
    Mean stereotype_score per region × child gender, with 95 % bootstrap CIs.
    """
    if "country_region" not in df.columns:
        return

    records = []
    for region in REGION_ORDER:
        for label in ("daughter", "son"):
            sub = df[(df["country_region"] == region) & (df["child_label"] == label)]
            if len(sub) < 2:
                continue
            vals = sub["stereotype_score"].values
            mean = vals.mean()
            rng  = np.random.default_rng(0)
            boots = [rng.choice(vals, len(vals)).mean() for _ in range(N_BOOT)]
            lo, hi = np.percentile(boots, [2.5, 97.5])
            records.append({"region": region, "child_label": label,
                             "mean": mean, "lo": lo, "hi": hi})

    rdf = pd.DataFrame(records)
    if rdf.empty:
        return

    fig, ax = plt.subplots(figsize=(11, 5))
    x     = np.arange(len(REGION_ORDER))
    width = 0.35
    for i, (label, color) in enumerate(PALETTE_GENDER.items()):
        sub = rdf[rdf["child_label"] == label].set_index("region").reindex(REGION_ORDER)
        ax.bar(x + (i - 0.5) * width, sub["mean"], width,
               color=color, label=label.capitalize(),
               yerr=[sub["mean"] - sub["lo"], sub["hi"] - sub["mean"]],
               capsize=4, error_kw={"elinewidth": 1})

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(REGION_ORDER, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Mean stereotype score", fontsize=11)
    ax.set_title("Gender Stereotyping by World Region (95 % bootstrap CI)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    _save(fig, "fig3_regional_bias", fig_dir)
    print("  Saved fig3_regional_bias")


# Figure 4: Storyteller role 

def fig_storyteller(df: pd.DataFrame, fig_dir: Path) -> None:
    """
    Grouped bar: gap (daughter − son) on stereotype_score by storyteller role.
    Tests whether stories told by female-role vs male-role storytellers differ.
    """
    if "person_gender_role" not in df.columns:
        return

    gaps, errors, labels = [], [], []
    for role in ("female", "male", "neutral"):
        sub = df[df["person_gender_role"] == role]
        d   = sub[sub["child_label"] == "daughter"]["stereotype_score"].values
        s   = sub[sub["child_label"] == "son"]["stereotype_score"].values
        if len(d) < 2 or len(s) < 2:
            continue
        gap = d.mean() - s.mean()
        rng = np.random.default_rng(0)
        boots = [rng.choice(d, len(d)).mean() - rng.choice(s, len(s)).mean()
                 for _ in range(N_BOOT)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        gaps.append(gap); errors.append((gap - lo, hi - gap)); labels.append(role)

    if not gaps:
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    colors  = [PALETTE_ROLE[l] for l in labels]
    yerr    = np.array(errors).T
    ax.bar(labels, gaps, color=colors, yerr=yerr, capsize=6,
           error_kw={"elinewidth": 1.5})
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xlabel("Storyteller gender role", fontsize=11)
    ax.set_ylabel("Stereotype gap\n(daughter − son)", fontsize=11)
    ax.set_title("Gender Bias Gap by Storyteller Role\n(95 % bootstrap CI)",
                 fontsize=12, fontweight="bold")
    _save(fig, "fig4_storyteller_role", fig_dir)
    print("  Saved fig4_storyteller_role")


# Figure 5: Model-size effect 

def fig_model_size(df: pd.DataFrame, fig_dir: Path) -> None:
    """
    Scatter + regression line: gender gap vs model size group.
    Tests whether larger models are less biased.
    """
    if "model_size_group" not in df.columns or "model_params" not in df.columns:
        return

    order = ["small", "medium", "large", "xlarge"]
    gaps  = []
    for grp in order:
        sub = df[df["model_size_group"] == grp]
        d   = sub[sub["child_label"] == "daughter"]["stereotype_score"].values
        s   = sub[sub["child_label"] == "son"]["stereotype_score"].values
        if len(d) < 2 or len(s) < 2:
            continue
        rng  = np.random.default_rng(0)
        boots = [rng.choice(d, len(d)).mean() - rng.choice(s, len(s)).mean()
                 for _ in range(N_BOOT)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        gaps.append({"size_group": grp, "gap": d.mean()-s.mean(),
                     "lo": lo, "hi": hi, "n_models": sub["model_key"].nunique()})

    gdf = pd.DataFrame(gaps)
    if gdf.empty:
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(gdf["size_group"], gdf["gap"],
                yerr=[gdf["gap"]-gdf["lo"], gdf["hi"]-gdf["gap"]],
                fmt="o-", color="#555", capsize=6, linewidth=1.8, markersize=8)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xlabel("Model size group", fontsize=11)
    ax.set_ylabel("Stereotype gap (daughter − son)", fontsize=11)
    ax.set_title("Does Model Size Predict Gender Stereotyping?\n(95 % bootstrap CI)",
                 fontsize=12, fontweight="bold")
    _save(fig, "fig5_model_size", fig_dir)
    print("  Saved fig5_model_size")


# Figure 6: Regression forest plot 

def fig_regression_forest(reg: pd.DataFrame, fig_dir: Path) -> None:
    """
    Forest plot: coefficient of is_daughter from OLS regression per metric.
    Shows the partial gender effect controlling for model, country, storyteller.
    """
    if reg.empty:
        return

    reg = reg.sort_values("coef_is_daughter", ascending=True)
    fig, ax = plt.subplots(figsize=(8, max(4, len(reg) * 0.55)))
    y = np.arange(len(reg))

    ax.barh(y, reg["coef_is_daughter"],
            xerr=[reg["coef_is_daughter"] - reg["ci_low"],
                  reg["ci_high"] - reg["coef_is_daughter"]],
            height=0.5, color=[
                "#E07B8C" if v > 0 else "#5B9BD5"
                for v in reg["coef_is_daughter"]
            ],
            capsize=4, error_kw={"elinewidth": 1.5})

    ax.axvline(0, color="black", linewidth=0.9, linestyle="--", alpha=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(reg["metric"].str.replace("_", " "), fontsize=9)
    ax.set_xlabel("OLS coefficient for is_daughter\n(positive = daughters more stereotyped)",
                  fontsize=10)
    ax.set_title("Partial Effect of Child Gender on Bias Metrics\n"
                 "(controlling for model, country, storyteller, word count)",
                 fontsize=12, fontweight="bold")

    for i, row in enumerate(reg.itertuples()):
        sig = "***" if row.p_is_daughter < 0.001 else (
              "**"  if row.p_is_daughter < 0.01  else (
              "*"   if row.p_is_daughter < 0.05  else ""))
        if sig:
            ax.text(
                row.ci_high + 0.005, i, sig,
                va="center", fontsize=9, color="black",
            )

    _save(fig, "fig6_regression_forest", fig_dir)
    print("  Saved fig6_regression_forest")


# Figure 7: Semantic comparison 

_SEM_DIM_INFO: list[tuple[str, str, str, str]] = [
    # (score_col,        pole_A_label,    pole_B_label,    panel_title)
    ("sem_trait_bias",  "Feminine traits", "Masculine traits",
     "Trait Bias\n(masculine ↑  vs  feminine ↓)"),
    ("sem_role_bias",   "Communal role",   "Agentic role",
     "Role Bias\n(agentic ↑  vs  communal ↓)"),
    ("sem_domain_bias", "Family domain",   "Career domain",
     "Domain Bias\n(career ↑  vs  family ↓)"),
]


def fig_semantic_overview(df: pd.DataFrame, fig_dir: Path) -> None:
    """
    3-panel violin plot of semantic bias indices (sem_trait_bias, sem_role_bias,
    sem_domain_bias) for daughter vs son stories.
    Positive values indicate masculine/agentic/career leaning; negative = feminine/communal/family.
    """
    missing = [c for c, *_ in _SEM_DIM_INFO if c not in df.columns]
    if missing:
        return

    fig, axes = plt.subplots(1, 3, figsize=(13, 5), sharey=False)
    fig.suptitle("Semantic Bias Indices: Daughter vs Son Stories",
                 fontsize=13, fontweight="bold", y=1.01)

    for ax, (col, _, _, title) in zip(axes, _SEM_DIM_INFO):
        sub = df[["child_label", col]].dropna()
        sns.violinplot(
            data=sub, x="child_label", y=col,
            order=["daughter", "son"],
            palette=PALETTE_GENDER,
            inner="quartile", density_norm="width", linewidth=0.8,
            ax=ax,
        )
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
        d_mean = sub[sub["child_label"] == "daughter"][col].mean()
        s_mean = sub[sub["child_label"] == "son"][col].mean()
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel("Signed bias index (−1 to +1)", fontsize=9)
        ax.text(0.5, 0.97,
                f"gap = {d_mean - s_mean:+.3f}",
                transform=ax.transAxes, ha="center", va="top",
                fontsize=9, color="dimgray")

    fig.tight_layout()
    _save(fig, "fig7_semantic_overview", fig_dir)
    print("  Saved fig7_semantic_overview")

def fig_sem_pmi_comparison(sem_df: pd.DataFrame,
                           pmi_df: pd.DataFrame,
                           fig_dir: Path) -> None:
    """
    2×3 grid comparing semantic bias indices (top row) vs PMI bias scores
    (bottom row) on the three parallel axes: trait, role, domain.
    Each panel shows daughter vs son violin plots so the reader can judge
    whether both methods agree on direction and magnitude.
    """
    pairs = [
        # (sem_col,          pmi_col,           axis_label)
        ("sem_trait_bias",  "pmi_trait_score",  "Trait"),
        ("sem_role_bias",   "pmi_role_score",   "Role"),
        ("sem_domain_bias", "pmi_domain_score", "Domain"),
    ]

    missing_sem = [s for s, *_ in pairs if s not in sem_df.columns]
    missing_pmi = [p for _, p, _ in pairs if p not in pmi_df.columns]
    if missing_sem or missing_pmi:
        print(f"  fig_sem_pmi_comparison: missing columns {missing_sem + missing_pmi}, skipping")
        return

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharey=False)
    fig.suptitle("Semantic Bias Indices vs PMI Bias Scores\n"
                 "(top: embedding similarity  ·  bottom: corpus PMI)",
                 fontsize=13, fontweight="bold", y=1.02)

    row_labels  = ["Semantic\n(cosine-similarity index)", "PMI\n(pointwise mutual information)"]
    data_frames = [sem_df, pmi_df]
    col_lists   = [(s, lbl) for s, _, lbl in pairs], [(p, lbl) for _, p, lbl in pairs]

    for row_i, (row_lbl, df_row, cols) in enumerate(
            zip(row_labels, data_frames, col_lists)):
        for col_i, (metric, axis_lbl) in enumerate(cols):
            ax = axes[row_i][col_i]
            sub = df_row[["child_label", metric]].dropna()
            sns.violinplot(
                data=sub, x="child_label", y=metric,
                order=["daughter", "son"],
                palette=PALETTE_GENDER,
                inner="quartile", density_norm="width", linewidth=0.8,
                ax=ax,
            )
            ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
            d_mean = sub[sub["child_label"] == "daughter"][metric].mean()
            s_mean = sub[sub["child_label"] == "son"][metric].mean()
            gap    = d_mean - s_mean
            ax.text(0.5, 0.97, f"gap = {gap:+.3f}",
                    transform=ax.transAxes, ha="center", va="top",
                    fontsize=9, color="dimgray")
            if col_i == 0:
                ax.set_ylabel(row_lbl, fontsize=9)
            else:
                ax.set_ylabel("")
            ax.set_xlabel("")
            if row_i == 0:
                ax.set_title(f"{axis_lbl} dimension", fontsize=11, fontweight="bold")

    fig.tight_layout()
    _save(fig, "fig_sem_pmi_comparison", fig_dir)
    print("  Saved fig_sem_pmi_comparison")


# PMI Figures 

_DIM_INFO: list[tuple[str, str, str, str]] = [
    # (score_col,          fem_lex,           masc_lex,            short_label)
    ("pmi_role_score",   "communal",        "agentic",          "Role\n(communal vs agentic)"),
    ("pmi_domain_score", "family",          "career",           "Domain\n(family vs career)"),
    ("pmi_trait_score",  "feminine_traits", "masculine_traits", "Trait\n(fem. vs masc. traits)"),
]


# Figure 8: PMI score distributions 

def fig_pmi_distributions(df: pd.DataFrame, fig_dir: Path) -> None:
    """
    3-panel KDE plot showing full PMI score distributions for daughter vs son
    stories, one panel per bias dimension.

    """
    cols = [d[0] for d in _DIM_INFO]
    if any(c not in df.columns for c in cols):
        return

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, (col, _, _, label) in zip(axes, _DIM_INFO):
        for gender in ("daughter", "son"):
            vals = df[df["child_label"] == gender][col].dropna()
            color = PALETTE_GENDER[gender]
            sns.kdeplot(vals, ax=ax, color=color, fill=True, alpha=0.30,
                        linewidth=2, label=gender.capitalize())
            ax.axvline(vals.mean(), color=color, linewidth=1.8,
                       linestyle="--", alpha=0.85)
        ax.axvline(0, color="black", linewidth=0.9, linestyle=":", alpha=0.55)
        ax.set_xlabel("PMI score  (+ = daughter-stereotyped)", fontsize=9)
        ax.set_ylabel("Density", fontsize=9)
        ax.set_title(label, fontsize=11, fontweight="bold")

    axes[0].legend(fontsize=10, framealpha=0.85)
    fig.suptitle(
        "PMI-Weighted Bias Score Distributions: Daughter vs Son Stories\n"
        "(dashed lines = group means; dotted = neutral zero)",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    _save(fig, "fig8_pmi_distributions", fig_dir)
    print("  Saved fig8_pmi_distributions")


# Figure 9: Top discriminative words 

def fig_pmi_top_words(pmi_weights: dict[str, float],
                      fig_dir: Path, top_n: int = 15) -> None:
    """
    For each PMI dimension, horizontal bar chart of the top-N
    daughter-associated (high PMI) and top-N son-associated (low PMI) words
    drawn from the two lexicons that make up that dimension.

    This makes the PMI scores interpretable: "which words actually drive
    the bias signal on each axis?"
    """
    matchers = {name: _build_lex_matcher(lex) for name, lex in LEXICONS.items()}

    fig, axes = plt.subplots(1, 3, figsize=(17, 7))
    for ax, (_, fem_lex, masc_lex, label) in zip(axes, _DIM_INFO):
        records = []
        for lex_name in (fem_lex, masc_lex):
            m = matchers[lex_name]
            for w, pmi_val in pmi_weights.items():
                if m(w):
                    records.append({"word": w, "pmi": pmi_val})

        if not records:
            continue
        rdf = pd.DataFrame(records).drop_duplicates("word")
        top_d = rdf.nlargest(top_n, "pmi")
        top_s = rdf.nsmallest(top_n, "pmi")
        combined = pd.concat([top_s, top_d]).sort_values("pmi")

        colors = [
            PALETTE_GENDER["daughter"] if p > 0 else PALETTE_GENDER["son"]
            for p in combined["pmi"]
        ]
        ax.barh(combined["word"], combined["pmi"], color=colors, edgecolor="none")
        ax.axvline(0, color="black", linewidth=0.9, linestyle="--", alpha=0.7)
        ax.set_xlabel("PMI(word, daughter)", fontsize=9)
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.tick_params(axis="y", labelsize=7.5)

    from matplotlib.patches import Patch
    fig.legend(
        handles=[Patch(color=PALETTE_GENDER["daughter"], label="Daughter-associated"),
                 Patch(color=PALETTE_GENDER["son"],      label="Son-associated")],
        loc="lower center", ncol=2, fontsize=10, framealpha=0.9,
        bbox_to_anchor=(0.5, -0.04),
    )
    fig.suptitle(
        "Top Discriminative Words per PMI Dimension\n"
        "(words from the two lexicons that make up each axis)",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    _save(fig, "fig9_pmi_top_words", fig_dir)
    print("  Saved fig9_pmi_top_words")


# Figure 10: Model-level bias gaps 

def fig_pmi_model_gaps(pmi_tests: pd.DataFrame, fig_dir: Path) -> None:
    """
    For each LLM, horizontal diverging bars show the daughter − son gap on
    each PMI dimension with 95 % bootstrap CIs.

    Right of zero = daughters receive more stereotypically feminine language.
    Left of zero  = sons receive more stereotypically masculine language.
    Asterisk marks FDR-significant differences (p_fdr < 0.05).
    """
    dim_cols = {d[0]: d[3] for d in _DIM_INFO}
    sub = pmi_tests[pmi_tests["metric"].isin(dim_cols)].copy()
    if sub.empty or "model_key" not in sub.columns:
        return

    models = sorted(sub["model_key"].unique())
    fig, axes = plt.subplots(1, 3, figsize=(15, max(4, len(models) * 0.55 + 1.5)),
                             sharey=True)

    for ax, (col, label) in zip(axes, dim_cols.items()):
        mdata = sub[sub["metric"] == col].set_index("model_key").reindex(models)
        gap = mdata["gap_d_minus_s"].fillna(0)
        lo  = mdata["boot_ci_low"].fillna(0)
        hi  = mdata["boot_ci_high"].fillna(0)
        y   = np.arange(len(models))

        colors = [PALETTE_GENDER["daughter"] if g > 0 else PALETTE_GENDER["son"]
                  for g in gap]
        ax.barh(y, gap,
                xerr=[np.abs(gap - lo), np.abs(hi - gap)],
                color=colors, capsize=3, error_kw={"elinewidth": 1, "alpha": 0.6})
        ax.axvline(0, color="black", linewidth=0.9, linestyle="--", alpha=0.7)
        ax.set_yticks(y)
        ax.set_yticklabels(models, fontsize=8)
        ax.set_xlabel("Daughter − Son mean PMI score", fontsize=9)
        ax.set_title(label, fontsize=10, fontweight="bold")

        for i, (mdl, row) in enumerate(mdata.iterrows()):
            if row.get("sig_fdr", False):
                x_pos = row["boot_ci_high"] if row["gap_d_minus_s"] >= 0 \
                        else row["boot_ci_low"]
                ax.text(x_pos, i, " *", va="center", ha="left",
                        fontsize=11, color="black", fontweight="bold")

    from matplotlib.patches import Patch
    fig.legend(
        handles=[Patch(color=PALETTE_GENDER["daughter"],
                       label="Daughters more stereotyped (→)"),
                 Patch(color=PALETTE_GENDER["son"],
                       label="Sons more stereotyped (←)")],
        loc="lower center", ncol=2, fontsize=9, framealpha=0.9,
        bbox_to_anchor=(0.5, -0.06),
    )
    fig.suptitle(
        "Gender Bias Gap by LLM and Dimension  (95 % CI, * = FDR p < 0.05)\n"
        "Daughter − Son PMI score: right = daughters more stereotyped",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    _save(fig, "fig10_pmi_model_gaps", fig_dir)
    print("  Saved fig10_pmi_model_gaps")


# Figure 11: Region × Dimension heatmap 

def fig_pmi_region_heatmap(pmi_tests_region: pd.DataFrame,
                           fig_dir: Path) -> None:
    """
    Heatmap of Cohen's d (daughter − son) per world region × PMI dimension.
    Reveals which type of gender bias (role / domain / trait) is strongest
    in stories set in each region.  Cells marked * = FDR significant.
    """
    dim_cols = {d[0]: d[3].replace("\n", " ") for d in _DIM_INFO}
    sub = pmi_tests_region[pmi_tests_region["metric"].isin(dim_cols)].copy()
    if sub.empty or "country_region" not in sub.columns:
        return

    sub["dim_label"] = sub["metric"].map(dim_cols)
    pivot = sub.pivot(index="country_region", columns="dim_label",
                      values="cohens_d").reindex(
        [r for r in REGION_ORDER if r in sub["country_region"].unique()]
    )
    sig = sub.pivot(index="country_region", columns="dim_label",
                    values="sig_fdr").reindex(pivot.index)
    annot = pivot.round(2).astype(str)
    annot[sig == True] = annot[sig == True] + " *"   # noqa: E712

    fig, ax = plt.subplots(figsize=(9, max(4, len(pivot) * 0.65)))
    sns.heatmap(
        pivot, annot=annot, fmt="s",
        cmap="RdBu_r", center=0, vmin=-1.0, vmax=1.0,
        linewidths=0.5, ax=ax,
        cbar_kws={"label": "Cohen's d  (positive = daughters higher)"},
    )
    ax.set_title(
        "PMI-Weighted Gender Bias by World Region and Dimension\n"
        "(* = FDR p < 0.05;  positive = daughter stories more stereotyped)",
        fontsize=11, fontweight="bold",
    )
    ax.set_xlabel("Bias dimension", fontsize=10)
    ax.set_ylabel("World region", fontsize=10)
    plt.xticks(rotation=25, ha="right", fontsize=9)
    plt.yticks(rotation=0,  fontsize=9)
    _save(fig, "fig11_pmi_region_heatmap", fig_dir)
    print("  Saved fig11_pmi_region_heatmap")


# Figure 12: Per-model PMI distributions 

def fig_pmi_distributions_by_model(df: pd.DataFrame, fig_dir: Path) -> None:
    """
    Split-violin PMI score distributions by model: one subplot per bias
    dimension (stacked vertically).  Each model position has a single violin
    body split down the middle: left half = daughter stories, right half = son.
    Inner quartile lines show medians and IQR.
    """
    cols_present = [d[0] for d in _DIM_INFO if d[0] in df.columns]
    if not cols_present or "model_key" not in df.columns:
        return

    models = sorted(df["model_key"].unique())
    n_dim  = len(cols_present)
    fig, axes = plt.subplots(n_dim, 1,
                             figsize=(max(10, len(models) * 1.1), n_dim * 4),
                             sharex=True)
    if n_dim == 1:
        axes = [axes]

    for ax, (col, _, _, label) in zip(axes, _DIM_INFO):
        if col not in df.columns:
            continue
        sns.violinplot(
            data=df, x="model_key", y=col,
            hue="child_label", hue_order=["daughter", "son"],
            order=models, palette=PALETTE_GENDER,
            inner="quartile", density_norm="width",
            linewidth=0.7, split=True, ax=ax, legend=False,
        )
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.set_ylabel(f"PMI score\n{label.split(chr(10))[0]}", fontsize=9)
        ax.set_xlabel("")
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.tick_params(axis="x", rotation=30, labelsize=8)

    axes[-1].set_xlabel("Model", fontsize=10)

    from matplotlib.patches import Patch
    fig.legend(
        handles=[Patch(color=PALETTE_GENDER["daughter"], label="Daughter"),
                 Patch(color=PALETTE_GENDER["son"],      label="Son")],
        loc="upper right", fontsize=10, framealpha=0.9,
    )
    fig.suptitle(
        "PMI Bias Score Distributions by Model\n"
        "(split violin — left half = daughter, right half = son; "
        "lines = median & IQR)",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    _save(fig, "fig12_pmi_by_model", fig_dir)
    print("  Saved fig12_pmi_by_model")


# Figure 13: PMI bias gap vs model size 

def fig_pmi_model_size(df: pd.DataFrame, fig_dir: Path) -> None:
    """
    Line plot: daughter − son PMI gap as a function of model size group,
    one line per bias dimension, with 95 % bootstrap CIs.
    """
    if "model_size_group" not in df.columns:
        return

    size_order   = ["small", "medium", "large", "xlarge"]
    dim_colors   = ["#C0415A", "#3A6BAD", "#3E9A55"]
    dim_markers  = ["o", "s", "^"]

    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(8, 5))

    for (col, _, _, label), color, marker in zip(_DIM_INFO, dim_colors, dim_markers):
        if col not in df.columns:
            continue
        gaps, lo_errs, hi_errs, x_ticks = [], [], [], []
        for grp in size_order:
            sub = df[df["model_size_group"] == grp]
            d   = sub[sub["child_label"] == "daughter"][col].dropna().values
            s   = sub[sub["child_label"] == "son"][col].dropna().values
            if len(d) < 2 or len(s) < 2:
                continue
            gap   = d.mean() - s.mean()
            boots = [rng.choice(d, len(d)).mean() - rng.choice(s, len(s)).mean()
                     for _ in range(N_BOOT)]
            lo, hi = np.percentile(boots, [2.5, 97.5])
            gaps.append(gap); lo_errs.append(gap - lo)
            hi_errs.append(hi - gap); x_ticks.append(grp)

        if not gaps:
            continue
        x = np.arange(len(x_ticks))
        ax.errorbar(x, gaps, yerr=[lo_errs, hi_errs],
                    fmt=marker + "-", color=color, capsize=5,
                    linewidth=2, markersize=8,
                    label=label.split("\n")[0])

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    # re-draw x-ticks at only the groups that had data
    present = [g for g in size_order
               if any(df[df["model_size_group"] == g]["child_label"].isin(
                   ["daughter", "son"]))]
    ax.set_xticks(np.arange(len(present)))
    ax.set_xticklabels(present, fontsize=11)
    ax.set_xlabel("Model size group", fontsize=11)
    ax.set_ylabel("Daughter − Son mean PMI score\n(95 % bootstrap CI)", fontsize=10)
    ax.set_title(
        "Does Model Size Predict Gender Stereotyping?\n"
        "(positive = daughters receive more stereotyped language)",
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=10, framealpha=0.85)
    fig.tight_layout()
    _save(fig, "fig13_pmi_model_size", fig_dir)
    print("  Saved fig13_pmi_model_size")


# Figure 14: PMI score heatmap: female (green) and male (orange) per model × dimension 

def fig_pmi_model_dimension_heatmap(df: pd.DataFrame, fig_dir: Path) -> None:
    """
    Single combined table-style heatmap: models (rows) × [F, M] sub-columns
    for each of 3 bias dimensions (Role, Domain, Trait).

    F columns (green gradient): mean pmi_X_fem_raw per model.
    M columns (orange gradient): mean pmi_X_masc_raw per model (original, negative values).
    Color intensity = strength of signal; more-negative masc → darker orange.
    No colorbar. Dimension names and F/M labels appear at the top.
    """
    from matplotlib.patches import Rectangle

    _DIMS = [
        ("pmi_role_fem_raw",   "pmi_role_masc_raw",   "Role"),
        ("pmi_domain_fem_raw", "pmi_domain_masc_raw", "Domain"),
        ("pmi_trait_fem_raw",  "pmi_trait_masc_raw",  "Trait"),
    ]

    for fem_col, masc_col, _ in _DIMS:
        if fem_col not in df.columns or masc_col not in df.columns:
            print(f"  fig14 skipped — missing {fem_col!r}")
            return

    models       = sorted(df["model_key"].unique())
    model_labels = [m.replace("ollama-", "") for m in models]
    n_models     = len(models)
    n_dims       = len(_DIMS)

    # Aggregate mean per model
    fem_mat  = np.zeros((n_models, n_dims))
    masc_mat = np.zeros((n_models, n_dims))
    for j, (fem_col, masc_col, _) in enumerate(_DIMS):
        for i, mdl in enumerate(models):
            sub = df[df["model_key"] == mdl]
            fem_mat[i, j]  = sub[fem_col].mean()
            masc_mat[i, j] = sub[masc_col].mean()   # keep original negative values

    # Normalise for color intensity (0 = lightest, 1 = darkest within each type)
    f_min, f_max = fem_mat.min(),  fem_mat.max()
    m_min, m_max = masc_mat.min(), masc_mat.max()

    def _nf(v):   # fem: higher positive → darker green
        return (v - f_min) / (f_max - f_min + EPS)

    def _nm(v):   # masc: more negative → darker orange
        return 1.0 - (v - m_min) / (m_max - m_min + EPS)

    green_cmap  = plt.cm.Greens
    orange_cmap = plt.cm.Oranges
    INTENSITY_LO, INTENSITY_HI = 0.20, 0.82   # avoid too-light / too-dark extremes

    # Layout constants (data-units = inches because xlim/ylim match figsize)
    cw      = 1.55   # cell width
    ch      = 0.70   # cell height
    lm      = 3.00   # left margin for model labels
    gap     = 0.18   # horizontal gap between dimension groups
    pad_b   = 0.18   # bottom padding
    fm_h    = 0.36   # F/M header bar height
    dim_h   = 0.44   # dimension title row height
    pad_t   = 0.12   # top padding

    content_h = n_models * ch
    fig_h = pad_b + content_h + fm_h + dim_h + pad_t
    fig_w = lm + n_dims * (2 * cw + gap) - gap + 0.22

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.set_aspect("auto")
    ax.axis("off")

    def _col_x(dim_idx: int, is_masc: bool) -> float:
        return lm + dim_idx * (2 * cw + gap) + (cw if is_masc else 0)

    
    for i, mlabel in enumerate(model_labels):
        y = pad_b + (n_models - 1 - i) * ch   # top model first

        ax.text(lm - 0.10, y + ch / 2, mlabel,
                ha="right", va="center", fontsize=20, fontweight="bold", color="black")

        for j in range(n_dims):
            xf = _col_x(j, False)
            xm = _col_x(j, True)

            # Feminine cell
            fi   = INTENSITY_LO + (INTENSITY_HI - INTENSITY_LO) * _nf(fem_mat[i, j])
            fc   = green_cmap(fi)
            ax.add_patch(Rectangle((xf, y), cw, ch,
                                   facecolor=fc, edgecolor="white", linewidth=0.5))
            txt_c = "white" if fi > 0.58 else "black"
            ax.text(xf + cw / 2, y + ch / 2 - 0.03, f"{fem_mat[i, j]:.2f}".replace("-", "- "),
                    ha="center", va="center", fontsize=20, fontweight="bold", color=txt_c)

            # Masculine cell
            mi   = INTENSITY_LO + (INTENSITY_HI - INTENSITY_LO) * _nm(masc_mat[i, j])
            mc   = orange_cmap(mi)
            ax.add_patch(Rectangle((xm, y), cw, ch,
                                   facecolor=mc, edgecolor="white", linewidth=0.5))
            txt_c = "white" if mi > 0.58 else "black"
            ax.text(xm + cw / 2, y + ch / 2 - 0.03, f"{masc_mat[i, j]:.2f}".replace("-", "- "),
                    ha="center", va="center", fontsize=20, fontweight="bold", color=txt_c)

    # F / M labels
    y_fm = pad_b + content_h + 0.04
    for j in range(n_dims):
        xf = _col_x(j, False)
        xm = _col_x(j, True)
        ax.text(xf + cw / 2, y_fm + fm_h / 2, "F",
                ha="center", va="center", fontsize=20, fontweight="bold", color="black")
        ax.text(xm + cw / 2, y_fm + fm_h / 2, "M",
                ha="center", va="center", fontsize=20, fontweight="bold", color="black")

    # Dimension title row
    y_dim = y_fm + fm_h + 0.04
    for j, (_, _, dlabel) in enumerate(_DIMS):
        xf     = _col_x(j, False)
        xm     = _col_x(j, True)
        x_mid  = (xf + xm + cw) / 2
        x_end  = xm + cw
        ax.text(x_mid, y_dim + dim_h / 2, dlabel,
                ha="center", va="center", fontsize=20, fontweight="bold", color="black")
        ax.plot([xf, x_end], [y_dim, y_dim], color="black", linewidth=0.9)

    fig.suptitle(
        "PMI Scores by Model and Bias Dimension  "
        "(F = feminine lexicon · M = masculine lexicon · darker = stronger signal)",
        fontsize=20, fontweight="bold",
    )
    fig.subplots_adjust(top=0.94, bottom=0.02, left=0.02, right=0.98)
    _save(fig, "fig14_pmi_model_dimension_heatmap", fig_dir)
    print("  Saved fig14_pmi_model_dimension_heatmap")


# Figure 15: Semantic score heatmap — feminine (green) and masculine (orange) per model × dimension ──

def fig_sem_model_dimension_heatmap(df: pd.DataFrame, fig_dir: Path) -> None:
    """
    Same table-style layout as fig14 but using semantic cosine-similarity scores.

    F columns (green):  mean sem_X_feminine per model  (sem_feminine_traits,
                        sem_communal_role, sem_family_domain).
    M columns (orange): mean sem_X_masculine per model (sem_masculine_traits,
                        sem_agentic_role,  sem_career_domain).
    Positive sem_trait/role/domain_bias → daughter-stereotyped (matches PMI convention
    after the sidx argument swap).  No colorbar.
    """
    from matplotlib.patches import Rectangle

    _DIMS = [
        ("sem_feminine_traits", "sem_masculine_traits", "Trait"),
        ("sem_communal_role",   "sem_agentic_role",     "Role"),
        ("sem_family_domain",   "sem_career_domain",    "Domain"),
    ]

    for fem_col, masc_col, _ in _DIMS:
        if fem_col not in df.columns or masc_col not in df.columns:
            print(f"  fig15 skipped — missing {fem_col!r}")
            return

    models       = sorted(df["model_key"].unique())
    model_labels = [m.replace("ollama-", "") for m in models]
    n_models     = len(models)
    n_dims       = len(_DIMS)

    fem_mat  = np.zeros((n_models, n_dims))
    masc_mat = np.zeros((n_models, n_dims))
    for j, (fem_col, masc_col, _) in enumerate(_DIMS):
        for i, mdl in enumerate(models):
            sub = df[df["model_key"] == mdl]
            fem_mat[i, j]  = sub[fem_col].mean()
            masc_mat[i, j] = sub[masc_col].mean()

    f_min, f_max = fem_mat.min(),  fem_mat.max()
    m_min, m_max = masc_mat.min(), masc_mat.max()

    def _nf(v):
        return (v - f_min) / (f_max - f_min + EPS)

    def _nm(v):
        return (v - m_min) / (m_max - m_min + EPS)

    green_cmap  = plt.cm.Greens
    orange_cmap = plt.cm.Oranges
    INTENSITY_LO, INTENSITY_HI = 0.20, 0.82

    cw      = 1.55
    ch      = 0.70
    lm      = 3.00
    gap     = 0.18
    pad_b   = 0.18
    fm_h    = 0.36
    dim_h   = 0.44
    pad_t   = 0.12

    content_h = n_models * ch
    fig_h = pad_b + content_h + fm_h + dim_h + pad_t
    fig_w = lm + n_dims * (2 * cw + gap) - gap + 0.22

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.set_aspect("auto")
    ax.axis("off")

    def _col_x(dim_idx: int, is_masc: bool) -> float:
        return lm + dim_idx * (2 * cw + gap) + (cw if is_masc else 0)

    for i, mlabel in enumerate(model_labels):
        y = pad_b + (n_models - 1 - i) * ch

        ax.text(lm - 0.10, y + ch / 2, mlabel,
                ha="right", va="center", fontsize=20, fontweight="bold", color="black")

        for j in range(n_dims):
            xf = _col_x(j, False)
            xm = _col_x(j, True)

            fi = INTENSITY_LO + (INTENSITY_HI - INTENSITY_LO) * _nf(fem_mat[i, j])
            fc = green_cmap(fi)
            ax.add_patch(Rectangle((xf, y), cw, ch,
                                   facecolor=fc, edgecolor="white", linewidth=0.5))
            ax.text(xf + cw / 2, y + ch / 2 - 0.03,
                    f"{fem_mat[i, j]:.3f}".replace("-", "- "),
                    ha="center", va="center", fontsize=20, fontweight="bold",
                    color="white" if fi > 0.58 else "black")

            mi = INTENSITY_LO + (INTENSITY_HI - INTENSITY_LO) * _nm(masc_mat[i, j])
            mc = orange_cmap(mi)
            ax.add_patch(Rectangle((xm, y), cw, ch,
                                   facecolor=mc, edgecolor="white", linewidth=0.5))
            ax.text(xm + cw / 2, y + ch / 2 - 0.03,
                    f"{masc_mat[i, j]:.3f}".replace("-", "- "),
                    ha="center", va="center", fontsize=20, fontweight="bold",
                    color="white" if mi > 0.58 else "black")

    y_fm = pad_b + content_h + 0.04
    for j in range(n_dims):
        xf = _col_x(j, False)
        xm = _col_x(j, True)
        ax.text(xf + cw / 2, y_fm + fm_h / 2, "F",
                ha="center", va="center", fontsize=20, fontweight="bold", color="black")
        ax.text(xm + cw / 2, y_fm + fm_h / 2, "M",
                ha="center", va="center", fontsize=20, fontweight="bold", color="black")

    y_dim = y_fm + fm_h + 0.04
    for j, (_, _, dlabel) in enumerate(_DIMS):
        xf    = _col_x(j, False)
        xm    = _col_x(j, True)
        x_mid = (xf + xm + cw) / 2
        ax.text(x_mid, y_dim + dim_h / 2, dlabel,
                ha="center", va="center", fontsize=20, fontweight="bold")
        ax.plot([xf, xm + cw], [y_dim, y_dim], color="black", linewidth=0.9)

    fig.suptitle(
        "Semantic Similarity Scores by Model and Bias Dimension  "
        "(F = feminine prototype · M = masculine prototype · darker = stronger signal)",
        fontsize=20, fontweight="bold",
    )
    fig.subplots_adjust(top=0.94, bottom=0.02, left=0.02, right=0.98)
    _save(fig, "fig15_sem_model_dimension_heatmap", fig_dir)
    print("  Saved fig15_sem_model_dimension_heatmap")


# ── Helper: spider / radar chart ─────────────────────────────────────────────

def _draw_radar_chart(
    values_per_model: dict[str, list[float]],
    categories: list[str],
    colors: dict,
    title: str,
    fig_dir: Path,
    fname: str,
) -> None:
    """
    Spider / radar chart with one polygon per model.
    Each axis is normalised independently to [0, 1]; actual numeric values
    are annotated at 20 %, 40 %, 60 %, 80 %, 100 % of each spoke's range.
    """
    N      = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)

    all_vals = np.array(list(values_per_model.values()), dtype=float)
    a_min    = all_vals.min(axis=0)
    a_max    = all_vals.max(axis=0)
    a_rng    = np.where(a_max > a_min, a_max - a_min, 1.0)

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor("#f8f8f8")
    ax.set_facecolor("#f2f2f2")

    # Concentric grid rings
    ring_pts = np.linspace(0, 2 * np.pi, 300)
    for ring in np.linspace(0.2, 1.0, 5):
        ax.plot(ring_pts, [ring] * 300, color="gray", alpha=0.25, linewidth=0.7, zorder=1)

    # Spokes
    for angle in angles:
        ax.plot([angle, angle], [0, 1.05], color="gray", alpha=0.4, linewidth=0.8, zorder=1)

    # Numeric value labels at 20 %, 40 %, 60 %, 80 %, 100 % of each spoke
    n_ticks = 5
    for j, angle in enumerate(angles):
        for k in range(1, n_ticks + 1):
            r_frac = k / n_ticks
            actual = a_min[j] + r_frac * a_rng[j]
            ax.text(
                angle, r_frac, f"{actual:.2f}",
                ha="center", va="center", fontsize=7.5, color="dimgray",
                bbox=dict(facecolor="white", alpha=0.55, edgecolor="none", pad=0.4),
                zorder=5,
            )

    # Model polygons
    for model_key, vals in values_per_model.items():
        label = model_key.replace("ollama-", "")
        color = colors.get(model_key, "gray")
        norm  = [(v - a_min[j]) / a_rng[j] for j, v in enumerate(vals)]
        ang_c = list(angles) + [angles[0]]
        nrm_c = norm + [norm[0]]
        ax.plot(ang_c, nrm_c, "o-", linewidth=2.2, markersize=7,
                color=color, label=label, zorder=4)
        ax.fill(ang_c, nrm_c, alpha=0.12, color=color, zorder=3)

    ax.set_xticks(angles)
    ax.set_xticklabels(categories, size=14, fontweight="bold")
    ax.set_yticks([])
    ax.set_ylim(0, 1.18)
    ax.set_title(title, size=15, fontweight="bold", pad=30)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(-0.45, 1.25),
        fontsize=9.5,
        framealpha=0.9,
        title="Model",
        title_fontsize=10,
    )
    fig.tight_layout()
    _save(fig, fname, fig_dir)


# Figure 14 (radar): PMI bias spider charts ────────────────────────────────────

def fig_pmi_radar(df: pd.DataFrame, fig_dir: Path) -> None:
    """
    Two spider charts derived from Fig 14 data.

    Daughter chart — axes: PMI Role / Domain / Trait (feminine raw scores).
    Son chart      — axes: PMI Role / Domain / Trait (masculine raw scores, abs).

    One polygon trace per model; axes normalised independently.
    """
    fem_cols  = [("pmi_role_fem_raw",   "PMI\nRole"),
                 ("pmi_domain_fem_raw", "PMI\nDomain"),
                 ("pmi_trait_fem_raw",  "PMI\nTrait")]
    masc_cols = [("pmi_role_masc_raw",   "PMI\nRole"),
                 ("pmi_domain_masc_raw", "PMI\nDomain"),
                 ("pmi_trait_masc_raw",  "PMI\nTrait")]

    for col, _ in fem_cols + masc_cols:
        if col not in df.columns:
            print(f"  fig_pmi_radar skipped — missing {col!r}")
            return

    models     = sorted(df["model_key"].unique())
    cmap_fn    = plt.cm.get_cmap("tab20", len(models))
    colors     = {m: cmap_fn(i) for i, m in enumerate(models)}
    categories = [lbl for _, lbl in fem_cols]

    fem_vals  = {m: [df[df["model_key"] == m][c].mean() for c, _ in fem_cols]
                 for m in models}
    masc_vals = {m: [abs(df[df["model_key"] == m][c].mean()) for c, _ in masc_cols]
                 for m in models}

    _draw_radar_chart(
        fem_vals, categories, colors,
        "PMI Bias Radar — Feminine Lexicon  (Daughter-Stereotyped Language)",
        fig_dir, "fig14_pmi_radar_daughter",
    )
    _draw_radar_chart(
        masc_vals, categories, colors,
        "PMI Bias Radar — Masculine Lexicon  (Son-Stereotyped Language)",
        fig_dir, "fig14_pmi_radar_son",
    )
    print("  Saved fig14_pmi_radar_daughter / fig14_pmi_radar_son")


# Figure 15 (radar): Semantic bias spider charts ───────────────────────────────

def fig_sem_radar(df: pd.DataFrame, fig_dir: Path) -> None:
    """
    Two spider charts derived from Fig 15 data.

    Daughter chart — axes: Sem Trait / Role / Domain (feminine prototype scores).
    Son chart      — axes: Sem Trait / Role / Domain (masculine prototype scores).

    One polygon trace per model; axes normalised independently.
    """
    fem_cols  = [("sem_feminine_traits", "Sem\nTrait"),
                 ("sem_communal_role",   "Sem\nRole"),
                 ("sem_family_domain",   "Sem\nDomain")]
    masc_cols = [("sem_masculine_traits", "Sem\nTrait"),
                 ("sem_agentic_role",     "Sem\nRole"),
                 ("sem_career_domain",    "Sem\nDomain")]

    for col, _ in fem_cols + masc_cols:
        if col not in df.columns:
            print(f"  fig_sem_radar skipped — missing {col!r}")
            return

    models     = sorted(df["model_key"].unique())
    cmap_fn    = plt.cm.get_cmap("tab20", len(models))
    colors     = {m: cmap_fn(i) for i, m in enumerate(models)}
    categories = [lbl for _, lbl in fem_cols]

    fem_vals  = {m: [df[df["model_key"] == m][c].mean() for c, _ in fem_cols]
                 for m in models}
    masc_vals = {m: [df[df["model_key"] == m][c].mean() for c, _ in masc_cols]
                 for m in models}

    _draw_radar_chart(
        fem_vals, categories, colors,
        "Semantic Bias Radar — Feminine Prototypes  (Daughter-Stereotyped Language)",
        fig_dir, "fig15_sem_radar_daughter",
    )
    _draw_radar_chart(
        masc_vals, categories, colors,
        "Semantic Bias Radar — Masculine Prototypes  (Son-Stereotyped Language)",
        fig_dir, "fig15_sem_radar_son",
    )
    print("  Saved fig15_sem_radar_daughter / fig15_sem_radar_son")


# Main

def main() -> None:
    args    = parse_args()
    out_dir = Path(args.output_dir)
    res_dir = out_dir / "results"
    fig_dir = out_dir / "figures"
    res_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load and enrich
    raw = load_data(args.input)
    df  = enrich(raw)
    print(f"\nAnalysing {len(df):,} stories "
          f"({df['child_label'].value_counts().to_dict()})")

    # 2. Lexicon scoring (raw counts + legacy composite indices)
    print("\n Lexicon analysis")
    scored = score_lexicon(df)
    scored.to_csv(res_dir / "story_level_lexicon.csv", index=False)
    print(f"  Story-level scores saved ({len(scored):,} rows)")

    # 2b. PMI-weighted scoring
    pmi_scored = None
    if not args.skip_pmi:
        print("\n PMI-weighted analysis")
        pmi_weights = compute_pmi_weights(scored)
        print(f"  PMI weights computed for {len(pmi_weights):,} tokens "
              f"(min_count=5)")
        pmi_scored = score_pmi(scored, pmi_weights)
        pmi_scored.to_csv(res_dir / "story_level_pmi.csv", index=False)
        print(f"  PMI scores saved  ({len(pmi_scored):,} rows, "
              f"dims: {PMI_METRICS})")
    else:
        print("\n PMI analysis skipped (--skip-pmi) ")

    # 3. Statistical tests — child gender effect per model
    print("\n Statistical tests (legacy lexicon)")
    tests_model   = pairwise_tests(scored, "model_key",         CORE_METRICS, args.n_bootstrap)
    tests_region  = pairwise_tests(scored, "country_region",    CORE_METRICS, args.n_bootstrap)
    tests_person  = pairwise_tests(scored, "person_gender_role",CORE_METRICS, args.n_bootstrap)
    tests_overall = pairwise_tests(
        scored.assign(_all="all"), "_all", CORE_METRICS, args.n_bootstrap
    )

    tests_model.to_csv(res_dir / "tests_by_model.csv",           index=False)
    tests_region.to_csv(res_dir / "tests_by_region.csv",         index=False)
    tests_person.to_csv(res_dir / "tests_by_person_role.csv",    index=False)
    tests_overall.to_csv(res_dir / "tests_overall.csv",          index=False)
    print(f"  Significant (FDR) — model: {tests_model['sig_fdr'].sum()}, "
          f"region: {tests_region['sig_fdr'].sum()}, "
          f"person: {tests_person['sig_fdr'].sum()}")

    # 3b. Statistical tests — PMI metrics
    pmi_tests_model = pd.DataFrame()
    if pmi_scored is not None:
        print("\n Statistical tests (PMI-weighted)")
        pmi_tests_model   = pairwise_tests(pmi_scored, "model_key",         PMI_METRICS, args.n_bootstrap)
        pmi_tests_region  = pairwise_tests(pmi_scored, "country_region",    PMI_METRICS, args.n_bootstrap)
        pmi_tests_person  = pairwise_tests(pmi_scored, "person_gender_role",PMI_METRICS, args.n_bootstrap)
        pmi_tests_overall = pairwise_tests(
            pmi_scored.assign(_all="all"), "_all", PMI_METRICS, args.n_bootstrap
        )
        pmi_tests_model.to_csv(res_dir  / "pmi_tests_by_model.csv",        index=False)
        pmi_tests_region.to_csv(res_dir / "pmi_tests_by_region.csv",       index=False)
        pmi_tests_person.to_csv(res_dir / "pmi_tests_by_person_role.csv",  index=False)
        pmi_tests_overall.to_csv(res_dir / "pmi_tests_overall.csv",        index=False)
        print(f"  Significant (FDR) — model: {pmi_tests_model['sig_fdr'].sum()}, "
              f"region: {pmi_tests_region['sig_fdr'].sum()}, "
              f"person: {pmi_tests_person['sig_fdr'].sum()}")

    # 4. Regression decomposition (legacy + PMI)
    print("\n Regression analysis ")
    reg = run_regressions(scored)
    if not reg.empty:
        reg.to_csv(res_dir / "regression_is_daughter.csv", index=False)
        print(reg[["metric", "coef_is_daughter", "p_is_daughter", "adj_r_squared"]]
              .round(4).to_string(index=False))
    if pmi_scored is not None:
        pmi_reg = run_regressions(pmi_scored)
        if not pmi_reg.empty:
            pmi_reg = pmi_reg[pmi_reg["metric"].isin(PMI_METRICS)]
            pmi_reg.to_csv(res_dir / "pmi_regression_is_daughter.csv", index=False)
            if not pmi_reg.empty:
                print(pmi_reg[["metric", "coef_is_daughter", "p_is_daughter",
                               "adj_r_squared"]].round(4).to_string(index=False))

    # 5. Semantic analysis
    sem_scored = None
    tests_sem  = pd.DataFrame()
    if not args.skip_semantic:
        print("\n Semantic analysis ")
        sem_scored = score_semantic(scored, args.embed_model)
        sem_scored.to_csv(res_dir / "story_level_semantic.csv", index=False)
        tests_sem_model  = pairwise_tests(sem_scored, "model_key",          SEM_METRICS, args.n_bootstrap)
        tests_sem_region = pairwise_tests(sem_scored, "country_region",     SEM_METRICS, args.n_bootstrap)
        tests_sem_person = pairwise_tests(sem_scored, "person_gender_role", SEM_METRICS, args.n_bootstrap)
        tests_sem_overall = pairwise_tests(
            sem_scored.assign(_all="all"), "_all", SEM_METRICS, args.n_bootstrap
        )
        tests_sem_model.to_csv(res_dir  / "sem_tests_by_model.csv",       index=False)
        tests_sem_region.to_csv(res_dir / "sem_tests_by_region.csv",      index=False)
        tests_sem_person.to_csv(res_dir / "sem_tests_by_person_role.csv", index=False)
        tests_sem_overall.to_csv(res_dir / "sem_tests_overall.csv",       index=False)
        tests_sem = tests_sem_model
        print(f"  Significant (FDR) — model: {tests_sem_model['sig_fdr'].sum()}, "
              f"region: {tests_sem_region['sig_fdr'].sum()}, "
              f"person: {tests_sem_person['sig_fdr'].sum()}")
    else:
        print("\n Semantic analysis skipped (--skip-semantic) ")

    # 6. Figures
    print("\n Generating figures... ")

    if not args.skip_legacy_figs:
        fig_main_result(scored, fig_dir)
        fig_effect_heatmap(tests_model,  "model_key",
                           "fig2_effect_heatmap_model",
                           "Cohen's d (daughter − son) by Model × Metric  (* = FDR p<0.05)",
                           fig_dir)
        fig_effect_heatmap(tests_region, "country_region",
                           "fig2b_effect_heatmap_region",
                           "Cohen's d by World Region × Metric  (* = FDR p<0.05)",
                           fig_dir)
        fig_regional(scored, fig_dir)
        fig_storyteller(scored, fig_dir)
        fig_model_size(scored, fig_dir)
        if not reg.empty:
            fig_regression_forest(reg, fig_dir)
        if sem_scored is not None:
            fig_semantic_overview(sem_scored, fig_dir)

    if sem_scored is not None and pmi_scored is not None:
        fig_sem_pmi_comparison(sem_scored, pmi_scored, fig_dir)

    if pmi_scored is not None:
        # Fig 8: full score distributions (KDE) 
        fig_pmi_distributions(pmi_scored, fig_dir)
        # Fig 9: top discriminative words 
        fig_pmi_top_words(pmi_weights, fig_dir)
        # Fig 10: per-model bias gaps 
        if not pmi_tests_model.empty:
            fig_pmi_model_gaps(pmi_tests_model, fig_dir)
        # Fig 11: region × dimension heatmap 
        if not pmi_tests_region.empty:
            fig_pmi_region_heatmap(pmi_tests_region, fig_dir)
        # Fig 12: per-model split violins 
        fig_pmi_distributions_by_model(pmi_scored, fig_dir)
        # Fig 13: bias gap vs model size 
        fig_pmi_model_size(pmi_scored, fig_dir)
        # Fig 14: female (green) / male (orange) PMI heatmap: models × dimensions
        fig_pmi_model_dimension_heatmap(pmi_scored, fig_dir)

    if sem_scored is not None:
        # Fig 15: feminine (green) / masculine (orange) semantic heatmap: models × dimensions
        fig_sem_model_dimension_heatmap(sem_scored, fig_dir)

    # 7. Summary
    print("GENDER BIAS ANALYSIS: COMPLETE")
    print(f"  Stories analysed : {len(scored):,}")
    print(f"  Models           : {scored['model_key'].nunique()}")
    if "country" in scored.columns:
        print(f"  Countries        : {scored['country'].nunique()}")
    if "person" in scored.columns:
        print(f"  Storytellers     : {scored['person'].nunique()}")

    if pmi_scored is not None:
        print("\n  PMI bias scores (mean ± std):")
        for col, _, _, label in _DIM_INFO:
            d_m = pmi_scored[pmi_scored["child_label"] == "daughter"][col]
            s_m = pmi_scored[pmi_scored["child_label"] == "son"][col]
            print(f"  {label.split(chr(10))[0]:30s}  "
                  f"daughter={d_m.mean():+.3f}±{d_m.std():.3f}  "
                  f"son={s_m.mean():+.3f}±{s_m.std():.3f}  "
                  f"gap={d_m.mean()-s_m.mean():+.3f}")
    print(f"\n  Results  : {res_dir}")
    print(f"  Figures  : {fig_dir}")


if __name__ == "__main__":
    main()


