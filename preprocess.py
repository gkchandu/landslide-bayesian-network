"""
Step 2: Preprocessing for Landslide Risk Assessment (India) - Bayesian Network.

Run from the project root:  python preprocess.py

Outputs:
  data/landslide_clean.csv     cleaned, discretized data (string categories, used by the BN)
  data/landslide_encoded.csv   same data, integer-encoded (used by sklearn baselines)
  data/encoding_map.json       category -> integer mapping
  data/feature_scores.csv      mutual information / chi-square of each feature vs severity
"""
import json
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.metrics import mutual_info_score

RAW = "data/raw/Global_Landslide_Catalog_Export.csv"
OUT_CLEAN = "data/landslide_clean.csv"
OUT_ENC = "data/landslide_encoded.csv"
OUT_MAP = "data/encoding_map.json"
OUT_SCORES = "data/feature_scores.csv"

# Severity thresholds (fatalities per event) - adjust after seeing the class balance
MEDIUM_MIN, HIGH_MIN = 1, 5
RARE_FRACTION = 0.02          # categories below 2% of rows are merged into "other"
FEATURES = ["season", "zone", "trigger", "category", "size", "setting", "accuracy"]
TARGET = "severity"


# ---------- helpers ----------
def group_rare(s, min_frac=RARE_FRACTION, keep=("unknown",), label="other"):
    freq = s.value_counts(normalize=True)
    rare = set(freq[freq < min_frac].index) - set(keep)
    return s.where(~s.isin(rare), label)


def month_to_season(m):
    # Indian meteorological seasons
    if m in (12, 1, 2):
        return "winter"
    if m in (3, 4, 5):
        return "pre_monsoon"
    if m in (6, 7, 8, 9):
        return "monsoon"
    return "post_monsoon"          # Oct, Nov


def to_zone(lat, lon):
    """Rough physiographic zone from coordinates (approximation, state it in the report)."""
    if lon >= 88 and lat >= 24:
        return "northeast_india"
    if lat >= 28:
        return "western_himalaya"
    if lon <= 77.5 and lat <= 21:
        return "western_ghats"
    return "peninsular_central_other"


def to_severity(f):
    if f >= HIGH_MIN:
        return "high"
    if f >= MEDIUM_MIN:
        return "medium"
    return "low"


TRIGGER_MAP = {
    "downpour": "downpour", "rain": "rain", "continuous_rain": "continuous_rain",
    "monsoon": "monsoon", "tropical_cyclone": "tropical_cyclone", "earthquake": "earthquake",
    "mining": "human_activity", "construction": "human_activity",
    "flooding": "other", "snowfall_snowmelt": "other", "freeze_thaw": "other",
    "no_apparent_trigger": "other", "other": "other", "unknown": "unknown",
}
SIZE_MAP = {"small": "small", "medium": "medium", "large": "large_plus",
            "very_large": "large_plus", "catastrophic": "large_plus", "unknown": "unknown"}
ACC_MAP = {"exact": "precise", "1km": "precise", "5km": "precise",
           "10km": "approx", "25km": "approx",
           "50km": "coarse", "100km": "coarse", "250km": "coarse", "unknown": "unknown"}


def main():
    df = pd.read_csv(RAW)
    print("Raw shape:", df.shape)

    # ---- 1. filter to India ----
    df = df[df["country_name"].astype(str).str.strip().str.lower() == "india"].copy()
    print("India records:", len(df))

    # ---- 2. basic cleaning ----
    df["event_date"] = pd.to_datetime(df["event_date"], format="%m/%d/%Y %I:%M:%S %p", errors="coerce")
    df["event_date"] = df["event_date"].fillna(pd.to_datetime(df["event_date"], errors="coerce"))
    before = len(df)
    df = df.dropna(subset=["event_date", "latitude", "longitude"])
    df = df[df["latitude"].between(6, 37.5) & df["longitude"].between(68, 98)]   # inside India's extent
    print(f"Dropped {before - len(df)} rows with bad date/coordinates")

    df = df.drop_duplicates(subset=["event_id"])
    df = df.drop_duplicates(subset=["event_date", "latitude", "longitude", "landslide_trigger"])

    # ---- 3. missing target -> drop (severity cannot be inferred) ----
    before = len(df)
    df = df.dropna(subset=["fatality_count"])
    print(f"Dropped {before - len(df)} rows with missing fatality_count")

    # ---- 4. feature engineering / discretization ----
    out = pd.DataFrame(index=df.index)
    out["season"] = df["event_date"].dt.month.map(month_to_season)
    out["zone"] = [to_zone(a, b) for a, b in zip(df["latitude"], df["longitude"])]

    trig = df["landslide_trigger"].fillna("unknown").str.strip().str.lower().map(TRIGGER_MAP).fillna("other")
    out["trigger"] = group_rare(trig)

    cat = df["landslide_category"].fillna("unknown").str.strip().str.lower()
    out["category"] = group_rare(cat)

    out["size"] = df["landslide_size"].fillna("unknown").str.strip().str.lower().map(SIZE_MAP).fillna("unknown")

    setting = df["landslide_setting"].fillna("unknown").str.strip().str.lower()
    out["setting"] = group_rare(setting)

    out["accuracy"] = df["location_accuracy"].fillna("unknown").astype(str).str.strip().str.lower().map(ACC_MAP).fillna("unknown")

    out[TARGET] = df["fatality_count"].astype(float).map(to_severity)

    # ---- 5. summary of each variable ----
    print("\nFinal shape:", out.shape)
    for c in FEATURES + [TARGET]:
        print(f"\n--- {c} ---")
        print(out[c].value_counts())

    # ---- 6. feature selection: mutual information + chi-square vs severity ----
    rows = []
    for c in FEATURES:
        mi = mutual_info_score(out[c], out[TARGET])
        try:
            chi2, p, dof, _ = chi2_contingency(pd.crosstab(out[c], out[TARGET]))
        except ValueError:
            chi2, p, dof = np.nan, np.nan, np.nan
        rows.append({"feature": c, "mutual_info": round(mi, 4),
                     "chi2": round(chi2, 2), "p_value": round(p, 4), "dof": dof})
    scores = pd.DataFrame(rows).sort_values("mutual_info", ascending=False)
    print("\nFeature relevance to severity:\n", scores.to_string(index=False))
    scores.to_csv(OUT_SCORES, index=False)

    # ---- 7. save clean + encoded versions ----
    out.to_csv(OUT_CLEAN, index=False)
    enc, mapping = out.copy(), {}
    for c in out.columns:
        cats = sorted(out[c].unique())
        mapping[c] = {k: i for i, k in enumerate(cats)}
        enc[c] = out[c].map(mapping[c])
    enc.to_csv(OUT_ENC, index=False)
    with open(OUT_MAP, "w") as f:
        json.dump(mapping, f, indent=2)
    print(f"\nSaved: {OUT_CLEAN}, {OUT_ENC}, {OUT_MAP}, {OUT_SCORES}")


if __name__ == "__main__":
    main()
