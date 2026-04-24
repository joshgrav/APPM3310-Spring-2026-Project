print(">>> pf_analysis.py has started running")

import pandas as pd
import numpy as np
import re

def slugify(name : str) -> str:
    """Turn 'King's Row into 'Kings_row' for filenames."""
    return re.sub(r"[^0-9A-Za-z]+", "_", name).strip("_")

# ============================
# 1. Load data
# ============================

def load_winrate_data(csv_path: str = "overwatch_winrates_by_map.csv") -> pd.DataFrame:
    """
    Load the hero/map winrate CSV produced by scrape_ow_stats.py.
    """
    print(f"[load_winrate_data] loading {csv_path!r} ...")
    df = pd.read_csv(csv_path)
    required_cols = {"hero", "map", "map_type", "winrate"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")
    print(f"[load_winrate_data] loaded DataFrame with shape {df.shape}")
    return df


# ============================
# 2. Build PF matrix + eigenvector
# ============================

def compute_avg_winrates(df: pd.DataFrame) -> pd.Series:
    """
    Given a DataFrame, compute the average winrate for each hero
    across all rows in df.
    Returns a Series indexed by hero.
    """
    hero_means = df.groupby("hero")["winrate"].mean()
    print(f"[compute_avg_winrates] computed means for {len(hero_means)} heroes")
    return hero_means


def build_pf_matrix(hero_means: pd.Series, epsilon: float = 1e-3):
    """
    Construct a positive matrix A for Perron-Frobenius analysis.

    hero_means: Series indexed by hero name, containing mean winrates.

    Returns:
        A: (n x n) numpy array
        heroes: list of hero names (order corresponds to rows/cols of A)
    """
    heroes = list(hero_means.index)
    n = len(heroes)
    w = hero_means.to_numpy().reshape((n, 1))  # column vector

    # B_ij = w_i - w_j
    B = w - w.T

    # Shift to make everything positive
    B_min = B.min()
    if B_min <= 0:
        A = B - B_min + epsilon
    else:
        A = B.copy()

    print(f"[build_pf_matrix] built A of shape {A.shape}")
    return A, heroes


def dominant_eigenpair(A: np.ndarray):
    """
    Compute the dominant eigenvalue and eigenvector of A.
    Returns (lambda_max, v) where v is normalized to sum to 1.
    """
    vals, vecs = np.linalg.eig(A)

    # index of eigenvalue with largest real part
    idx = np.argmax(vals.real)
    lambda_max = vals[idx].real
    v = vecs[:, idx].real  # take real part

    # Flip sign if needed so sum is positive
    if v.sum() < 0:
        v = -v

    # Normalize to sum to 1
    v = v / v.sum()

    print(f"[dominant_eigenpair] lambda_max = {lambda_max:.4f}")
    return lambda_max, v


def compute_pf_ratings(hero_means: pd.Series) -> pd.DataFrame:
    """
    Compute Perron-Frobenius ratings from hero mean winrates.
    Returns a DataFrame with columns: rank, hero, pf_rating.
    """
    A, heroes = build_pf_matrix(hero_means)
    lambda_max, v = dominant_eigenpair(A)

    ratings_df = pd.DataFrame({
        "hero": heroes,
        "pf_rating": v
    })

    # Sort descending and add rank
    ratings_df = ratings_df.sort_values("pf_rating", ascending=False)
    ratings_df["rank"] = np.arange(1, len(ratings_df) + 1)
    ratings_df = ratings_df[["rank", "hero", "pf_rating"]]

    print("[compute_pf_ratings] produced ratings DataFrame")
    return ratings_df


# ============================
# 3. Ratings by map_type + variance
# ============================

def compute_ratings_by_map_type(df: pd.DataFrame):
    """
    Compute PF ratings separately for each map_type.
    Returns a dict: map_type -> ratings DataFrame.
    """
    results = {}
    for mt in sorted(df["map_type"].unique()):
        print(f"[compute_ratings_by_map_type] processing map_type = {mt}")
        sub = df[df["map_type"] == mt]
        hero_means = compute_avg_winrates(sub)
        ratings = compute_pf_ratings(hero_means)
        results[mt] = ratings
    return results


def compute_map_dependence_variance(ratings_by_type: dict) -> pd.DataFrame:
    """
    Given map_type -> ratings_df, compute the variance in PF rating
    for each hero across map types.
    Returns a DataFrame with columns: hero, mean_rating, variance, num_types.
    """
    frames = []
    for mt, df_mt in ratings_by_type.items():
        tmp = df_mt[["hero", "pf_rating"]].rename(columns={"pf_rating": f"pf_{mt}"})
        frames.append(tmp)

    merged = frames[0]
    for f in frames[1:]:
        merged = pd.merge(merged, f, on="hero", how="outer")

    pf_cols = [c for c in merged.columns if c.startswith("pf_")]
    merged["mean_rating"] = merged[pf_cols].mean(axis=1, skipna=True)
    merged["variance"] = merged[pf_cols].var(axis=1, skipna=True)
    merged["num_types"] = merged[pf_cols].notna().sum(axis=1)

    merged = merged.sort_values("variance", ascending=False)

    print("[compute_map_dependence_variance] computed variance table")
    return merged[["hero", "mean_rating", "variance", "num_types"]]


def compute_ratings_by_map(df: pd.DataFrame):
    """
    Compute PF ratings separately for each individual map
    (all data are Competitive - Role Queue).
    Returns a dict: map_name -> ratings DataFrame.
    """
    results = {}
    for m in sorted(df["map"].unique()):
        print(f"[compute_ratings_by_map] processing map = {m}")
        sub = df[df["map"] == m]
        hero_means = compute_avg_winrates(sub)
        ratings = compute_pf_ratings(hero_means)
        results[m] = ratings
    return results

# ============================
# 4. Main driver
# ============================

def main():
    print("[main] starting PF analysis")
    df = load_winrate_data()

    # --- Global PF ranking (all maps combined) ---
    hero_means_global = compute_avg_winrates(df)
    global_ratings = compute_pf_ratings(hero_means_global)
    print("\nGlobal hero ranking (top 10):")
    print(global_ratings.head(10))

    global_ratings.to_csv("pf_ranking_global.csv", index=False)
    with open("pf_ranking_global.tex", "w", encoding="utf-8") as f:
        f.write(global_ratings.to_latex(index=False, float_format="%.4f"))

    # --- Rankings by map_type ---
    ratings_by_type = compute_ratings_by_map_type(df)
    for mt, ratings in ratings_by_type.items():
        print(f"\nTop 5 heroes for map_type = {mt}:")
        print(ratings.head(5))
        ratings.to_csv(f"pf_ranking_{mt}.csv", index=False)
        with open(f"pf_ranking_{mt}.tex", "w", encoding="utf-8") as f:
            f.write(ratings.to_latex(index=False, float_format="%.4f"))

      # --- Rankings by individual map (still only Comp Role Queue) ---
    ratings_by_map = compute_ratings_by_map(df)
    for m, ratings in ratings_by_map.items():
        print(f"\nTop 5 heroes for map = {m}:")
        print(ratings.head(5))
        map_slug = slugify(m)
        ratings.to_csv(f"pf_ranking_map_{map_slug}.csv", index=False)
        with open(f"pf_ranking_map_{map_slug}.tex", "w", encoding="utf-8") as f:
            f.write(ratings.to_latex(index=False, float_format="%.4f"))


    # --- Map-dependence variance ---
    var_df = compute_map_dependence_variance(ratings_by_type)
    print("\nMost map-dependent heroes (top 10 by variance):")
    print(var_df.head(10))

    var_df.to_csv("pf_map_dependence_variance.csv", index=False)
    with open("pf_map_dependence_variance.tex", "w", encoding="utf-8") as f:
        f.write(var_df.to_latex(index=False, float_format="%.4f"))

    print("[main] PF analysis complete")


if __name__ == "__main__":
    print("[pf_analysis] __main__ block reached")
    main()
