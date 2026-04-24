import os
import glob
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------- Small helpers ----------

def slugify(name: str) -> str:
    return re.sub(r"[^0-9A-Za-z]+", "_", name).strip("_")


def ensure_output_dir(dirname: str = "figures"):
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    return dirname


# ---------- 1. Global PF ranking bar plot ----------

def plot_global_pf_ranking(csv_path="pf_ranking_global.csv", top_n=20):
    """
    Bar plot of the top N heroes by global PF rating.
    """
    df = pd.read_csv(csv_path)
    df_sorted = df.sort_values("pf_rating", ascending=False).head(top_n)

    plt.figure(figsize=(10, 6))
    plt.bar(df_sorted["hero"], df_sorted["pf_rating"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("PF rating")
    plt.title(f"Global PF Ranking (Top {top_n})")

    outdir = ensure_output_dir()
    outpath = os.path.join(outdir, "global_pf_ranking_top{}.png".format(top_n))
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()
    print(f"[plot_global_pf_ranking] Saved figure to {outpath}")


# ---------- 2. PF ranking by map type ----------

def plot_pf_by_map_type(map_types=None):
    """
    For each map_type (Hybrid, Escort, Control, Push, Flashpoint),
    make a bar plot of the top 10 heroes by PF rating.
    """
    if map_types is None:
        # These should match the map_type values in your DataFrame
        map_types = ["Hybrid", "Escort", "Control", "Push", "Flashpoint"]

    outdir = ensure_output_dir()

    for mt in map_types:
        csv_name = f"pf_ranking_{mt}.csv"
        if not os.path.exists(csv_name):
            print(f"[plot_pf_by_map_type] WARNING: {csv_name} not found, skipping.")
            continue

        df = pd.read_csv(csv_name)
        df_sorted = df.sort_values("pf_rating", ascending=False).head(10)

        plt.figure(figsize=(10, 6))
        plt.bar(df_sorted["hero"], df_sorted["pf_rating"])
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("PF rating")
        plt.title(f"PF Ranking by Map Type: {mt} (Top 10)")

        outpath = os.path.join(outdir, f"pf_ranking_by_maptype_{slugify(mt)}.png")
        plt.tight_layout()
        plt.savefig(outpath, dpi=200)
        plt.close()
        print(f"[plot_pf_by_map_type] Saved figure to {outpath}")


# ---------- 3. Heatmap: PF ratings (hero × map) ----------

def load_per_map_pf_tables(pattern="pf_ranking_map_*.csv"):
    """
    Load all per-map PF ranking CSVs (pf_ranking_map_XXX.csv)
    and assemble them into a DataFrame where:
      - index = hero
      - columns = map name
      - values = PF rating
    """
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files found matching pattern {pattern}")

    matrices = []
    map_names = []

    for path in files:
        df = pd.read_csv(path)
        # Extract map name from filename: pf_ranking_map_<slug>.csv
        base = os.path.basename(path)
        # Just use the slug as label, or you can map back to pretty names
        slug = base.replace("pf_ranking_map_", "").replace(".csv", "")
        map_names.append(slug)

        # Use hero as index, pf_rating as column
        mat = df.set_index("hero")["pf_rating"]
        matrices.append(mat)

    # Concatenate into DataFrame with heroes as index
    heat_df = pd.concat(matrices, axis=1)
    heat_df.columns = map_names

    # Sort heroes by global mean PF rating, so heatmap has a meaningful order
    mean_pf = heat_df.mean(axis=1)
    heat_df = heat_df.loc[mean_pf.sort_values(ascending=False).index]

    return heat_df


def plot_pf_heatmap_by_map(pattern="pf_ranking_map_*.csv"):
    """
    Create a heatmap where:
      - rows = heroes
      - columns = maps
      - cell values = PF rating
    """
    heat_df = load_per_map_pf_tables(pattern=pattern)

    plt.figure(figsize=(max(12, 0.4 * heat_df.shape[1]), 10))
    plt.imshow(heat_df.values, aspect="auto")
    plt.colorbar(label="PF rating")

    plt.yticks(np.arange(len(heat_df.index)), heat_df.index)
    plt.xticks(np.arange(len(heat_df.columns)), heat_df.columns, rotation=90)

    plt.title("Per-Map Perron–Frobenius Ratings (Heroes × Maps)")
    plt.xlabel("Map")
    plt.ylabel("Hero")

    outdir = ensure_output_dir()
    outpath = os.path.join(outdir, "pf_heatmap_hero_vs_map.png")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()
    print(f"[plot_pf_heatmap_by_map] Saved heatmap to {outpath}")


# ---------- 4. Variance plot (map dependence) ----------

def plot_map_dependence_variance(csv_path="pf_map_dependence_variance.csv", top_n=15):
    """
    Bar plot of the heroes with highest variance in PF rating across map types.
    """
    df = pd.read_csv(csv_path)
    df_sorted = df.sort_values("variance", ascending=False).head(top_n)

    plt.figure(figsize=(10, 6))
    plt.bar(df_sorted["hero"], df_sorted["variance"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Variance of PF rating across map types")
    plt.title(f"Most Map-Dependent Heroes (Top {top_n} by variance)")

    outdir = ensure_output_dir()
    outpath = os.path.join(outdir, f"pf_map_dependence_variance_top{top_n}.png")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()
    print(f"[plot_map_dependence_variance] Saved figure to {outpath}")


# ---------- Main hook to run everything ----------

def main():
    # 1) Global PF bar plot
    plot_global_pf_ranking(top_n=20)

    # 2) PF ranking by map type
    plot_pf_by_map_type()

    # 3) Per-map hero PF heatmap
    plot_pf_heatmap_by_map()

    # 4) Variance plot (map dependence)
    plot_map_dependence_variance(top_n=15)


if __name__ == "__main__":
    main()