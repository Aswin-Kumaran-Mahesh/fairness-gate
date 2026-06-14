"""
Synthetic Property Valuation Dataset Generator
Boyaca Region, Colombia -- UPTC AI Laboratory
Generates 8,500 records with controlled DPD injection of 0.287
"""

import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split

SEEDS = [17, 42, 91, 123, 248, 377, 512, 684, 907, 1024]
N = 8500
N_HIGH = 4250
N_LOW = 4250
OUTPUT_DIR = "data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "property_valuation.csv")

def generate_dataset(seed=42):
    rng = np.random.default_rng(seed)
    n = N_HIGH + N_LOW

    income_group = np.array(["High"] * N_HIGH + ["Low"] * N_LOW)
    is_high = (income_group == "High").astype(float)

    location_score     = (rng.normal(0.57, 0.15, n) + 0.165 * is_high).clip(0, 1)
    neigh_low          = (rng.binomial(1, 0.45, n) - 0.245 * is_high).clip(0, 1)
    property_size      = rng.normal(118, 38, n).clip(30, 400) + 31 * is_high
    proximity          = (rng.normal(0.53, 0.15, n) + 0.14 * is_high).clip(0, 1)
    construction_year  = rng.integers(1965, 2023, n).astype(float)
    num_rooms          = rng.integers(1, 8, n).astype(float)
    floor_level        = rng.integers(1, 12, n).astype(float)
    distance_to_center = (rng.normal(6.5, 2.8, n) - 2.3 * is_high).clip(0.5, 20)
    lot_size           = rng.normal(170, 55, n).clip(40, 600) + 38 * is_high
    building_condition = (rng.normal(3.0, 0.9, n) + 0.57 * is_high).clip(1, 5)
    access_utilities   = (rng.binomial(1, 0.75, n) + 0.16 * is_high).clip(0, 1)
    school_proximity   = (rng.normal(0.52, 0.15, n) + 0.12 * is_high).clip(0, 1)

    df = pd.DataFrame({
        "location_score":                location_score,
        "neighborhood_income_level_Low": neigh_low,
        "property_size":                 property_size,
        "proximity_to_services":         proximity,
        "construction_year":             construction_year,
        "num_rooms":                     num_rooms,
        "floor_level":                   floor_level,
        "distance_to_center":            distance_to_center,
        "lot_size":                      lot_size,
        "building_condition":            building_condition,
        "access_to_utilities":           access_utilities,
        "school_proximity":              school_proximity,
        "income_group":                  income_group,
    })

    def norm(col):
        return (col - col.min()) / (col.max() - col.min() + 1e-9)

    score = (
          0.28 * norm(df["location_score"])
        + 0.18 * (1 - df["neighborhood_income_level_Low"])
        + 0.20 * norm(df["property_size"])
        + 0.13 * norm(df["proximity_to_services"])
        + 0.09 * norm(df["building_condition"])
        + 0.08 * df["access_to_utilities"]
        + 0.04 * norm(df["school_proximity"])
        + rng.normal(0, 0.14, n)
    )

    threshold = np.percentile(score, 50)
    df["above_median_valuation"] = (score > threshold).astype(int)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return df

def verify_dpd(df):
    hi_rate = df[df["income_group"] == "High"]["above_median_valuation"].mean()
    lo_rate = df[df["income_group"] == "Low"]["above_median_valuation"].mean()
    return abs(hi_rate - lo_rate), hi_rate, lo_rate

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("SYNTHETIC PROPERTY VALUATION DATASET GENERATOR")
    print("UPTC AI Laboratory -- Boyaca Region, Colombia")
    print("=" * 60)
    print(f"Target records  : {N:,} ({N_HIGH:,} high-income / {N_LOW:,} low-income)")
    print(f"Features        : 12")
    print(f"Target variable : above_median_valuation")
    print(f"Sensitive attr  : income_group")
    print(f"Target DPD      : ~0.287")
    print()

    dpds = []
    for seed in SEEDS:
        df = generate_dataset(seed)
        dpd, hi_rate, lo_rate = verify_dpd(df)
        dpds.append(dpd)
        print(f"  Seed {seed:>4}  |  DPD = {dpd:.3f}  |  SR_High = {hi_rate:.3f}  |  SR_Low = {lo_rate:.3f}")

    mean_dpd = np.mean(dpds)
    std_dpd  = np.std(dpds)
    print()
    print(f"  Mean DPD across {len(SEEDS)} seeds : {mean_dpd:.3f} +/- {std_dpd:.3f}")
    print()

    df_canonical = generate_dataset(42)
    df_canonical.to_csv(OUTPUT_FILE, index=False)
    print(f"  Canonical dataset saved : {OUTPUT_FILE}")
    print(f"  Records                 : {len(df_canonical):,}")
    print()

    train, temp = train_test_split(df_canonical, test_size=0.30,
                                   stratify=df_canonical["income_group"],
                                   random_state=42)
    val, test = train_test_split(temp, test_size=0.50,
                                 stratify=temp["income_group"],
                                 random_state=42)
    print("  Split summary (stratified by income_group):")
    print(f"    Train : {len(train):,} records (70%)")
    print(f"    Val   : {len(val):,} records (15%)")
    print(f"    Test  : {len(test):,} records (15%)")
    print()
    print("  Done.")

if __name__ == "__main__":
    main()
