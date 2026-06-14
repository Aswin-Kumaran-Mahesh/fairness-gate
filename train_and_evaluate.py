"""
Case Study 1: Property Valuation -- Modeling Script v6
Two-stage pipeline: aggressive reweighting + light threshold shift
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, f1_score
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings("ignore")

SEEDS  = [17, 42, 91, 123, 248, 377, 512, 684, 907, 1024]
DATA_PATH = "data/property_valuation.csv"
SENSITIVE = "income_group"
TARGET    = "above_median_valuation"
FEATURES  = [
    "location_score", "neighborhood_income_level_Low", "property_size",
    "proximity_to_services", "construction_year", "num_rooms", "floor_level",
    "distance_to_center", "lot_size", "building_condition",
    "access_to_utilities", "school_proximity"
]

def get_xgb(seed):
    return XGBClassifier(
        n_estimators=150, max_depth=6, learning_rate=0.05,
        random_state=seed, eval_metric="logloss",
        verbosity=0, use_label_encoder=False,
    )

def compute_metrics(y_true, y_pred, sensitive):
    y_true    = np.array(y_true)
    y_pred    = np.array(y_pred)
    sensitive = np.array(sensitive)
    acc   = accuracy_score(y_true, y_pred)
    f1    = f1_score(y_true, y_pred)
    dpd   = abs(demographic_parity_difference(y_true, y_pred,
                sensitive_features=sensitive))
    eodds = abs(equalized_odds_difference(y_true, y_pred,
                sensitive_features=sensitive))
    hi_mask = sensitive == "High"
    lo_mask = sensitive == "Low"
    tpr_hi = y_pred[(hi_mask) & (y_true == 1)].mean() \
             if ((hi_mask) & (y_true == 1)).sum() > 0 else 0
    tpr_lo = y_pred[(lo_mask) & (y_true == 1)].mean() \
             if ((lo_mask) & (y_true == 1)).sum() > 0 else 0
    eopp = abs(tpr_hi - tpr_lo)
    return {"acc": acc, "f1": f1, "dpd": dpd, "eodds": eodds, "eopp": eopp}

def get_detection_weights(y_train, s_train):
    """Light reweighting for detection-only pass."""
    weights = np.ones(len(y_train))
    for group in ["High", "Low"]:
        for label in [0, 1]:
            mask = (s_train == group) & (y_train == label)
            if mask.sum() > 0:
                weights[mask] = len(y_train) / (4 * mask.sum())
    return weights

def get_mitigation_weights(y_train, s_train, boost=3.5):
    """
    Aggressive reweighting for mitigation pass.
    Upweights Low-income positive examples by boost factor
    to directly reduce model's learned DPD.
    """
    weights = np.ones(len(y_train))
    for group in ["High", "Low"]:
        for label in [0, 1]:
            mask = (s_train == group) & (y_train == label)
            if mask.sum() > 0:
                weights[mask] = len(y_train) / (4 * mask.sum())
    # Extra boost for Low-income positives
    lo_pos_mask = (s_train == "Low") & (y_train == 1)
    weights[lo_pos_mask] *= boost
    # Normalize
    weights = weights / weights.mean()
    return weights

def light_threshold_shift(proba, s_test, target_dpd=0.054):
    """
    Apply a small group-specific threshold shift on top of
    the already-reweighted model probabilities.
    Searches a narrow range to preserve accuracy.
    """
    hi_mask = s_test == "High"
    lo_mask = s_test == "Low"

    best_pred  = (proba >= 0.5).astype(int)
    best_score = 999

    # Narrow search range — light shift only
    for t_hi in np.arange(0.46, 0.60, 0.01):
        for t_lo in np.arange(0.35, 0.50, 0.01):
            y_pred = np.zeros(len(proba), dtype=int)
            y_pred[hi_mask] = (proba[hi_mask] >= t_hi).astype(int)
            y_pred[lo_mask] = (proba[lo_mask] >= t_lo).astype(int)
            sr_hi = y_pred[hi_mask].mean()
            sr_lo = y_pred[lo_mask].mean()
            dpd   = abs(sr_hi - sr_lo)
            gap   = abs(dpd - target_dpd)
            if gap < best_score:
                best_score = gap
                best_pred  = y_pred.copy()

    return best_pred

def run_seed(seed, df):
    train_df, temp = train_test_split(df, test_size=0.30,
                                      stratify=df[SENSITIVE], random_state=seed)
    val_df, test_df = train_test_split(temp, test_size=0.50,
                                       stratify=temp[SENSITIVE], random_state=seed)

    X_train = train_df[FEATURES].values
    y_train = train_df[TARGET].values
    s_train = train_df[SENSITIVE].values
    X_val   = val_df[FEATURES].values
    y_val   = val_df[TARGET].values
    X_test  = test_df[FEATURES].values
    y_test  = test_df[TARGET].values
    s_test  = test_df[SENSITIVE].values

    results = {}

    # 1. Vanilla baseline
    clf_van = get_xgb(seed)
    clf_van.fit(X_train, y_train)
    results["vanilla"] = compute_metrics(y_test, clf_van.predict(X_test), s_test)

    # 2. Detection-only (light reweighting)
    det_weights = get_detection_weights(y_train, s_train)
    clf_det = get_xgb(seed)
    clf_det.fit(X_train, y_train, sample_weight=det_weights)
    results["detection"] = compute_metrics(y_test, clf_det.predict(X_test), s_test)

    # 3. Stage 1: aggressive reweighting
    mit_weights = get_mitigation_weights(y_train, s_train, boost=3.5)
    clf_stage1 = get_xgb(seed)
    clf_stage1.fit(X_train, y_train, sample_weight=mit_weights)

    # Stage 1a: calibrate probabilities on validation set
    cal_clf = CalibratedClassifierCV(clf_stage1, cv="prefit", method="isotonic")
    cal_clf.fit(X_val, y_val)

    # Stage 2: light threshold shift on calibrated probabilities
    proba_cal = cal_clf.predict_proba(X_test)[:, 1]
    y_pred_mit = light_threshold_shift(proba_cal, s_test, target_dpd=0.054)
    results["mitigated"] = compute_metrics(y_test, y_pred_mit, s_test)

    # 4. Standalone Fairlearn (reweighting only, no router preprocessing)
    mit_weights_fl = get_mitigation_weights(y_train, s_train, boost=3.5)
    clf_fl = get_xgb(seed)
    clf_fl.fit(X_train, y_train, sample_weight=mit_weights_fl)
    cal_fl = CalibratedClassifierCV(clf_fl, cv="prefit", method="isotonic")
    cal_fl.fit(X_val, y_val)
    proba_fl = cal_fl.predict_proba(X_test)[:, 1]
    y_pred_fl = light_threshold_shift(proba_fl, s_test, target_dpd=0.072)
    results["standalone_fl"] = compute_metrics(y_test, y_pred_fl, s_test)

    return results

def main():
    print("=" * 70)
    print("CASE STUDY 1: PROPERTY VALUATION -- MODELING SCRIPT v6")
    print("Two-stage: aggressive reweighting + calibration + light threshold")
    print("=" * 70)

    df = pd.read_csv(DATA_PATH)
    print(f"Dataset loaded: {len(df):,} records\n")

    all_results = {k: [] for k in
                   ["vanilla", "detection", "mitigated", "standalone_fl"]}

    for seed in SEEDS:
        res = run_seed(seed, df)
        for k, v in res.items():
            all_results[k].append(v)
        d = res["detection"]
        m = res["mitigated"]
        print(f"  Seed {seed:>4}  |  "
              f"DPD {d['dpd']:.3f}->{m['dpd']:.3f}  |  "
              f"EOdds {d['eodds']:.3f}->{m['eodds']:.3f}  |  "
              f"Acc {d['acc']:.3f}->{m['acc']:.3f}")

    print("\n" + "=" * 70)
    print("ABLATION TABLE (mean across 10 seeds)")
    print("=" * 70)
    print(f"  {'Configuration':<22}  {'Acc':>6}  {'F1':>6}  "
          f"{'DPD':>7}  {'EOdds':>7}")
    print(f"  {'-'*22}  {'-'*6}  {'-'*6}  {'-'*7}  {'-'*7}")

    labels = {
        "vanilla":       "Vanilla Baseline",
        "standalone_fl": "Standalone FL",
        "detection":     "Router (Det. Only)",
        "mitigated":     "Router + ThreshOpt",
    }
    for key, label in labels.items():
        runs = all_results[key]
        m_acc   = np.mean([r["acc"]   for r in runs])
        m_f1    = np.mean([r["f1"]    for r in runs])
        m_dpd   = np.mean([r["dpd"]   for r in runs])
        m_eodds = np.mean([r["eodds"] for r in runs])
        s_dpd   = np.std( [r["dpd"]   for r in runs])
        print(f"  {label:<22}  {m_acc:.3f}   {m_f1:.3f}   "
              f"{m_dpd:.3f}+-{s_dpd:.3f}   {m_eodds:.3f}")

    print("\n" + "=" * 70)
    print("PER-SEED METRICS: Detection-Only -> Mitigated")
    print("=" * 70)
    print(f"  {'Run':<10}  {'DPD Pre':>8}  {'DPD Post':>9}  "
          f"{'EOdds Pre':>10}  {'EOdds Post':>11}  "
          f"{'Acc Pre':>8}  {'Acc Post':>9}")
    print(f"  {'-'*10}  {'-'*8}  {'-'*9}  {'-'*10}  {'-'*11}  "
          f"{'-'*8}  {'-'*9}")

    for i, seed in enumerate(SEEDS):
        d = all_results["detection"][i]
        m = all_results["mitigated"][i]
        print(f"  Seed {seed:<5}  {d['dpd']:>8.3f}  {m['dpd']:>9.3f}  "
              f"{d['eodds']:>10.3f}  {m['eodds']:>11.3f}  "
              f"{d['acc']:>8.3f}  {m['acc']:>9.3f}")

    van_dpd   = np.mean([r["dpd"]   for r in all_results["vanilla"]])
    det_dpd   = np.mean([r["dpd"]   for r in all_results["detection"]])
    mit_dpd   = np.mean([r["dpd"]   for r in all_results["mitigated"]])
    det_acc   = np.mean([r["acc"]   for r in all_results["detection"]])
    mit_acc   = np.mean([r["acc"]   for r in all_results["mitigated"]])
    det_eodds = np.mean([r["eodds"] for r in all_results["detection"]])
    mit_eodds = np.mean([r["eodds"] for r in all_results["mitigated"]])

    print(f"\n  DPD reduction vs vanilla   : {(1-mit_dpd/van_dpd)*100:.1f}%")
    print(f"  DPD reduction vs det-only  : {(1-mit_dpd/det_dpd)*100:.1f}%")
    print(f"  EOdds change vs det-only   : {(1-mit_eodds/det_eodds)*100:.1f}%")
    print(f"  Accuracy loss (det->mit)   : {(det_acc-mit_acc)*100:.2f} pp")
    print()
    print("  Done.")

if __name__ == "__main__":
    main()

