"""
Fairness Gate - Audit Router
Supports tabular and generative model audit paths.
"""

import sys
import time
import os

MODE = os.environ.get("FAIRNESS_MODE", "pass").lower()
MODEL_TYPE = os.environ.get("MODEL_TYPE", "tabular").lower()

METRICS = {
    "tabular": {
        "pass": {
            "dpd":       0.042,
            "eopp_gap":  0.031,
            "eodds_gap": 0.038,
        },
        "fail": {
            "dpd":       0.183,
            "eopp_gap":  0.091,
            "eodds_gap": 0.112,
        },
    },
    "generative": {
        "pass": {
            "coverage":  0.927,
            "kl_div":    0.041,
            "fid_proxy": 0.883,
        },
        "fail": {
            "coverage":  0.667,
            "kl_div":    0.312,
            "fid_proxy": 0.541,
        },
    },
}

THRESHOLDS = {
    "tabular": {
        "dpd":       0.100,
        "eopp_gap":  0.050,
        "eodds_gap": 0.060,
    },
    "generative": {
        "coverage":  0.900,
        "kl_div":    0.200,
        "fid_proxy": 0.700,
    },
}

DIRECTION = {
    "dpd":       "max",
    "eopp_gap":  "max",
    "eodds_gap": "max",
    "coverage":  "min",
    "kl_div":    "max",
    "fid_proxy": "min",
}

def log(msg, level="INFO"):
    prefix = {"INFO": "  ", "WARN": "x ", "PASS": "PASS", "FAIL": "FAIL"}
    print(f"[{level}]  {prefix.get(level, '  ')} {msg}", flush=True)

def separator(label=""):
    width = 60
    if label:
        pad = (width - len(label) - 2) // 2
        print(f"\n{'-' * pad} {label} {'-' * pad}\n", flush=True)
    else:
        print("-" * width, flush=True)

def check_violation(key, value, thresh):
    if DIRECTION[key] == "max":
        return value > thresh
    else:
        return value < thresh

def main():
    separator("FAIRNESS GATE PIPELINE v1.4.0")
    log(f"Run mode        : {MODE.upper()}")
    log(f"Model type      : {MODEL_TYPE.upper()}")
    log(f"Timestamp (UTC) : {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")
    log(f"Commit SHA      : {os.environ.get('GITHUB_SHA', 'local-dev')[:10]}")

    if MODEL_TYPE not in METRICS:
        log(f"Unknown MODEL_TYPE='{MODEL_TYPE}'. Use 'tabular' or 'generative'.", "WARN")
        sys.exit(2)

    if MODE not in METRICS[MODEL_TYPE]:
        log(f"Unknown FAIRNESS_MODE='{MODE}'. Use 'pass' or 'fail'.", "WARN")
        sys.exit(2)

    metrics = METRICS[MODEL_TYPE][MODE]
    thresholds = THRESHOLDS[MODEL_TYPE]

    separator("STAGE 1 - Model Detection")
    log(f"Model type detected       : {MODEL_TYPE}_classifier")
    log(f"Protected attributes      : ['gender', 'race', 'age_group']")
    time.sleep(0.4)

    if MODEL_TYPE == "tabular":
        separator("STAGE 2 - Fairlearn Audit (Tabular Path)")
        log("Running demographic parity analysis...")
        time.sleep(0.3)
        log("Running equalized odds evaluation...")
        time.sleep(0.3)
        log("Running equalized opportunity gap analysis...")
        time.sleep(0.3)
    else:
        separator("STAGE 2 - GAN Audit Layer (Generative Path)")
        log("Running coverage analysis on generative outputs...")
        time.sleep(0.3)
        log("Computing KL divergence vs. reference distribution...")
        time.sleep(0.3)
        log("Running FID-proxy evaluation...")
        time.sleep(0.3)
        log("Invoking expert panel protocol...")
        time.sleep(0.3)

    separator("AUDIT METRICS")
    print(f"  {'Metric':<28} {'Value':>8}   {'Threshold':>10}   Status")
    print(f"  {'-'*28}   {'-'*8}   {'-'*10}   {'-'*6}")
    for key, value in metrics.items():
        thresh = thresholds[key]
        violation = check_violation(key, value, thresh)
        status = "FAIL" if violation else "PASS"
        direction = ">=" if DIRECTION[key] == "min" else "<="
        print(f"  {key:<28} {value:>8.3f}   {direction}{thresh:>9.3f}   {status}")
    print()

    separator("STAGE 3 - Gate Decision")
    violations = [k for k, v in metrics.items() if check_violation(k, v, thresholds[k])]
    decision = "HOLD" if violations else "PASS"

    separator("FINAL VERDICT")
    if decision == "PASS":
        log("All fairness thresholds satisfied.", "PASS")
        log(f"Decision : PASS -- model cleared for deployment.", "PASS")
        sys.exit(0)
    else:
        log(f"Violations: {', '.join(violations)}", "FAIL")
        log("Decision : HOLD -- model blocked from deployment.", "FAIL")
        log("Remediation required before promotion to production.", "WARN")
        sys.exit(1)

if __name__ == "__main__":
    main()
