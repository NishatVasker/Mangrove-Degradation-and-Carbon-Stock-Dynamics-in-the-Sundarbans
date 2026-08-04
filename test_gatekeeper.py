#!/usr/bin/env python3
"""
Adversarial stress test for the Confidence Gatekeeper.

The gate's whole claim is that it flags predictions the model should not be
trusted on. In-distribution testing cannot verify that: if test data looks like
training data, everything is legitimately trustworthy and REJECT never fires.
So we inject three distinct failure modes and check the gate catches each.
"""
import numpy as np, pandas as pd, sys
sys.path.insert(0, "/home/claude")
from uhi_pipeline import ConfidenceGatekeeper
import lightgbm as lgb

rng = np.random.default_rng(3)
N, P = 12000, 6

def truth(X):
    return (30 + 8*X[:,0] - 6*X[:,1] + 3*X[:,2]*X[:,3]
            - 2*X[:,4] + rng.normal(0, 0.4, len(X)))

X = rng.normal(0, 1, (N, P))
y = truth(X)
tr, cal, te = np.split(rng.permutation(N), [7000, 9500])

model = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05,
                          random_state=0, verbose=-1).fit(X[tr], y[tr])
gk = ConfidenceGatekeeper().fit(model, X[cal], y[cal], X[tr])

print("="*74)
print("CONFIDENCE GATEKEEPER — ADVERSARIAL VALIDATION")
print("="*74)

def evaluate(name, Xq, yq):
    sc = gk.score(Xq)
    err = np.abs(yq - sc.prediction.to_numpy())
    cov = ((yq >= sc.lower_90) & (yq <= sc.upper_90)).mean()
    print(f"\n{name}")
    print(f"  coverage {cov*100:5.1f}% | mean |err| {err.mean():5.2f} | "
          f"mean width {sc.interval_width.mean():5.2f}")
    parts = []
    for v in ("ACCEPT","REVIEW","REJECT"):
        m = (sc.verdict==v).to_numpy()
        if m.sum():
            parts.append(f"{v} {100*m.mean():4.1f}% (MAE {err[m].mean():.2f})")
        else:
            parts.append(f"{v}  0.0%")
    print("  " + " | ".join(parts))
    return sc, err

# 1. in-distribution baseline
sc0, e0 = evaluate("[1] In-distribution (gate should mostly ACCEPT)", X[te], y[te])

# 2. covariate shift: pixels far outside the training feature envelope
X_ood = rng.normal(0, 1, (2000, P)) + 4.0
evaluate("[2] Out-of-domain shift (+4 sigma) — gate should REJECT", X_ood, truth(X_ood))

# 3. concept drift: same feature space, different physics
X_d = rng.normal(0, 1, (2000, P))
y_d = 30 - 8*X_d[:,0] + 6*X_d[:,1] + rng.normal(0, .4, 2000)   # signs flipped
evaluate("[3] Concept drift (relationship inverted)", X_d, y_d)

# 4. does trust actually rank error? (the core claim)
print("\n" + "="*74)
from scipy.stats import spearmanr
rho, pv = spearmanr(sc0.trust_score, e0)
print(f"Spearman(trust, |error|) in-distribution = {rho:+.3f}  (p={pv:.2g})")
print("Expected NEGATIVE: higher trust must mean lower error.")

q = pd.qcut(sc0.trust_score, 5, labels=[f"Q{i}" for i in range(1,6)], duplicates="drop")
tab = pd.DataFrame({"trust_quintile": q, "abs_err": e0}).groupby(
    "trust_quintile", observed=True).abs_err.agg(["mean","count"])
print("\nError by trust quintile (Q1 = least trusted):")
print(tab.round(3).to_string())
lo, hi = tab["mean"].iloc[0], tab["mean"].iloc[-1]
print(f"\nQ1/Q5 error ratio = {lo/hi:.2f}x  "
      f"({'PASS - gate discriminates' if lo/hi > 1.3 else 'WEAK - gate adds little'})")
