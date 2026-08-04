# Dhaka UHI — analysis toolkit

Live link : https://nishatvaskersundorban.netlify.app/

<img width="1920" height="1200" alt="image" src="https://github.com/user-attachments/assets/cda63b3a-ddb9-44bb-98b1-f9bcac88f4d5" />

Two things here. **Your data still hasn't reached me** — `/mnt/user-data/uploads/` has been empty every time I've checked across this whole session — so both were validated against synthetic data engineered to look like your exports.

| | What it is | How to use it |
|---|---|---|
| **`uhi_explorer.html`** | Interactive viewer. Drop a CSV, see results immediately. | Double-click it. No install, no server, no internet. |
| **`pipeline/uhi_pipeline.py`** | The full analysis. Publication figures + tables + report. | `python3 uhi_pipeline.py` |

Start with the HTML — it needs nothing installed and will tell you within seconds whether your columns are being read correctly. Then run the Python for the real output.

---

## 1. The interactive viewer

Open `uhi_explorer.html` in any browser. Click **Load demo data** to see it working before you trust it with yours, then drop your own CSV on it.

It detects your columns automatically (NDVI, NDBI, NDWI, albedo, coordinates, year, LST — in GEE, ArcGIS, and most other naming conventions), converts Kelvin to Celsius, fixes integer-scaled indices, and then runs: VIF screening, gradient-boosted trees under both random and spatially-blocked cross-validation, permutation importance, partial dependence, k-means climate zones, and Mann–Kendall/Sen if there's a year column. If it picks the wrong target column you can override it in section 02 and re-run.

Everything runs on your machine. Nothing is uploaded anywhere.

**Verified against known ground truth** (not just "it renders"):

| Component | Test | Result |
|---|---|---|
| OLS solver | recover coefficients 30, 8, −5, 0.4, 0 | 29.99, 8.00, −5.01, 0.41, −0.01 |
| Boosted trees | 5-fold CV on known signal | R² 0.982 |
| Permutation importance | rank a > b > c > noise | correct order, correct directions |
| VIF | detect a 0.97-correlated pair | 1062 on both, 1.0 on the clean one |
| Mann–Kendall / Sen | recover slope 0.050/yr | 0.0526, p < 0.001 |
| Moran's I | smooth field vs white noise | 0.952 vs 0.048 |
| k-means | separate two clear blobs | exact |
| Full render path | 19 DOM checks | 19/19 |

On the demo data it ranks the hottest cluster as *high NDBI / low NDWI* at 34.2 °C and the coolest as *low NDBI / high NDWI* at 27.7 °C — built-up-and-dry hottest, wet-and-vegetated coolest, which is the physically correct ordering.

---

## 2. The pipeline

```bash
python3 pipeline/uhi_pipeline.py          # full
python3 pipeline/uhi_pipeline.py --fast   # ~2x quicker
```

Outputs go to `/mnt/user-data/outputs/`: `ANALYSIS_REPORT.md`, `figures/`, `tables/`. See `sample_outputs/` for what it produces (from synthetic data — the numbers are not yours).

Stages: schema-agnostic ingestion → cleaning and VIF → spatial blocked CV → 9-model benchmark plus stacking → leakage audit → GWR → SHAP → confidence gating → Mann–Kendall/Sen and change detection → climate zones → report.

### Figures are journal-ready

89 mm and 183 mm column widths, 600 dpi PNG **and vector PDF** of every panel, 7–8 pt Arial, Okabe–Ito colourblind-safe palette, perceptually uniform ramps, panel labels (a)/(b)/(c), captions embedded, `pdf.fonttype 42` so text stays editable in Illustrator. 27 figures, each in both formats.

---

## 3. Three findings that matter more than the code

**Target autocorrelation is the wrong leakage diagnostic.** My synthetic LST had Moran's I ≈ 0.95, yet random-vs-spatial optimism was only +0.002. High autocorrelation in the *target* is fine — good predictors reproduce that structure legitimately. What leaks is autocorrelated **residual**. The pipeline now measures both, and in validation this discriminated cleanly: per-zone residuals came back unstructured (0.035–0.044, matching the negligible optimism), the pooled model hit 0.352 and was correctly flagged. This corrects what I told you in my first message.

**The Confidence Gatekeeper has a characterised blind spot.** Adversarial testing (`pipeline/test_gatekeeper.py`):

| Test | Result | |
|---|---|---|
| In-distribution | Accept MAE 0.53 / Review 0.75 / **Reject 1.49** | pass — 2.8× separation |
| Out-of-domain (+4σ) | 100% Reject | pass |
| **Concept drift** (relationship inverted) | **60.5% still Accept**, MAE 12.89 | **fail** |
| Trust ranks error | Spearman −0.259, Q1/Q5 = 2.60× | pass |

The concept-drift failure is structural, not a bug: conformal calibration assumes exchangeability and the applicability domain only sees feature geometry, so when features look normal but the physics has changed, both gates are blind. This is directly relevant to the JCU project — train on 2020 Landsat 8 OLI/TIRS, apply to 1995 Landsat 5 TM, and you have sensor drift plus three decades of urban change. The mitigation is a temporal holdout, which sits outside what the gate can self-detect. Write that limitation in rather than around; a scoring system with a mapped blind spot is a stronger contribution than one claiming universal coverage.

**GWR's naive non-stationarity test produces false positives.** My first version flagged five drivers as spatially non-stationary on data where the coefficients are stationary *by construction*. Local coefficients always wobble — a regression on a few hundred neighbours is noisy — and comparing that wobble to a global standard error computed on the full sample flags almost anything. Replaced with a two-stage test: an AICc gate at the model level, then a Monte Carlo permutation null per driver. False positives went 5 → 0.

---

## 4. Everything the test suite caught

1. ArcGIS `POINT_X`/`POINT_Y` undetected → a zone silently lost true spatial blocking
2. SHAP crashed on non-tree models (index misalignment after Kernel explainer truncation)
3. Gatekeeper ACCEPT fired on 0% of samples — thresholds were unreachable constants
4. Conformal intervals under-covered (84–90% vs 90%) — wrong formulation, fixed to normalised conformal
5. Change detection matched raw column spellings, which never agree across exports
6. Pooled model crashed — coordinates weren't renamed to the canonical vocabulary
7. GWR false positives (above)
8. Trust scores crushed to ~0.35 with a tail clipped to exactly 0 — a linear distance ratio is the wrong scale; replaced with rank-against-held-out-distribution
9. Browser engine 20 s per run → 1.3 s, by pre-binning features once instead of re-sorting at every tree node
10. Overlapping point labels and panel labels colliding with axes in two figures

Docstrings that no longer matched the code (SVR listed but absent, "split conformal" after the upgrade to normalised) were corrected, and five dead imports removed. `pyflakes` is clean; final run: **0 errors, 27 figures, 11 tables**.

---



One caveat on the sample outputs: panel (b) of the gatekeeper figures shows no separation between Accept/Review/Reject. That's honest — my synthetic data has uniform noise everywhere, so there is genuinely no local difficulty variation to find. The adversarial test proves the gate separates when real variation exists. On your data, expect separation.
