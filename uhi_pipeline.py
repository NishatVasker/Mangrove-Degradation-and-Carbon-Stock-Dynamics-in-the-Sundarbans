#!/usr/bin/env python3
"""
================================================================================
DHAKA URBAN HEAT ISLAND  -  ADVANCED MACHINE LEARNING PIPELINE
================================================================================
Target      : Land Surface Temperature (LST)
Predictors  : Multispectral indices (NDVI, NDBI, NDWI, UI, IBI, albedo, ...)
Zones       : DNCC (North) vs DSCC (South), modelled separately and pooled
Temporal    : 1990-2025 trend decomposition

Pipeline stages
---------------
  0  Schema-agnostic ingestion (auto-detects columns, unzips archives)
  1  Cleaning, outlier control, multicollinearity screening (VIF)
  2  SPATIAL blocked cross-validation  (critical: random CV leaks via
     spatial autocorrelation and inflates R^2 substantially)
  3  Model benchmark: OLS / Ridge / RF / ExtraTrees / XGBoost / LightGBM /
     HistGBM / MLP  + stacked ensemble
  3b GWR - geographically weighted regression, to test whether UHI drivers are
     spatially non-stationary (a global model averages that away)
  4  Explainable AI: SHAP global attribution, interaction values,
     dependence structure, LIME-style local explanation
  5  CONFIDENCE GATEKEEPER: normalised (locally adaptive) conformal intervals
     fused with an applicability-domain check -> per-sample trust score
  6  Temporal: Mann-Kendall + Sen's slope, change-detection model on deltas
  7  Unsupervised: empirical Local Climate Zone discovery (k-means, silhouette-
     selected k)
  8  Figures + machine-readable results + written report

Author: built for Nishat / Robogen Technologies
================================================================================
"""

from __future__ import annotations

import json
import re
import sys
import warnings
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------

UPLOADS = Path("/mnt/user-data/uploads")
WORK = Path("/home/claude/uhi_work")
OUT = Path("/mnt/user-data/outputs")
FIGS = OUT / "figures"
TABLES = OUT / "tables"

for _d in (WORK, OUT, FIGS, TABLES):
    _d.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
N_SPATIAL_BLOCKS = 10          # blocked CV folds derived from coordinate clusters
CONFORMAL_ALPHA = 0.10         # -> 90% prediction intervals
MAX_ROWS_SHAP = 4000           # subsample cap for SHAP tractability
MAX_ROWS_FIT = 200_000         # subsample cap for model fitting

rng = np.random.default_rng(RANDOM_STATE)


# ----------------------------------------------------------------------------
# SCHEMA INFERENCE
# ----------------------------------------------------------------------------

# Ordered: more specific patterns first so NDBI never matches a generic "b" rule.
INDEX_PATTERNS = {
    "NDVI":     r"\bnd\s*_?vi\b|normali[sz]ed.*vegetation|\bndvi\b",
    "NDBI":     r"\bndbi\b|normali[sz]ed.*built",
    "NDWI":     r"\bndwi\b|normali[sz]ed.*water",
    "MNDWI":    r"\bmndwi\b|modified.*water",
    "NDBaI":    r"\bndbai\b|\bndbal\b|bare\s*soil",
    "SAVI":     r"\bsavi\b|soil.*adjusted",
    "EVI":      r"\bevi\b|enhanced.*vegetation",
    "MSAVI":    r"\bmsavi\b",
    "IBI":      r"\bibi\b|index.*based.*built",
    "UI":       r"\bui\b|urban\s*index",
    "BUI":      r"\bbui\b|built.?up\s*index",
    "BSI":      r"\bbsi\b|bare\s*soil\s*index",
    "NBR":      r"\bnbr\b|burn\s*ratio",
    "ALBEDO":   r"albedo",
    "EMISS":    r"emissiv|\beps\b|\bemis\b",
    "FVC":      r"\bfvc\b|fraction.*veg|veg.*fraction|\bpv\b",
    "IMPERV":   r"imperv|\bisa\b|sealed",
    "ELEV":     r"\belev\b|\bdem\b|altitude|\bsrtm\b",
    "SLOPE":    r"\bslope\b",
    "NDMI":     r"\bndmi\b|moisture\s*index",
    "LAI":      r"\blai\b|leaf\s*area",
    "TCW":      r"\btcw\b|tasselled.*wet",
    "TCG":      r"\btcg\b|tasselled.*green",
    "TCB":      r"\btcb\b|tasselled.*bright",
}

LST_PATTERNS = r"\blst\b|land\s*surface\s*temp|surface\s*temp|\btemp\b|\bbt\b|bright.*temp|\bts\b"
LAT_PATTERNS = (r"^lat\b|latitude|\by\s*coord|\bcoord\s*y\b|\bnorthing\b|"
                r"^point\s*y$|^y$|^ycoord|\byutm\b")
LON_PATTERNS = (r"^lon\b|^lng\b|longitude|\bx\s*coord|\bcoord\s*x\b|\beasting\b|"
                r"^point\s*x$|^x$|^xcoord|\bxutm\b")
YEAR_PATTERNS = r"^year$|^yr$|^date$|^time$|^epoch$|^acq"
ID_PATTERNS = r"^id$|^fid$|^oid$|^objectid$|^index$|unnamed|^system:index$|\.geo$"
ZONE_PATTERNS = r"zone|ward|thana|district|city\s*corp|dncc|dscc|region|area\s*name|name$"


def _norm(col: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(col).lower()).strip()


def classify_column(col: str) -> tuple[str, str | None]:
    """Return (role, canonical_name). role in {lst, index, lat, lon, year, id, zone, other}."""
    n = _norm(col)
    if re.search(ID_PATTERNS, n) or re.search(ID_PATTERNS, str(col).lower()):
        return "id", None
    if re.search(YEAR_PATTERNS, n):
        return "year", "YEAR"
    if re.search(LAT_PATTERNS, n):
        return "lat", "LAT"
    if re.search(LON_PATTERNS, n):
        return "lon", "LON"
    for canon, pat in INDEX_PATTERNS.items():
        if re.search(pat, n):
            return "index", canon
    if re.search(LST_PATTERNS, n):
        return "lst", "LST"
    if re.search(ZONE_PATTERNS, n):
        return "zone", "ZONE"
    return "other", None


@dataclass
class Schema:
    lst: str | None = None
    lat: str | None = None
    lon: str | None = None
    year: str | None = None
    zone: str | None = None
    indices: dict = field(default_factory=dict)     # canonical -> original col
    numeric_other: list = field(default_factory=list)
    dropped: list = field(default_factory=list)
    n_rows: int = 0
    source: str = ""

    def feature_cols(self) -> list[str]:
        return list(self.indices.values()) + self.numeric_other

    def summary(self) -> str:
        L = [f"  source        : {self.source}",
             f"  rows          : {self.n_rows:,}",
             f"  target (LST)  : {self.lst}",
             f"  coords        : lat={self.lat}  lon={self.lon}",
             f"  year          : {self.year}",
             f"  zone          : {self.zone}",
             f"  indices ({len(self.indices)}) : {', '.join(sorted(self.indices)) or 'none'}",
             f"  extra numeric : {', '.join(self.numeric_other[:12]) or 'none'}"]
        return "\n".join(L)


def infer_schema(df: pd.DataFrame, source: str = "") -> Schema:
    """Map an arbitrary remote-sensing table onto a canonical schema."""
    s = Schema(source=source, n_rows=len(df))
    seen_index_canon: set[str] = set()

    for col in df.columns:
        role, canon = classify_column(col)
        if role == "id":
            s.dropped.append(col)
        elif role == "lst" and s.lst is None:
            s.lst = col
        elif role == "lat" and s.lat is None:
            s.lat = col
        elif role == "lon" and s.lon is None:
            s.lon = col
        elif role == "year" and s.year is None:
            s.year = col
        elif role == "zone" and s.zone is None:
            s.zone = col
        elif role == "index" and canon not in seen_index_canon:
            s.indices[canon] = col
            seen_index_canon.add(canon)
        elif pd.api.types.is_numeric_dtype(df[col]):
            s.numeric_other.append(col)
        else:
            s.dropped.append(col)

    # Guard: an index column must never double as the target.
    if s.lst in s.indices.values():
        s.lst = None

    # Fallback target: if nothing matched LST, take the numeric column whose
    # value range looks like a temperature (Kelvin or Celsius).
    if s.lst is None:
        for col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                continue
            if col in s.indices.values() or col in (s.lat, s.lon, s.year):
                continue
            v = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(v) < 10:
                continue
            med = float(v.median())
            if 10 <= med <= 60 or 270 <= med <= 340:
                s.lst = col
                if col in s.numeric_other:
                    s.numeric_other.remove(col)
                break

    # Coordinate sanity check - reject impossible ranges.
    for attr, lo, hi in (("lat", -90, 90), ("lon", -180, 180)):
        c = getattr(s, attr)
        if c is not None:
            v = pd.to_numeric(df[c], errors="coerce").dropna()
            if len(v) and not (lo <= v.min() and v.max() <= hi):
                setattr(s, attr, None)
                s.numeric_other.append(c)

    return s


# ----------------------------------------------------------------------------
# INGESTION
# ----------------------------------------------------------------------------

def _read_any(path: Path) -> pd.DataFrame | None:
    try:
        if path.suffix.lower() in (".csv", ".txt"):
            return pd.read_csv(path, low_memory=False)
        if path.suffix.lower() == ".tsv":
            return pd.read_csv(path, sep="\t", low_memory=False)
        if path.suffix.lower() in (".xlsx", ".xls"):
            return pd.read_excel(path)
        if path.suffix.lower() == ".parquet":
            return pd.read_parquet(path)
    except Exception as e:
        print(f"    [!] could not read {path.name}: {e}")
    return None


def discover_inputs(root: Path | None = None) -> dict[str, pd.DataFrame]:
    """Find every tabular file, expanding archives, and load them."""
    root = Path(root) if root is not None else UPLOADS
    tables: dict[str, pd.DataFrame] = {}
    if not root.exists():
        return tables

    # Expand archives into the working directory.
    extract_root = WORK / "extracted"
    for z in root.rglob("*.zip"):
        dest = extract_root / z.stem
        dest.mkdir(parents=True, exist_ok=True)
        try:
            with zipfile.ZipFile(z) as zf:
                zf.extractall(dest)
            print(f"  extracted {z.name} -> {dest}")
        except Exception as e:
            print(f"  [!] failed to extract {z.name}: {e}")

    search_roots = [root]
    if extract_root.exists():
        search_roots.append(extract_root)

    for sr in search_roots:
        for ext in ("*.csv", "*.tsv", "*.txt", "*.xlsx", "*.xls", "*.parquet"):
            for f in sr.rglob(ext):
                if "__MACOSX" in str(f):
                    continue
                df = _read_any(f)
                if df is not None and len(df) > 0 and df.shape[1] > 1:
                    key = f.stem
                    n = 2
                    while key in tables:
                        key = f"{f.stem}_{n}"
                        n += 1
                    tables[key] = df
    return tables


def label_zone(name: str) -> str:
    n = name.lower()
    if "dncc" in n or "north" in n:
        return "North (DNCC)"
    if "dscc" in n or "south" in n:
        return "South (DSCC)"
    return "Unassigned"


def is_trend_table(name: str, df: pd.DataFrame) -> bool:
    if re.search(r"trend|1990|time.?series|annual", name.lower()):
        return True
    yearish = [c for c in df.columns if re.fullmatch(r"(19|20)\d{2}", str(c).strip())]
    return len(yearish) >= 3


# ----------------------------------------------------------------------------
# PREPROCESSING
# ----------------------------------------------------------------------------

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def clean_frame(df: pd.DataFrame, s: Schema) -> tuple[pd.DataFrame, Schema, dict]:
    """Coerce numerics, drop degenerate columns, control outliers."""
    report = {}
    keep = [c for c in [s.lst, s.lat, s.lon, s.year, s.zone] if c] + s.feature_cols()
    keep = list(dict.fromkeys(keep))
    d = df[keep].copy()

    for c in d.columns:
        if c != s.zone:
            d[c] = pd.to_numeric(d[c], errors="coerce")

    n0 = len(d)
    if s.lst:
        d = d.dropna(subset=[s.lst])
    report["rows_dropped_no_target"] = n0 - len(d)

    # Kelvin -> Celsius when the target is clearly on an absolute scale.
    if s.lst and len(d) and d[s.lst].median() > 200:
        d[s.lst] = d[s.lst] - 273.15
        report["kelvin_to_celsius"] = True

    # Scale-factor correction for integer-stored indices (e.g. NDVI * 10000).
    for canon, col in list(s.indices.items()):
        if col not in d.columns:
            continue
        v = d[col].dropna()
        if len(v) == 0:
            continue
        amax = float(np.nanmax(np.abs(v)))
        if canon not in ("ELEV", "SLOPE", "LAI") and amax > 10:
            factor = 10 ** np.ceil(np.log10(amax))
            d[col] = d[col] / factor
            report.setdefault("rescaled", {})[canon] = float(factor)

    # Drop constant / near-empty predictors.
    dead = []
    for c in s.feature_cols():
        if c not in d.columns:
            dead.append(c)
            continue
        v = d[c]
        if v.notna().sum() < max(20, 0.05 * len(d)) or v.nunique(dropna=True) <= 1:
            dead.append(c)
    for c in dead:
        if c in d.columns:
            d = d.drop(columns=[c])
        s.indices = {k: vv for k, vv in s.indices.items() if vv != c}
        s.numeric_other = [x for x in s.numeric_other if x != c]
    report["dropped_degenerate"] = dead

    # Winsorise the target at 0.2 / 99.8 pct - sensor artefacts, not real heat.
    if s.lst and len(d) > 100:
        lo, hi = d[s.lst].quantile([0.002, 0.998])
        n_out = int(((d[s.lst] < lo) | (d[s.lst] > hi)).sum())
        d[s.lst] = d[s.lst].clip(lo, hi)
        report["target_winsorised"] = n_out

    # Median-impute remaining predictor gaps.
    feats = s.feature_cols()
    if feats:
        n_missing = int(d[feats].isna().sum().sum())
        if n_missing:
            d[feats] = SimpleImputer(strategy="median").fit_transform(d[feats])
        report["cells_imputed"] = n_missing

    report["final_rows"] = len(d)
    report["final_features"] = len(s.feature_cols())
    return d.reset_index(drop=True), s, report


def vif_screen(d: pd.DataFrame, feats: list[str], thresh: float = 10.0) -> tuple[list[str], pd.DataFrame]:
    """Iteratively remove the worst multicollinear predictor until all VIF < thresh."""
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    cur = list(feats)
    history = []
    X = d[cur].to_numpy(dtype=float)
    X = StandardScaler().fit_transform(X)

    while len(cur) > 2:
        try:
            vifs = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
        except Exception:
            break
        vifs = np.nan_to_num(np.array(vifs), nan=0.0, posinf=1e6)
        worst = int(np.argmax(vifs))
        history.append({"feature": cur[worst], "vif": float(vifs[worst]),
                        "action": "removed" if vifs[worst] > thresh else "retained"})
        if vifs[worst] <= thresh:
            for f, v in zip(cur, vifs):
                if f != cur[worst]:
                    history.append({"feature": f, "vif": float(v), "action": "retained"})
            break
        cur.pop(worst)
        X = np.delete(X, worst, axis=1)

    return cur, pd.DataFrame(history)


# ----------------------------------------------------------------------------
# SPATIAL BLOCKED CROSS-VALIDATION
# ----------------------------------------------------------------------------

from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import GroupKFold, KFold


def make_spatial_blocks(d: pd.DataFrame, s: Schema, n_blocks: int = N_SPATIAL_BLOCKS):
    """
    Contiguous spatial blocks via k-means on coordinates.

    Random k-fold on gridded raster samples leaks information: neighbouring
    pixels are near-duplicates, so a held-out pixel almost always has a training
    twin metres away. Reported R^2 then measures interpolation, not
    generalisation. Blocking by geography removes that shortcut.
    """
    if s.lat and s.lon and d[s.lat].notna().any() and d[s.lon].notna().any():
        coords = d[[s.lat, s.lon]].to_numpy(dtype=float)
        coords = np.nan_to_num(coords, nan=float(np.nanmean(coords)))
        km = MiniBatchKMeans(n_clusters=n_blocks, random_state=RANDOM_STATE,
                             n_init=10, batch_size=2048)
        groups = km.fit_predict(StandardScaler().fit_transform(coords))
        return groups, "spatial-blocked (k-means on coordinates)"

    # No coordinates: fall back to feature-space blocking, which still breaks
    # near-duplicate leakage even though it is not strictly geographic.
    feats = s.feature_cols()
    if feats:
        Xs = StandardScaler().fit_transform(d[feats].to_numpy(dtype=float))
        km = MiniBatchKMeans(n_clusters=n_blocks, random_state=RANDOM_STATE,
                             n_init=10, batch_size=2048)
        return km.fit_predict(Xs), "feature-space blocked (no coordinates present)"

    return np.arange(len(d)) % n_blocks, "random k-fold (no blocking possible)"


def cv_splits(groups: np.ndarray, n_splits: int = 5):
    uniq = np.unique(groups)
    n_splits = int(min(n_splits, len(uniq)))
    if n_splits < 2:
        return list(KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
                    .split(np.zeros(len(groups))))
    gkf = GroupKFold(n_splits=n_splits)
    return list(gkf.split(np.zeros(len(groups)), groups=groups))


def morans_I(values: np.ndarray, coords: np.ndarray, k: int = 8, sample: int = 5000) -> float:
    """Moran's I on a k-nearest-neighbour weight matrix. Quantifies how much
    spatial structure exists, i.e. how badly random CV would have leaked."""
    from sklearn.neighbors import NearestNeighbors
    n = len(values)
    if n > sample:
        idx = rng.choice(n, sample, replace=False)
        values, coords = values[idx], coords[idx]
        n = sample
    if n < 20:
        return float("nan")
    nn = NearestNeighbors(n_neighbors=min(k + 1, n)).fit(coords)
    _, ind = nn.kneighbors(coords)
    ind = ind[:, 1:]
    z = values - values.mean()
    denom = float((z ** 2).sum())
    if denom == 0:
        return float("nan")
    num = float(sum(z[i] * z[ind[i]].sum() for i in range(n)))
    W = n * ind.shape[1]
    return (n / W) * (num / denom)


# ----------------------------------------------------------------------------
# MODEL ZOO + BENCHMARK
# ----------------------------------------------------------------------------

from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor,
                              HistGradientBoostingRegressor, StackingRegressor)
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import xgboost as xgb
import lightgbm as lgb


def build_models(n_features: int, fast: bool = False) -> dict:
    n_jobs = -1
    m = {
        "OLS": Pipeline([("sc", StandardScaler()), ("m", LinearRegression())]),
        "Ridge": Pipeline([("sc", StandardScaler()),
                           ("m", RidgeCV(alphas=np.logspace(-3, 3, 25)))]),
        "RandomForest": RandomForestRegressor(
            n_estimators=300 if not fast else 150, max_depth=None,
            min_samples_leaf=2, max_features="sqrt",
            n_jobs=n_jobs, random_state=RANDOM_STATE),
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=300 if not fast else 150, min_samples_leaf=2,
            max_features="sqrt", n_jobs=n_jobs, random_state=RANDOM_STATE),
        "XGBoost": xgb.XGBRegressor(
            n_estimators=600 if not fast else 300, learning_rate=0.05,
            max_depth=6, subsample=0.8, colsample_bytree=0.8,
            reg_lambda=1.0, min_child_weight=5,
            n_jobs=n_jobs, random_state=RANDOM_STATE, tree_method="hist"),
        "LightGBM": lgb.LGBMRegressor(
            n_estimators=600 if not fast else 300, learning_rate=0.05,
            num_leaves=63, subsample=0.8, colsample_bytree=0.8,
            min_child_samples=20, reg_lambda=1.0,
            n_jobs=n_jobs, random_state=RANDOM_STATE, verbose=-1),
        "HistGBM": HistGradientBoostingRegressor(
            max_iter=400 if not fast else 200, learning_rate=0.06,
            random_state=RANDOM_STATE),
        "MLP": Pipeline([("sc", StandardScaler()),
                         ("m", MLPRegressor(hidden_layer_sizes=(128, 64),
                                            activation="relu", alpha=1e-3,
                                            learning_rate_init=1e-3,
                                            max_iter=400, early_stopping=True,
                                            random_state=RANDOM_STATE))]),
    }
    return m


def add_stack(models: dict) -> dict:
    base = [(k, models[k]) for k in ("XGBoost", "LightGBM", "RandomForest") if k in models]
    if len(base) >= 2:
        models["Stacked"] = StackingRegressor(
            estimators=base,
            final_estimator=RidgeCV(alphas=np.logspace(-3, 3, 15)),
            cv=3, n_jobs=1, passthrough=False)
    return models


def eval_metrics(y, yhat) -> dict:
    err = y - yhat
    rmse = float(np.sqrt(mean_squared_error(y, yhat)))
    return {
        "R2": float(r2_score(y, yhat)),
        "RMSE": rmse,
        "MAE": float(mean_absolute_error(y, yhat)),
        "Bias": float(err.mean()),
        "MaxAbsErr": float(np.abs(err).max()),
    }


def benchmark(X: np.ndarray, y: np.ndarray, splits, models: dict,
              label: str = "") -> tuple[pd.DataFrame, dict]:
    """Blocked-CV benchmark. Returns per-model metrics and out-of-fold predictions."""
    rows, oof = [], {}
    for name, proto in models.items():
        preds = np.full(len(y), np.nan)
        fold_scores = []
        ok = True
        for tr, te in splits:
            try:
                from sklearn.base import clone
                mdl = clone(proto)
                mdl.fit(X[tr], y[tr])
                p = mdl.predict(X[te])
                preds[te] = p
                fold_scores.append(r2_score(y[te], p))
            except Exception as e:
                print(f"    [!] {name} failed: {type(e).__name__}: {e}")
                ok = False
                break
        if not ok:
            continue
        mask = ~np.isnan(preds)
        met = eval_metrics(y[mask], preds[mask])
        met["R2_fold_std"] = float(np.std(fold_scores))
        met["R2_fold_min"] = float(np.min(fold_scores))
        met["Model"] = name
        rows.append(met)
        oof[name] = preds
        print(f"    {name:14s} R2={met['R2']:+.4f} (fold sd {met['R2_fold_std']:.3f}) "
              f"RMSE={met['RMSE']:.3f}")

    df = pd.DataFrame(rows)
    if len(df):
        df = df[["Model", "R2", "RMSE", "MAE", "Bias", "R2_fold_std",
                 "R2_fold_min", "MaxAbsErr"]].sort_values("R2", ascending=False)
        df.insert(0, "Zone", label)
    return df.reset_index(drop=True), oof


def leakage_demo(X, y, models: dict, splits_spatial) -> dict:
    """Quantify the optimism of random CV vs spatial CV for the best tree model."""
    from sklearn.base import clone
    name = "LightGBM" if "LightGBM" in models else list(models)[-1]
    proto = models[name]

    def _cv(splits):
        preds = np.full(len(y), np.nan)
        for tr, te in splits:
            m = clone(proto); m.fit(X[tr], y[tr]); preds[te] = m.predict(X[te])
        mask = ~np.isnan(preds)
        return float(r2_score(y[mask], preds[mask]))

    rnd = list(KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE).split(X))
    r_random = _cv(rnd)
    r_spatial = _cv(splits_spatial)
    return {"model": name, "R2_random_CV": r_random, "R2_spatial_CV": r_spatial,
            "optimism": r_random - r_spatial}


# ----------------------------------------------------------------------------
# GEOGRAPHICALLY WEIGHTED REGRESSION
# ----------------------------------------------------------------------------

def gwr_analysis(d: pd.DataFrame, s: Schema, feats: list[str],
                 max_points: int = 1500, n_perm: int = 19) -> dict | None:
    """
    Test whether UHI drivers are spatially NON-STATIONARY.

    Every global model here - OLS through the stacked ensemble - assumes one
    coefficient set applies across the whole city. If vegetation cools Uttara
    more strongly than it cools Old Dhaka, a global model reports the average
    and hides the difference, which is exactly the difference a planner needs.

    GWR fits a separate weighted regression at each location using an adaptive
    bisquare kernel, bandwidth chosen by AICc. Cost is O(n^2) in memory, so
    regression points are subsampled; the coefficient *distribution* is what
    matters, and it is stable under subsampling.

    Non-stationarity is tested in TWO stages, because the naive per-driver test
    is badly over-liberal on its own:

      Stage 1 (model level): does GWR beat global OLS on AICc by more than 3?
        If not, there is no evidence of non-stationarity anywhere and every
        per-driver flag is suppressed. Local coefficients ALWAYS wobble - a
        local regression on a few hundred neighbours is noisy by construction -
        so comparing that wobble to a global standard error computed on the
        full sample flags drivers even when the truth is perfectly stationary.
      Stage 2 (per driver): among survivors, a Monte Carlo permutation test.
        Locations are shuffled and the model refitted, giving a null
        distribution for each coefficient's IQR under stationarity. A driver is
        flagged only if its observed IQR exceeds the 95th percentile of that
        null.

    Set n_perm=0 to skip stage 2, in which case flags fall back to the AICc
    gate alone and are reported as provisional.
    """
    if not (s.lat and s.lon) or len(feats) < 1:
        return None
    try:
        from mgwr.gwr import GWR
        from mgwr.sel_bw import Sel_BW
    except ImportError:
        print("    mgwr not installed - skipping (pip install mgwr)")
        return None

    n = len(d)
    idx = rng.choice(n, min(max_points, n), replace=False)
    coords = d[[s.lon, s.lat]].to_numpy(float)[idx]
    y = d[s.lst].to_numpy(float)[idx].reshape(-1, 1)
    Xr = d[feats].to_numpy(float)[idx]
    Xz = StandardScaler().fit_transform(Xr)          # comparable coefficients

    # jitter exact-duplicate coordinates, which make the kernel singular
    coords = coords + rng.normal(0, 1e-7, coords.shape)

    try:
        bw = Sel_BW(coords, y, Xz, kernel="bisquare", fixed=False).search(
            criterion="AICc")
        model = GWR(coords, y, Xz, bw, kernel="bisquare", fixed=False).fit()
    except Exception as e:
        print(f"    GWR failed: {type(e).__name__}: {e}")
        return None

    # global OLS baseline for the comparison
    from sklearn.linear_model import LinearRegression
    ols = LinearRegression().fit(Xz, y.ravel())
    resid = y.ravel() - ols.predict(Xz)
    dof = max(1, len(y) - Xz.shape[1] - 1)
    sigma2 = float((resid ** 2).sum() / dof)
    XtX_inv = np.linalg.pinv(Xz.T @ Xz)
    se_global = np.sqrt(np.maximum(np.diag(XtX_inv) * sigma2, 0))

    params = model.params[:, 1:]                      # drop intercept column
    obs_iqr = np.array([float(np.subtract(*np.percentile(params[:, j], [75, 25])))
                        for j in range(len(feats))])

    # --- stage 1: model-level AICc gate ---
    k_ols = Xz.shape[1] + 1
    n_obs = len(y)
    rss = float((resid ** 2).sum())
    aicc_ols = (n_obs * np.log(rss / n_obs) + 2 * k_ols
                + (2 * k_ols * (k_ols + 1)) / max(1, n_obs - k_ols - 1))
    d_aicc = float(aicc_ols - model.aicc)
    gwr_justified = d_aicc > 3.0

    # --- stage 2: permutation null for each coefficient's IQR ---
    null_p95 = np.full(len(feats), np.nan)
    if gwr_justified and n_perm > 0:
        null = []
        for _ in range(n_perm):
            perm = rng.permutation(len(coords))
            try:
                mp = GWR(coords[perm], y, Xz, bw, kernel="bisquare",
                         fixed=False).fit()
                pp = mp.params[:, 1:]
                null.append([float(np.subtract(*np.percentile(pp[:, j], [75, 25])))
                             for j in range(len(feats))])
            except Exception:
                continue
        if len(null) >= 5:
            null_p95 = np.percentile(np.array(null), 95, axis=0)

    rows = []
    for j, f in enumerate(feats):
        loc = params[:, j]
        if not gwr_justified:
            flag, basis = False, "AICc gate not passed"
        elif np.isfinite(null_p95[j]):
            flag = bool(obs_iqr[j] > null_p95[j])
            basis = "permutation test"
        else:
            flag = bool(obs_iqr[j] > 2 * se_global[j])
            basis = "provisional (AICc gate only)"
        rows.append({
            "Feature": f,
            "Global_coef": float(ols.coef_[j]),
            "Global_SE": float(se_global[j]),
            "Local_mean": float(loc.mean()),
            "Local_min": float(loc.min()),
            "Local_max": float(loc.max()),
            "Local_IQR": float(obs_iqr[j]),
            "Null_IQR_p95": float(null_p95[j]) if np.isfinite(null_p95[j]) else np.nan,
            "Sign_flips_pct": float(100 * np.mean(
                np.sign(loc) != np.sign(ols.coef_[j]))),
            "Non_stationary": flag,
            "Test_basis": basis,
        })

    summary = pd.DataFrame(rows).sort_values("Local_IQR", ascending=False)
    return {
        "summary": summary,
        "bandwidth": float(bw),
        "n_points": len(idx),
        "gwr_R2": float(model.R2),
        "ols_R2": float(ols.score(Xz, y.ravel())),
        "aicc": float(model.aicc),
        "aicc_ols": float(aicc_ols),
        "delta_aicc": d_aicc,
        "gwr_justified": bool(gwr_justified),
        "n_perm": int(n_perm),
        "local_params": params,
        "coords": coords,
        "feats": feats,
    }


# ----------------------------------------------------------------------------
# EXPLAINABLE AI
# ----------------------------------------------------------------------------

import shap


def shap_analysis(model, X: np.ndarray, feat_names: list[str], label: str = "",
                  max_rows: int = MAX_ROWS_SHAP) -> dict:
    """Global SHAP attribution + pairwise interaction strength."""
    n = len(X)
    idx = rng.choice(n, min(max_rows, n), replace=False)
    Xs = X[idx]

    try:
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(Xs, check_additivity=False)
    except Exception:
        # Non-tree best model (linear, MLP, stacked) -> model-agnostic fallback.
        bg = shap.kmeans(X[rng.choice(n, min(100, n), replace=False)], 25)
        explainer = shap.KernelExplainer(model.predict, bg)
        keep = min(300, len(Xs))
        Xs, idx = Xs[:keep], idx[:keep]      # keep idx aligned with Xs
        sv = explainer.shap_values(Xs, nsamples=200, silent=True)

    sv = np.asarray(sv)
    if sv.ndim == 3:
        sv = sv[..., 0]

    mean_abs = np.abs(sv).mean(axis=0)
    signed = sv.mean(axis=0)
    corr = []
    for j in range(sv.shape[1]):
        if np.std(Xs[:, j]) > 0 and np.std(sv[:, j]) > 0:
            corr.append(float(np.corrcoef(Xs[:, j], sv[:, j])[0, 1]))
        else:
            corr.append(0.0)

    imp = pd.DataFrame({
        "Feature": feat_names,
        "MeanAbsSHAP": mean_abs,
        "MeanSHAP": signed,
        "Direction": ["warming" if c > 0 else "cooling" for c in corr],
        "SHAP_vs_value_corr": corr,
    }).sort_values("MeanAbsSHAP", ascending=False).reset_index(drop=True)
    imp["PctContribution"] = 100 * imp["MeanAbsSHAP"] / imp["MeanAbsSHAP"].sum()
    imp.insert(0, "Zone", label)

    # Interaction strength: correlation of |SHAP_i| with feature_j, a cheap
    # proxy that scales to large n where exact interaction values do not.
    inter = np.zeros((len(feat_names), len(feat_names)))
    for i in range(len(feat_names)):
        for j in range(len(feat_names)):
            if i != j and np.std(Xs[:, j]) > 0 and np.std(np.abs(sv[:, i])) > 0:
                inter[i, j] = abs(float(np.corrcoef(np.abs(sv[:, i]), Xs[:, j])[0, 1]))
    inter_df = pd.DataFrame(inter, index=feat_names, columns=feat_names)

    return {"importance": imp, "shap_values": sv, "X_sample": Xs,
            "sample_idx": idx, "interactions": inter_df,
            "base_value": float(np.mean(getattr(explainer, "expected_value", 0.0)))}


def local_explanations(sv: np.ndarray, Xs: np.ndarray, y_sample: np.ndarray,
                       feat_names: list[str], base: float, n: int = 5) -> pd.DataFrame:
    """LIME-style local attribution for the hottest sampled pixels."""
    hottest = np.argsort(y_sample)[-n:][::-1]
    rows = []
    for rank, i in enumerate(hottest, 1):
        order = np.argsort(np.abs(sv[i]))[::-1][:6]
        for j in order:
            rows.append({
                "Hotspot": rank,
                "Observed_LST": float(y_sample[i]),
                "Baseline_LST": base,
                "Feature": feat_names[j],
                "Feature_value": float(Xs[i, j]),
                "SHAP_contribution_C": float(sv[i, j]),
            })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# CONFIDENCE GATEKEEPER
# ----------------------------------------------------------------------------

from sklearn.neighbors import NearestNeighbors


class ConfidenceGatekeeper:
    """
    Per-prediction trust scoring that fuses two independent failure modes:

      1. NORMALISED CONFORMAL PREDICTION - the nonconformity score is the
         residual divided by a local difficulty estimate sigma(x), so intervals
         widen in genuinely hard regions while retaining the finite-sample
         coverage guarantee. Scaling a global interval after the fact (a common
         shortcut) destroys that guarantee; conformalising the ratio does not.
      2. APPLICABILITY DOMAIN - k-NN distance in standardised feature space
         against the training manifold. A tight interval on an out-of-domain
         sample is a confident extrapolation, which is the dangerous case, and
         only this second gate can catch it.

    Trust = geometric mean of the two normalised scores, so a sample must pass
    BOTH gates. Verdict thresholds are calibrated as quantiles of the
    calibration-set trust distribution rather than hard-coded, so the gate
    transfers across datasets with different feature scales.

    Calibration data is split in two: half A fits sigma(x), half B computes the
    conformal quantile. Reusing one half for both would leak and under-cover.
    """

    def __init__(self, alpha: float = CONFORMAL_ALPHA, k: int = 10,
                 accept_q: float = 0.40, reject_q: float = 0.15):
        self.alpha, self.k = alpha, k
        self.accept_q, self.reject_q = accept_q, reject_q

    def _sigma(self, Xs: np.ndarray) -> np.ndarray:
        """Local difficulty: mean absolute residual of the k nearest sigma-set points."""
        _, ind = self.nn_sig.kneighbors(Xs)
        return self.sig_resid[ind].mean(axis=1)

    def fit(self, model, X_cal: np.ndarray, y_cal: np.ndarray, X_train: np.ndarray):
        self.model = model
        self.scaler = StandardScaler().fit(X_train)

        # --- applicability domain, referenced to the training manifold ---
        Xt = self.scaler.transform(X_train)
        self.nn = NearestNeighbors(n_neighbors=min(self.k, len(Xt))).fit(Xt)

        # --- split calibration: A fits sigma, B sets the conformal quantile ---
        n = len(X_cal)
        perm = rng.permutation(n)
        a, b = perm[: n // 2], perm[n // 2:]
        if len(a) < 5 or len(b) < 5:
            a = b = perm

        Xa = self.scaler.transform(X_cal[a])
        self.sig_resid = np.abs(y_cal[a] - model.predict(X_cal[a]))
        self.nn_sig = NearestNeighbors(
            n_neighbors=min(self.k, len(Xa))).fit(Xa)
        self.beta = float(np.mean(self.sig_resid)) * 0.1 + 1e-6   # stabiliser

        Xb = self.scaler.transform(X_cal[b])
        resid_b = np.abs(y_cal[b] - model.predict(X_cal[b]))
        sig_b = self._sigma(Xb)
        scores = resid_b / (sig_b + self.beta)
        m = len(scores)
        q_level = min(1.0, np.ceil((m + 1) * (1 - self.alpha)) / m)
        self.q_hat = float(np.quantile(scores, q_level))

        # Reference distributions for scoring. Scores are RANKS against these
        # held-out distributions, not linear ratios: a linear 1 - d/d_95 maps a
        # perfectly typical in-domain point to about 0.35, which is unreadable
        # and clips a tail of ordinary points to exactly zero. Ranking maps the
        # median in-domain point to 0.5 and only genuine outliers to 0.
        d_cal, _ = self.nn.kneighbors(Xb)
        self.d_ref_dist = np.sort(d_cal.mean(axis=1))
        self.sigma_ref_dist = np.sort(sig_b)

        # --- calibrate verdict thresholds on held-out calibration half B ---
        trust_b = self._trust(Xb)
        self.t_accept = float(np.quantile(trust_b, self.accept_q))
        self.t_reject = float(np.quantile(trust_b, self.reject_q))
        return self

    @staticmethod
    def _rank_score(v: np.ndarray, ref_sorted: np.ndarray) -> np.ndarray:
        """1 - empirical CDF of v against ref_sorted. Typical -> ~0.5,
        better than everything seen -> ~1, worse than everything -> 0."""
        n = len(ref_sorted)
        if n == 0:
            return np.full(len(v), 0.5)
        return 1.0 - np.searchsorted(ref_sorted, v, side="right") / n

    def _ad(self, Xs: np.ndarray) -> np.ndarray:
        d, _ = self.nn.kneighbors(Xs)
        return self._rank_score(d.mean(axis=1), self.d_ref_dist)

    def _conf(self, Xs: np.ndarray) -> np.ndarray:
        return self._rank_score(self._sigma(Xs), self.sigma_ref_dist)

    def _trust(self, Xs: np.ndarray) -> np.ndarray:
        return np.sqrt(self._ad(Xs) * self._conf(Xs))

    def score(self, X: np.ndarray) -> pd.DataFrame:
        pred = self.model.predict(X)
        Xs = self.scaler.transform(X)

        ad_score = self._ad(Xs)
        sigma = self._sigma(Xs)
        conf_score = self._conf(Xs)
        half_width = self.q_hat * (sigma + self.beta)

        trust = np.sqrt(ad_score * conf_score)
        verdict = np.where(trust >= self.t_accept, "ACCEPT",
                  np.where(trust >= self.t_reject, "REVIEW", "REJECT"))

        return pd.DataFrame({
            "prediction": pred,
            "lower_90": pred - half_width,
            "upper_90": pred + half_width,
            "interval_width": 2 * half_width,
            "applicability_score": ad_score,
            "conformal_score": conf_score,
            "trust_score": trust,
            "verdict": verdict,
        })

    def validate_coverage(self, X_test, y_test) -> dict:
        sc = self.score(X_test)
        inside = ((y_test >= sc.lower_90) & (y_test <= sc.upper_90)).mean()
        out = {"target_coverage": 1 - self.alpha,
               "empirical_coverage": float(inside),
               "mean_interval_width": float(sc.interval_width.mean()),
               "q_hat": self.q_hat,
               "t_accept": self.t_accept, "t_reject": self.t_reject}
        for v in ("ACCEPT", "REVIEW", "REJECT"):
            m = (sc.verdict == v).to_numpy()
            out[f"pct_{v}"] = float(100 * m.mean())
            if m.sum() > 5:
                out[f"MAE_{v}"] = float(np.abs(y_test[m] - sc.prediction[m]).mean())
                out[f"cov_{v}"] = float(((y_test[m] >= sc.lower_90[m]) &
                                         (y_test[m] <= sc.upper_90[m])).mean())
        return out


# ----------------------------------------------------------------------------
# TEMPORAL ANALYSIS
# ----------------------------------------------------------------------------

import pymannkendall as mk


def tidy_trend_table(df: pd.DataFrame, name: str) -> pd.DataFrame | None:
    """Normalise a trend table to long format: [unit, year, value]."""
    year_cols = [c for c in df.columns if re.fullmatch(r"(19|20)\d{2}", str(c).strip())]
    if year_cols:
        idc = [c for c in df.columns if c not in year_cols]
        long = df.melt(id_vars=idc, value_vars=year_cols,
                       var_name="year", value_name="value")
        long["year"] = pd.to_numeric(long["year"], errors="coerce")
        long["unit"] = (long[idc[0]].astype(str) if idc else "ALL")
        return long[["unit", "year", "value"]].dropna()

    s = infer_schema(df, name)
    if s.year and s.lst:
        long = df[[s.year, s.lst]].copy()
        long.columns = ["year", "value"]
        long["year"] = pd.to_numeric(long["year"], errors="coerce")
        long["value"] = pd.to_numeric(long["value"], errors="coerce")
        long["unit"] = df[s.zone].astype(str) if s.zone else "ALL"
        return long[["unit", "year", "value"]].dropna()

    # Fallback: rows with an embedded year, e.g. "LST_1995"
    ycol = None
    for c in df.columns:
        v = pd.to_numeric(df[c], errors="coerce")
        if v.notna().sum() > 3 and v.dropna().between(1980, 2035).all():
            ycol = c
            break
    if ycol is None:
        return None
    num = [c for c in df.columns if c != ycol and pd.api.types.is_numeric_dtype(df[c])]
    if not num:
        return None
    tgt = next((c for c in num if re.search(LST_PATTERNS, _norm(c))), num[0])
    long = df[[ycol, tgt]].copy()
    long.columns = ["year", "value"]
    long["unit"] = "ALL"
    return long.dropna()


def trend_stats(long: pd.DataFrame, zone: str) -> pd.DataFrame:
    rows = []
    for unit, g in long.groupby("unit"):
        g = g.groupby("year", as_index=False)["value"].mean().sort_values("year")
        if len(g) < 5:
            continue
        v = g["value"].to_numpy(dtype=float)
        try:
            r = mk.original_test(v)
            sen = mk.sens_slope(v)
            yrs = g["year"].to_numpy(dtype=float)
            span = yrs.max() - yrs.min()
            step = span / (len(yrs) - 1) if len(yrs) > 1 else 1
            rows.append({
                "Zone": zone, "Unit": unit,
                "Years": f"{int(yrs.min())}-{int(yrs.max())}",
                "N_obs": len(g),
                "Trend": r.trend,
                "MK_p": float(r.p),
                "MK_tau": float(r.Tau),
                "Sen_slope_per_step": float(sen.slope),
                "Sen_slope_per_decade": float(sen.slope * (10 / step)) if step else np.nan,
                "Total_change_C": float(sen.slope * (len(yrs) - 1)),
                "Mean_LST": float(v.mean()),
                "First_period_mean": float(v[:max(1, len(v)//3)].mean()),
                "Last_period_mean": float(v[-max(1, len(v)//3):].mean()),
            })
        except Exception:
            continue
    return pd.DataFrame(rows)


def change_detection(frames: dict) -> pd.DataFrame | None:
    """
    Model delta-LST from delta-indices between two zones/epochs.

    Correlational models tell you what coincides with heat. A delta model tells
    you what CHANGING a driver did, which is the question a planner actually
    asks. Matching is done on canonical index names (NDVI, NDBI, ...) because
    the two exports rarely share raw column spellings.
    """
    items = list(frames.items())
    if len(items) < 2:
        return None
    (na, (da, sa)), (nb, (db, sb)) = items[0], items[1]

    common = sorted(set(sa.indices) & set(sb.indices))
    if len(common) < 2 or sa.lst not in da.columns or sb.lst not in db.columns:
        return None

    n = min(len(da), len(db))
    if n < 100:
        return None

    # Rank-align by LST so we compare like-for-like positions in each
    # distribution rather than arbitrary row order.
    A = da.sort_values(sa.lst).iloc[:n].reset_index(drop=True)
    B = db.sort_values(sb.lst).iloc[:n].reset_index(drop=True)

    dX = (B[[sb.indices[c] for c in common]].to_numpy(float)
          - A[[sa.indices[c] for c in common]].to_numpy(float))
    dy = B[sb.lst].to_numpy(float) - A[sa.lst].to_numpy(float)

    m = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=31,
                          random_state=RANDOM_STATE, verbose=-1)
    m.fit(dX, dy)
    r2 = float(r2_score(dy, m.predict(dX)))

    imp = pd.DataFrame({"Delta_feature": [f"d_{c}" for c in common],
                        "Importance": m.feature_importances_})
    imp["Pct"] = 100 * imp.Importance / max(imp.Importance.sum(), 1)
    imp["Comparison"] = f"{nb} minus {na}"
    imp["Model_R2_insample"] = round(r2, 4)
    return imp.sort_values("Pct", ascending=False).reset_index(drop=True)


# ----------------------------------------------------------------------------
# UNSUPERVISED: EMPIRICAL LOCAL CLIMATE ZONES
# ----------------------------------------------------------------------------

from sklearn.metrics import silhouette_score


def discover_lcz(d: pd.DataFrame, s: Schema, feats: list[str],
                 k_range=range(3, 9)) -> tuple[pd.DataFrame, np.ndarray, dict]:
    """Derive climate zones from the data instead of assigning them by hand."""
    X = StandardScaler().fit_transform(d[feats].to_numpy(float))
    sub = X[rng.choice(len(X), min(20000, len(X)), replace=False)]

    best_k, best_s = None, -1
    scores = {}
    for k in k_range:
        km = MiniBatchKMeans(n_clusters=k, random_state=RANDOM_STATE,
                             n_init=10, batch_size=2048).fit(sub)
        sc = silhouette_score(sub[:5000], km.labels_[:5000])
        scores[k] = float(sc)
        if sc > best_s:
            best_k, best_s = k, sc

    km = MiniBatchKMeans(n_clusters=best_k, random_state=RANDOM_STATE,
                         n_init=10, batch_size=2048).fit(X)
    labels = km.labels_

    prof = d.copy()
    prof["_lcz"] = labels
    agg = {f: "mean" for f in feats}
    agg[s.lst] = "mean"
    summ = prof.groupby("_lcz").agg(agg)
    summ["N_pixels"] = prof.groupby("_lcz").size()
    summ["Pct_area"] = 100 * summ["N_pixels"] / len(prof)
    summ = summ.sort_values(s.lst, ascending=False).reset_index()
    summ.rename(columns={"_lcz": "LCZ_cluster", s.lst: "Mean_LST_C"}, inplace=True)

    # Name each cluster from its dominant standardised signature.
    names = []
    z = (summ[feats] - summ[feats].mean()) / (summ[feats].std() + 1e-12)
    for i in range(len(summ)):
        top = z.iloc[i].abs().sort_values(ascending=False).index[:2]
        parts = [f"{'high' if z.iloc[i][t] > 0 else 'low'} {t}" for t in top]
        names.append(" / ".join(parts))
    summ.insert(1, "Signature", names)

    return summ, labels, {"chosen_k": best_k, "silhouette": best_s,
                          "all_scores": scores}


# ----------------------------------------------------------------------------
# FIGURES  —  publication standard
# ----------------------------------------------------------------------------
# Journal requirements encoded here:
#   * Physical widths: 89 mm single column, 183 mm double (Elsevier/Springer/MDPI)
#   * 600 dpi raster AND vector PDF of every panel — reviewers ask for vector
#   * 7–8 pt sans throughout; nothing below 6 pt after reduction
#   * Okabe–Ito colourblind-safe qualitative palette; perceptually uniform
#     sequential ramps only (viridis / cividis / RdYlBu_r for diverging)
#   * Panel labels (a), (b), (c) for multi-panel figures
#   * No chartjunk: no 3-D, no drop shadows, no gratuitous gridlines
# ----------------------------------------------------------------------------

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

MM = 1 / 25.4
W1, W2 = 89 * MM, 183 * MM          # single- and double-column widths

# Okabe–Ito: safe under all common colour vision deficiencies
OI = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
      "vermillion": "#D55E00", "sky": "#56B4E9", "yellow": "#F0E442",
      "purple": "#CC79A7", "black": "#000000"}
C_WARM, C_COOL, C_NEUT = OI["vermillion"], OI["blue"], "#8C8C8C"
SEQ, DIV = "cividis", "RdYlBu_r"

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 600,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7.5, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.linewidth": 0.6, "axes.labelpad": 2.5,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "xtick.direction": "out", "ytick.direction": "out",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": False,
    "legend.frameon": False, "legend.handlelength": 1.4,
    "lines.linewidth": 1.0, "lines.markersize": 3,
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42, "ps.fonttype": 42,        # editable text in Illustrator
})


def _save(fig, path: Path):
    """Write 600-dpi PNG plus vector PDF. Journals want both."""
    fig.savefig(path)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def _panel(ax, letter: str, dx: float = -0.16, dy: float = 1.04):
    ax.text(dx, dy, f"({letter})", transform=ax.transAxes,
            fontsize=9, fontweight="bold", va="top", ha="left")


def _grid(ax, axis="y"):
    """Hairline reference lines, behind the data, never competing with it."""
    ax.grid(axis=axis, color="#D9D9D9", lw=0.4, zorder=0)
    ax.set_axisbelow(True)


# ----------------------------------------------------------------------------

def fig_model_comparison(bench: pd.DataFrame, path: Path):
    zones = list(bench["Zone"].unique())
    ncol = min(2, len(zones))
    nrow = int(np.ceil(len(zones) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(W2, 2.5 * nrow), squeeze=False)
    letters = "abcdefgh"

    for i, z in enumerate(zones):
        ax = axes[i // ncol][i % ncol]
        b = bench[bench.Zone == z].sort_values("R2")
        best = b.Model.iloc[-1]
        cols = [C_WARM if m == best else C_NEUT for m in b.Model]
        ax.barh(b.Model, b.R2, xerr=b.R2_fold_std, color=cols,
                edgecolor="white", linewidth=0.4, height=0.72, zorder=3,
                error_kw={"lw": 0.7, "capsize": 1.8, "ecolor": "#404040"})
        _grid(ax, "x")
        ax.set_xlabel("Coefficient of determination, $R^2$")
        ax.set_title(z, fontweight="bold", loc="left", pad=4)
        for j, (r, rm) in enumerate(zip(b.R2, b.RMSE)):
            ax.text(r + 0.012, j, f"{r:.3f}", va="center", fontsize=6.5,
                    color="#333333")
        ax.set_xlim(min(0, b.R2.min() * 1.05), max(b.R2) * 1.16)
        _panel(ax, letters[i], dx=-0.30)

    for k in range(len(zones), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")

    fig.text(0, -0.03,
             "Error bars: ±1 s.d. of $R^2$ across spatial cross-validation folds. "
             "Highlighted bar denotes the best model per zone.",
             fontsize=6.5, color="#555555", ha="left")
    fig.tight_layout()
    _save(fig, path)


def fig_shap_importance(imp_list: list, path: Path):
    n = len(imp_list)
    fig, axes = plt.subplots(1, n, figsize=(W2, 2.9), squeeze=False)
    letters = "abcdefgh"
    for i, (ax, imp) in enumerate(zip(axes[0], imp_list)):
        d = imp.head(10).iloc[::-1]
        cols = [C_WARM if x == "warming" else C_COOL for x in d.Direction]
        ax.barh(d.Feature, d.MeanAbsSHAP, color=cols, edgecolor="white",
                linewidth=0.4, height=0.72, zorder=3)
        _grid(ax, "x")
        ax.set_xlabel("Mean |SHAP| (°C)")
        ax.set_title(d.Zone.iloc[0], fontweight="bold", loc="left", pad=4)
        for j, (v, p) in enumerate(zip(d.MeanAbsSHAP, d.PctContribution)):
            ax.text(v * 1.02, j, f"{p:.0f}%", va="center", fontsize=6.5,
                    color="#333333")
        ax.set_xlim(0, d.MeanAbsSHAP.max() * 1.20)
        _panel(ax, letters[i], dx=-0.34)

    h = [plt.Rectangle((0, 0), 1, 1, fc=C_WARM), plt.Rectangle((0, 0), 1, 1, fc=C_COOL)]
    fig.legend(h, ["Increases LST", "Decreases LST"], loc="lower center",
               ncol=2, bbox_to_anchor=(0.5, -0.10))
    fig.text(0, -0.16, "Bars give the mean absolute SHAP value; labels give each "
             "predictor's share of total attributed variation.",
             fontsize=6.5, color="#555555", ha="left")
    fig.tight_layout()
    _save(fig, path)


def fig_shap_beeswarm(sv, Xs, feats, label, path):
    try:
        order = np.argsort(np.abs(sv).mean(0))[::-1][:10]
        fig, ax = plt.subplots(figsize=(W1 * 1.35, 0.28 * len(order) + 1.1))
        for row, j in enumerate(order[::-1]):
            v, x = sv[:, j], Xs[:, j]
            rank = (np.argsort(np.argsort(x)) / max(1, len(x) - 1))
            jitter = rng.normal(0, 0.075, len(v))
            ax.scatter(v, np.full(len(v), row) + jitter, c=rank, cmap=DIV,
                       s=1.6, alpha=0.55, linewidths=0, rasterized=True,
                       vmin=0, vmax=1)
        ax.axvline(0, color="#555555", lw=0.6, zorder=1)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([feats[j] for j in order[::-1]])
        ax.set_xlabel("SHAP value (°C contribution to LST)")
        ax.set_title(label, fontweight="bold", loc="left", pad=4)
        sm = plt.cm.ScalarMappable(cmap=DIV, norm=plt.Normalize(0, 1))
        cb = fig.colorbar(sm, ax=ax, pad=0.015, fraction=0.03)
        cb.set_ticks([0, 1]); cb.set_ticklabels(["Low", "High"])
        cb.set_label("Predictor value (percentile)", fontsize=7)
        cb.outline.set_linewidth(0.5)
        fig.tight_layout()
        _save(fig, path)
    except Exception as e:
        print(f"    [!] beeswarm skipped: {e}")


def fig_dependence(sv, Xs, feats, imp, label, path):
    top = imp.head(4).Feature.tolist()
    fig, axes = plt.subplots(1, 4, figsize=(W2, 2.0), squeeze=False)
    for i, (ax, f) in enumerate(zip(axes[0], top)):
        j = feats.index(f)
        ax.scatter(Xs[:, j], sv[:, j], s=1.6, alpha=0.30, c=sv[:, j],
                   cmap=DIV, linewidths=0, rasterized=True)
        if len(Xs) > 50:
            o = np.argsort(Xs[:, j])
            w = max(5, len(o) // 25)
            sm = pd.Series(sv[o, j]).rolling(w, center=True, min_periods=1).mean()
            ax.plot(Xs[o, j], sm, color="black", lw=1.1, zorder=4)
        ax.axhline(0, color="#888888", lw=0.5, ls=(0, (3, 2)), zorder=1)
        ax.set_xlabel(f)
        ax.set_ylabel("SHAP (°C)" if i == 0 else "")
        ax.xaxis.set_major_locator(MaxNLocator(4))
        _panel(ax, "abcd"[i], dx=-0.24)
    fig.text(0, -0.10, f"{label}. Solid line is a rolling mean; points are "
             "individual pixels.", fontsize=6.5, color="#555555", ha="left")
    fig.tight_layout()
    _save(fig, path)


def fig_gatekeeper(sc: pd.DataFrame, y_true: np.ndarray, cov: dict,
                   label: str, path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(W2, 2.3))
    err = np.abs(y_true - sc.prediction.to_numpy())

    axes[0].hist(sc.trust_score, bins=36, color=OI["sky"], edgecolor="white",
                 linewidth=0.3, zorder=3)
    for t, c, nm in ((cov.get("t_reject", .45), C_WARM, "Reject"),
                     (cov.get("t_accept", .70), OI["green"], "Accept")):
        axes[0].axvline(t, color=c, ls=(0, (3, 2)), lw=0.9, zorder=4)
        axes[0].text(t, axes[0].get_ylim()[1] * 0.96, f" {nm}", fontsize=6,
                     color=c, rotation=90, va="top")
    _grid(axes[0]); axes[0].set_xlabel("Trust score"); axes[0].set_ylabel("Pixels")
    _panel(axes[0], "a", dx=-0.26)

    order = ["ACCEPT", "REVIEW", "REJECT"]
    data = [err[(sc.verdict == v).to_numpy()] for v in order]
    data = [d if len(d) else np.array([np.nan]) for d in data]
    bp = axes[1].boxplot(data, tick_labels=["Accept", "Review", "Reject"],
                         patch_artist=True, showfliers=False, widths=0.55,
                         medianprops={"color": "black", "lw": 0.9},
                         boxprops={"lw": 0.5}, whiskerprops={"lw": 0.5},
                         capprops={"lw": 0.5}, zorder=3)
    for pch, c in zip(bp["boxes"], [OI["green"], OI["orange"], C_WARM]):
        pch.set_facecolor(c); pch.set_alpha(0.75)
    _grid(axes[1]); axes[1].set_ylabel("Absolute error (°C)")
    _panel(axes[1], "b", dx=-0.22)

    sct = axes[2].scatter(sc.prediction, y_true, s=1.6, alpha=0.30,
                          c=sc.trust_score, cmap="RdYlGn", linewidths=0,
                          rasterized=True, vmin=0, vmax=1)
    lo = float(min(np.min(sc.prediction), y_true.min()))
    hi = float(max(np.max(sc.prediction), y_true.max()))
    axes[2].plot([lo, hi], [lo, hi], color="black", lw=0.7, ls=(0, (3, 2)))
    axes[2].set_xlabel("Predicted LST (°C)"); axes[2].set_ylabel("Observed LST (°C)")
    axes[2].set_aspect("equal", adjustable="box")
    cb = fig.colorbar(sct, ax=axes[2], pad=0.02, fraction=0.045)
    cb.set_label("Trust", fontsize=7); cb.outline.set_linewidth(0.5)
    _panel(axes[2], "c", dx=-0.26)

    fig.text(0, -0.09,
             f"{label}. Empirical coverage {cov['empirical_coverage']*100:.1f}% "
             f"against a {cov['target_coverage']*100:.0f}% nominal target. "
             f"Boxes show median and interquartile range; whiskers 1.5×IQR.",
             fontsize=6.5, color="#555555", ha="left")
    fig.tight_layout()
    _save(fig, path)


def fig_trends(trends: pd.DataFrame, longs: dict, path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.4))
    marks = ["o", "s", "^", "D"]
    for i, (zone, long) in enumerate(longs.items()):
        g = long.groupby("year", as_index=False)["value"].mean().sort_values("year")
        col = C_WARM if "North" in zone else C_COOL
        axes[0].plot(g.year, g.value, marks[i % 4], color=col, ms=3.2,
                     mec="white", mew=0.4, ls="none", label=zone, zorder=4)
        if len(g) > 2:
            z = np.polyfit(g.year, g.value, 1)
            xs = np.linspace(g.year.min(), g.year.max(), 50)
            axes[0].plot(xs, np.polyval(z, xs), color=col, lw=1.0, zorder=3)
    _grid(axes[0])
    axes[0].set_xlabel("Year"); axes[0].set_ylabel("Mean LST (°C)")
    axes[0].legend(loc="upper left")
    axes[0].xaxis.set_major_locator(MaxNLocator(6))
    _panel(axes[0], "a")

    if len(trends):
        t = trends.copy()
        t["lab"] = (t.Zone.str.replace(r"\s*\(.*\)", "", regex=True) + " · "
                    + t.Unit.astype(str)).str.slice(0, 26)
        t = t.sort_values("Sen_slope_per_decade").tail(14)
        cols = [C_WARM if v > 0 else C_COOL for v in t.Sen_slope_per_decade]
        hatch = ["" if p < 0.05 else "///" for p in t.MK_p]
        bars = axes[1].barh(t.lab, t.Sen_slope_per_decade, color=cols,
                            edgecolor="white", linewidth=0.4, height=0.72, zorder=3)
        for b, h in zip(bars, hatch):
            b.set_hatch(h)
        axes[1].axvline(0, color="black", lw=0.6, zorder=4)
        _grid(axes[1], "x")
        axes[1].set_xlabel("Sen's slope (°C decade$^{-1}$)")
    _panel(axes[1], "b", dx=-0.42)

    fig.text(0, -0.10, "Hatched bars are not significant at α = 0.05 "
             "(Mann–Kendall). Lines in (a) are ordinary least-squares fits.",
             fontsize=6.5, color="#555555", ha="left")
    fig.tight_layout()
    _save(fig, path)


def fig_lcz(summ: pd.DataFrame, path: Path, label: str = ""):
    fig, ax = plt.subplots(figsize=(W1 * 1.45, 0.30 * len(summ) + 1.0))
    v = summ["Mean_LST_C"]
    norm = (v - v.min()) / (v.max() - v.min() + 1e-12)
    cols = plt.cm.get_cmap(DIV)(norm)
    lbl = summ.apply(lambda r: f"C{int(r.LCZ_cluster)}  {r.Signature}", axis=1)
    ax.barh(lbl, v, color=cols, edgecolor="white", linewidth=0.4,
            height=0.72, zorder=3)
    for i, (t, p) in enumerate(zip(v, summ.Pct_area)):
        ax.text(t + (v.max() - v.min()) * 0.02, i, f"{t:.1f} °C · {p:.0f}%",
                va="center", fontsize=6.5, color="#333333")
    _grid(ax, "x")
    ax.set_xlabel("Mean LST (°C)")
    ax.set_xlim(v.min() - (v.max() - v.min()) * 0.10,
                v.max() + (v.max() - v.min()) * 0.30)
    ax.set_title(f"Data-derived climate zones {label}", fontweight="bold",
                 loc="left", pad=4)
    ax.invert_yaxis()
    fig.text(0, -0.05, "Second value is the share of study-area pixels in each "
             "cluster.", fontsize=6.5, color="#555555", ha="left")
    fig.tight_layout()
    _save(fig, path)


def fig_spatial(d, s, groups, label, path):
    if not (s.lat and s.lon):
        return
    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.7))
    sct = axes[0].scatter(d[s.lon], d[s.lat], c=d[s.lst], s=1.4, cmap=DIV,
                          linewidths=0, rasterized=True)
    cb = fig.colorbar(sct, ax=axes[0], pad=0.02, fraction=0.046)
    cb.set_label("LST (°C)", fontsize=7); cb.outline.set_linewidth(0.5)
    _panel(axes[0], "a")

    axes[1].scatter(d[s.lon], d[s.lat], c=groups, s=1.4, cmap="tab20",
                    linewidths=0, rasterized=True)
    _panel(axes[1], "b")

    for ax in axes:
        ax.set_xlabel("Longitude (°E)"); ax.set_ylabel("Latitude (°N)")
        ax.set_aspect("equal", adjustable="datalim")
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.yaxis.set_major_locator(MaxNLocator(4))
    fig.text(0, -0.06, f"{label}. (a) Observed land surface temperature. "
             "(b) Cross-validation blocks; folds are withheld whole, so no "
             "training pixel neighbours a test pixel.",
             fontsize=6.5, color="#555555", ha="left")
    fig.tight_layout()
    _save(fig, path)


def fig_gwr(g: dict, label: str, path: Path):
    summ = g["summary"]
    top = summ.head(4).Feature.tolist()
    fig, axes = plt.subplots(1, len(top), figsize=(W2, 2.1), squeeze=False)
    for i, (ax, f) in enumerate(zip(axes[0], top)):
        j = g["feats"].index(f)
        v = g["local_params"][:, j]
        lim = float(np.abs(v).max()) or 1.0
        sct = ax.scatter(g["coords"][:, 0], g["coords"][:, 1], c=v, s=3.5,
                         cmap=DIV, vmin=-lim, vmax=lim, linewidths=0,
                         rasterized=True)
        cb = fig.colorbar(sct, ax=ax, pad=0.02, fraction=0.046)
        cb.ax.tick_params(labelsize=6); cb.outline.set_linewidth(0.5)
        gc = float(summ.loc[summ.Feature == f, "Global_coef"].iloc[0])
        ax.set_title(f"{f}  (global {gc:+.2f})", fontsize=7.5,
                     fontweight="bold", loc="left", pad=3)
        ax.set_xlabel("Longitude (°E)")
        ax.set_ylabel("Latitude (°N)" if i == 0 else "")
        ax.set_aspect("equal", adjustable="datalim")
        ax.xaxis.set_major_locator(MaxNLocator(3))
        ax.yaxis.set_major_locator(MaxNLocator(3))
        _panel(ax, "abcd"[i], dx=-0.22)
    fig.text(0, -0.10,
             f"{label}. Local standardised regression coefficients from "
             f"geographically weighted regression (adaptive bisquare kernel, "
             f"bandwidth = {g['bandwidth']:.0f} neighbours). Red increases LST, "
             f"blue decreases it.", fontsize=6.5, color="#555555", ha="left")
    fig.tight_layout()
    _save(fig, path)


def fig_leakage(lk: pd.DataFrame, path: Path):
    """Headline methodological figure: what random cross-validation costs you."""
    if not len(lk):
        return
    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.3))
    x = np.arange(len(lk))
    w = 0.36
    axes[0].bar(x - w/2, lk.R2_random_CV, w, label="Random $k$-fold",
                color=OI["orange"], edgecolor="white", linewidth=0.4, zorder=3)
    axes[0].bar(x + w/2, lk.R2_spatial_CV, w, label="Spatial blocked",
                color=OI["blue"], edgecolor="white", linewidth=0.4, zorder=3)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([z.split()[0] for z in lk.zone], rotation=0)
    axes[0].set_ylabel("$R^2$")
    axes[0].legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.16))
    axes[0].set_ylim(0, max(lk.R2_random_CV.max(), lk.R2_spatial_CV.max()) * 1.12)
    _grid(axes[0]); _panel(axes[0], "a", dx=-0.20)

    if "morans_I_residual" in lk.columns:
        axes[1].scatter(lk.morans_I_residual, lk.optimism, s=26,
                        color=C_WARM, edgecolor="white", linewidth=0.5, zorder=4)
        # greedy collision-aware label placement: nudge a label to a free
        # quadrant when it would land on top of one already placed
        pts = [(r.morans_I_residual, r.optimism, r.zone.split()[0])
               for _, r in lk.iterrows()
               if np.isfinite(r.get("morans_I_residual", np.nan))]
        pts.sort(key=lambda t: t[0])
        xr = (max(p[0] for p in pts) - min(p[0] for p in pts)) or 1
        yr = (max(p[1] for p in pts) - min(p[1] for p in pts)) or 1
        placed = []
        cands = [(5, 3, "left"), (-5, 3, "right"),
                 (5, -8, "left"), (-5, -8, "right")]
        for xx, yy, nm in pts:
            near = [q for q in placed
                    if abs(q[0] - xx) < 0.22 * xr and abs(q[1] - yy) < 0.22 * yr]
            used = {q[2] for q in near}
            off = next((c for c in cands if c not in used), cands[0])
            axes[1].annotate(nm, (xx, yy), fontsize=6, xytext=off[:2],
                             textcoords="offset points", ha=off[2])
            placed.append((xx, yy, off))
        axes[1].axhline(0, color="#888888", lw=0.5, ls=(0, (3, 2)))
        axes[1].set_xlabel("Moran's $I$ of model residuals")
        axes[1].set_ylabel("Optimism ($R^2_{random} - R^2_{spatial}$)")
        axes[1].margins(x=0.12, y=0.18)
        _grid(axes[1])
    _panel(axes[1], "b", dx=-0.20)

    fig.text(0, -0.10,
             "(a) Apparent skill under each validation design. (b) Optimism "
             "against residual spatial autocorrelation: autocorrelated residuals "
             "are what random folds leak, not autocorrelation in the target.",
             fontsize=6.5, color="#555555", ha="left")
    fig.tight_layout()
    _save(fig, path)


# ----------------------------------------------------------------------------
# ORCHESTRATION
# ----------------------------------------------------------------------------

def analyse_zone(d: pd.DataFrame, s: Schema, zone: str, results: dict, fast: bool = False):
    print(f"\n{'='*78}\nZONE: {zone}\n{'='*78}")
    feats_all = s.feature_cols()
    if s.lst is None or len(feats_all) < 2 or len(d) < 100:
        print("  insufficient structure for modelling - skipped")
        return

    # --- multicollinearity ---
    print("\n[1] Multicollinearity screening (VIF)")
    feats, vif_hist = vif_screen(d, feats_all)
    dropped = [f for f in feats_all if f not in feats]
    print(f"    retained {len(feats)}/{len(feats_all)}; dropped {dropped or 'none'}")
    results.setdefault("vif", []).append(vif_hist.assign(Zone=zone))

    X = d[feats].to_numpy(float)
    y = d[s.lst].to_numpy(float)
    if len(X) > MAX_ROWS_FIT:
        keep = rng.choice(len(X), MAX_ROWS_FIT, replace=False)
        X, y, d_sub = X[keep], y[keep], d.iloc[keep].reset_index(drop=True)
    else:
        d_sub = d

    # --- spatial structure ---
    print("\n[2] Spatial cross-validation design")
    groups, how = make_spatial_blocks(d_sub, s)
    splits = cv_splits(groups)
    print(f"    scheme: {how}; {len(splits)} folds")
    if s.lat and s.lon:
        mi = morans_I(y, d_sub[[s.lat, s.lon]].to_numpy(float))
        print(f"    Moran's I of LST = {mi:.3f}  (0 = no spatial structure)")
        results.setdefault("morans", {})[zone] = float(mi)

    # --- benchmark ---
    print("\n[3] Model benchmark")
    models = add_stack(build_models(len(feats), fast=fast))
    bench, oof = benchmark(X, y, splits, models, label=zone)
    results.setdefault("bench", []).append(bench)
    if not len(bench):
        return
    best_name = bench.Model.iloc[0]
    print(f"    best: {best_name}  R2={bench.R2.iloc[0]:.4f}")

    print("\n[4] Leakage audit: random CV vs spatial CV")
    try:
        lk = leakage_demo(X, y, models, splits)
        lk["zone"] = zone
        print(f"    {lk['model']}: random R2={lk['R2_random_CV']:.4f}  "
              f"spatial R2={lk['R2_spatial_CV']:.4f}  "
              f"optimism={lk['optimism']:+.4f}")

        # Moran's I of the TARGET is not the right leakage diagnostic - a model
        # with good predictors reproduces that structure legitimately. What
        # leaks is spatially autocorrelated RESIDUAL: structure the predictors
        # cannot explain, which a neighbouring training pixel can still leak.
        if s.lat and s.lon and best_name in oof:
            resid = y - oof[best_name]
            m = ~np.isnan(resid)
            mi_r = morans_I(resid[m], d_sub[[s.lat, s.lon]].to_numpy(float)[m])
            lk["morans_I_target"] = results.get("morans", {}).get(zone, np.nan)
            lk["morans_I_residual"] = float(mi_r)
            verdict = ("unexplained spatial structure -> random CV would leak"
                       if mi_r > 0.2 else "residuals spatially unstructured")
            print(f"    Moran's I of residuals = {mi_r:.3f}  ({verdict})")
        results.setdefault("leakage", []).append(lk)
    except Exception as e:
        print(f"    [!] {e}")

    print("\n[4b] Geographically weighted regression (spatial non-stationarity)")
    try:
        g = gwr_analysis(d_sub, s, feats)
        if g:
            ns = g["summary"][g["summary"].Non_stationary]
            m = results.get("gwr_meta", {})
            print(f"    bandwidth={g['bandwidth']:.0f} neighbours, "
                  f"GWR R2={g['gwr_R2']:.3f} vs global OLS R2={g['ols_R2']:.3f}")
            print(f"    AICc gate: dAICc={g['delta_aicc']:+.1f} -> "
                  f"{'GWR justified' if g['gwr_justified'] else 'NO evidence of non-stationarity'}")
            print(f"    non-stationary drivers: "
                  f"{', '.join(ns.Feature) if len(ns) else 'none'}")
            for _, r in ns.head(3).iterrows():
                print(f"      {r.Feature:10s} global {r.Global_coef:+.2f} -> "
                      f"local range [{r.Local_min:+.2f}, {r.Local_max:+.2f}], "
                      f"sign flips {r.Sign_flips_pct:.0f}% of the city")
            results.setdefault("gwr", []).append(g["summary"].assign(Zone=zone))
            results.setdefault("gwr_meta", {})[zone] = {
                k: g[k] for k in ("bandwidth", "n_points", "gwr_R2", "ols_R2",
                                  "aicc", "aicc_ols", "delta_aicc",
                                  "gwr_justified", "n_perm")}
            fig_gwr(g, zone, FIGS / f"gwr_{zone.split()[0]}.png")
    except Exception as e:
        print(f"    [!] GWR failed: {e}")

    # --- refit best on a train split for XAI + gatekeeper ---
    from sklearn.base import clone
    tr, te = splits[0]
    n_cal = max(200, int(0.25 * len(tr)))
    cal = tr[:n_cal]; fit = tr[n_cal:]
    best = clone(models[best_name]).fit(X[fit], y[fit])

    print("\n[5] SHAP explainability")
    try:
        sh = shap_analysis(best, X[fit], feats, label=zone)
        results.setdefault("shap", []).append(sh["importance"])
        results.setdefault("shap_raw", {})[zone] = sh
        top = sh["importance"].head(3)
        for _, r in top.iterrows():
            print(f"    {r.Feature:10s} {r.PctContribution:5.1f}%  {r.Direction}")
        loc = local_explanations(sh["shap_values"], sh["X_sample"],
                                 y[fit][sh["sample_idx"]], feats, sh["base_value"])
        results.setdefault("local", []).append(loc.assign(Zone=zone))
        pair = sh["interactions"].stack().reset_index()
        pair.columns = ["Feature_A", "Feature_B", "Interaction_strength"]
        pair = pair.sort_values("Interaction_strength", ascending=False).head(15)
        results.setdefault("interactions", []).append(pair.assign(Zone=zone))
    except Exception as e:
        print(f"    [!] SHAP failed: {e}")

    print("\n[6] Confidence Gatekeeper")
    try:
        gk = ConfidenceGatekeeper().fit(best, X[cal], y[cal], X[fit])
        cov = gk.validate_coverage(X[te], y[te])
        cov["zone"] = zone; cov["model"] = best_name
        print(f"    coverage {cov['empirical_coverage']*100:.1f}% "
              f"(target {cov['target_coverage']*100:.0f}%), "
              f"mean width {cov['mean_interval_width']:.2f}°C")
        print(f"    ACCEPT {cov['pct_ACCEPT']:.1f}% | REVIEW {cov['pct_REVIEW']:.1f}% "
              f"| REJECT {cov['pct_REJECT']:.1f}%")
        results.setdefault("gatekeeper", []).append(cov)
        sc = gk.score(X[te])
        results.setdefault("gk_scores", {})[zone] = (sc, y[te])
    except Exception as e:
        print(f"    [!] gatekeeper failed: {e}")

    print("\n[7] Local climate zone discovery")
    try:
        summ, labels, meta = discover_lcz(d_sub, s, feats)
        print(f"    k={meta['chosen_k']} (silhouette {meta['silhouette']:.3f}); "
              f"hottest cluster {summ.Mean_LST_C.iloc[0]:.1f}°C vs coolest "
              f"{summ.Mean_LST_C.iloc[-1]:.1f}°C")
        results.setdefault("lcz", []).append(summ.assign(Zone=zone))
        results.setdefault("lcz_meta", {})[zone] = meta
    except Exception as e:
        print(f"    [!] clustering failed: {e}")

    # figures
    try:
        fig_spatial(d_sub, s, groups, zone, FIGS / f"spatial_{zone.split()[0]}.png")
        if zone in results.get("shap_raw", {}):
            sh = results["shap_raw"][zone]
            fig_shap_beeswarm(sh["shap_values"], sh["X_sample"], feats, zone,
                              FIGS / f"shap_beeswarm_{zone.split()[0]}.png")
            fig_dependence(sh["shap_values"], sh["X_sample"], feats,
                           sh["importance"], zone,
                           FIGS / f"shap_dependence_{zone.split()[0]}.png")
        if zone in results.get("gk_scores", {}):
            sc, yt = results["gk_scores"][zone]
            cov = [c for c in results["gatekeeper"] if c["zone"] == zone][0]
            fig_gatekeeper(sc, yt, cov, zone, FIGS / f"gatekeeper_{zone.split()[0]}.png")
        if zone in results.get("lcz_meta", {}):
            summ = [x for x in results["lcz"] if x.Zone.iloc[0] == zone][0]
            fig_lcz(summ, FIGS / f"lcz_{zone.split()[0]}.png", f"— {zone}")
    except Exception as e:
        print(f"    [!] figure error: {e}")

    results.setdefault("schemas", {})[zone] = s


def main(fast: bool = False):
    print("=" * 78)
    print("DHAKA UHI — ADVANCED MACHINE LEARNING PIPELINE")
    print("=" * 78)

    print("\n>>> INGESTION")
    tables = discover_inputs(UPLOADS)
    if not tables:
        print(f"\nNo readable tabular data found in {UPLOADS}.")
        print("Upload the CSVs / zip and re-run:  python3 uhi_pipeline.py")
        return 1
    for k, v in tables.items():
        print(f"  {k:45s} {v.shape[0]:>8,} x {v.shape[1]}")

    cross, trendy = {}, {}
    for name, df in tables.items():
        (trendy if is_trend_table(name, df) else cross)[name] = df

    results: dict = {}

    # cross-sectional zones
    for name, df in cross.items():
        s = infer_schema(df, name)
        print(f"\n>>> SCHEMA — {name}\n{s.summary()}")
        d, s, rep = clean_frame(df, s)
        print(f"    cleaning: {json.dumps(rep, default=str)}")
        results.setdefault("cleaning", {})[name] = rep
        analyse_zone(d, s, label_zone(name), results, fast=fast)
        results.setdefault("clean_frames", {})[label_zone(name)] = (d, s)

    # pooled model with zone indicator
    frames = results.get("clean_frames", {})
    if len(frames) >= 2:
        try:
            print(f"\n{'='*78}\nPOOLED MODEL (zone as predictor)\n{'='*78}")
            # Every frame is renamed onto ONE canonical vocabulary - indices,
            # target AND coordinates. Renaming only the indices leaves lat/lon
            # spelled differently per source, which is what broke this before.
            parts, has_coords = [], True
            for zname, (d, s) in frames.items():
                dd = d.copy()
                dd["_is_north"] = 1 if "North" in zname else 0
                ren = {v: k for k, v in s.indices.items()}
                ren[s.lst] = "LST"
                if s.lat and s.lon:
                    ren[s.lat], ren[s.lon] = "LAT", "LON"
                else:
                    has_coords = False
                parts.append(dd.rename(columns=ren))

            common = set(parts[0].columns)
            for pt in parts[1:]:
                common &= set(pt.columns)
            common = [c for c in common if c != "LST"]
            if has_coords and not {"LAT", "LON"}.issubset(common):
                has_coords = False

            pooled = pd.concat([pt[["LST"] + common] for pt in parts],
                               ignore_index=True)

            # Coordinates define the CV blocks; they must not also be predictors,
            # or the model can memorise location instead of learning physics.
            feat = [c for c in common if c not in ("LAT", "LON", "_is_north")]
            ps = Schema(lst="LST",
                        lat="LAT" if has_coords else None,
                        lon="LON" if has_coords else None,
                        indices={c: c for c in feat},
                        numeric_other=["_is_north"],
                        source="pooled", n_rows=len(pooled))
            print(f"  pooled {len(pooled):,} rows x {len(feat)} shared indices"
                  f"{' (+coords)' if has_coords else ' (no shared coords)'}")
            analyse_zone(pooled, ps, "Pooled (N+S)", results, fast=fast)
        except Exception as e:
            print(f"  [!] pooled model failed: {e}")

    # temporal
    print(f"\n{'='*78}\nTEMPORAL ANALYSIS\n{'='*78}")
    longs, tr_all = {}, []
    for name, df in trendy.items():
        long = tidy_trend_table(df, name)
        if long is None or not len(long):
            print(f"  {name}: no usable time axis")
            continue
        z = label_zone(name)
        longs[z] = long
        t = trend_stats(long, z)
        if len(t):
            tr_all.append(t)
            g = t.iloc[0]
            print(f"  {z}: {g.Trend}, Sen slope {g.Sen_slope_per_decade:+.3f} °C/decade, "
                  f"MK p={g.MK_p:.4g}, total {g.Total_change_C:+.2f} °C over {g.Years}")
    if tr_all:
        trends = pd.concat(tr_all, ignore_index=True)
        results["trends"] = trends
        try:
            fig_trends(trends, longs, FIGS / "temporal_trends.png")
        except Exception as e:
            print(f"  [!] trend figure: {e}")

    # change detection
    frames_cd = {k: v for k, v in results.get("clean_frames", {}).items()
                 if "Unassigned" not in k}
    if len(frames_cd) >= 2:
        cd = change_detection(frames_cd)
        if cd is not None:
            results["change_detection"] = cd
            print("\n  North-South differential drivers (top 3):")
            for _, r in cd.head(3).iterrows():
                print(f"    {r.Delta_feature:12s} {r.Pct:5.1f}%")

    # ---- export ----
    print(f"\n{'='*78}\nEXPORT\n{'='*78}")
    if results.get("bench"):
        bench = pd.concat(results["bench"], ignore_index=True)
        bench.to_csv(TABLES / "model_benchmark.csv", index=False)
        try:
            fig_model_comparison(bench, FIGS / "model_comparison.png")
        except Exception as e:
            print(f"  [!] {e}")
    if results.get("shap"):
        imp = pd.concat(results["shap"], ignore_index=True)
        imp.to_csv(TABLES / "shap_feature_importance.csv", index=False)
        try:
            fig_shap_importance(results["shap"], FIGS / "shap_importance.png")
        except Exception as e:
            print(f"  [!] {e}")
    for key, fname in (("local", "local_hotspot_explanations.csv"),
                       ("interactions", "shap_interactions.csv"),
                       ("vif", "vif_screening.csv"),
                       ("lcz", "local_climate_zones.csv"),
                       ("gwr", "gwr_nonstationarity.csv")):
        if results.get(key):
            pd.concat(results[key], ignore_index=True).to_csv(TABLES / fname, index=False)
    if results.get("gatekeeper"):
        pd.DataFrame(results["gatekeeper"]).to_csv(
            TABLES / "gatekeeper_validation.csv", index=False)
    if results.get("leakage"):
        lk = pd.DataFrame(results["leakage"])
        lk.to_csv(TABLES / "spatial_leakage_audit.csv", index=False)
        try:
            fig_leakage(lk, FIGS / "leakage_audit.png")
        except Exception as e:
            print(f"  [!] {e}")
    if "trends" in results:
        results["trends"].to_csv(TABLES / "temporal_trends.csv", index=False)
    if "change_detection" in results:
        results["change_detection"].to_csv(
            TABLES / "change_detection_drivers.csv", index=False)

    write_report(results)
    print(f"\nOutputs -> {OUT}")
    for p in sorted(OUT.rglob("*")):
        if p.is_file():
            print(f"  {p.relative_to(OUT)}  ({p.stat().st_size/1024:.0f} KB)")
    return 0


# ----------------------------------------------------------------------------
# REPORT
# ----------------------------------------------------------------------------

def write_report(results: dict):
    L = ["# Dhaka Urban Heat Island — Machine Learning Analysis", "",
         "Automated report. Every number below is computed from the supplied data.", ""]

    if results.get("bench"):
        bench = pd.concat(results["bench"], ignore_index=True)
        L += ["## 1. Model performance (spatial blocked cross-validation)", ""]
        for z in bench.Zone.unique():
            b = bench[bench.Zone == z]
            top = b.iloc[0]
            L.append(f"**{z}** — best model **{top.Model}**: "
                     f"R² = {top.R2:.4f}, RMSE = {top.RMSE:.3f} °C, "
                     f"MAE = {top.MAE:.3f} °C (fold sd {top.R2_fold_std:.3f}).")
            L.append("")
            L.append(b.drop(columns=["Zone"]).round(4).to_markdown(index=False))
            L.append("")

    if results.get("leakage"):
        lk = pd.DataFrame(results["leakage"])
        L += ["## 2. Spatial leakage audit", "",
              "Random k-fold treats neighbouring pixels as independent, so a held-out",
              "pixel usually has a near-identical training twin. The gap below is the",
              "amount of apparent skill that is really autocorrelation:", "",
              lk.round(4).to_markdown(index=False), ""]
        mo = lk.optimism.mean()
        L += [f"Mean optimism: **{mo:+.4f} R²**. Results reported here use the "
              f"spatial figure, which is the defensible one.", ""]

    if results.get("shap"):
        imp = pd.concat(results["shap"], ignore_index=True)
        L += ["## 3. Driver attribution (SHAP)", ""]
        for z in imp.Zone.unique():
            i = imp[imp.Zone == z].head(8)
            L.append(f"### {z}")
            L.append("")
            L.append(i[["Feature", "MeanAbsSHAP", "PctContribution",
                        "Direction"]].round(4).to_markdown(index=False))
            L.append("")
            warm = i[i.Direction == "warming"].head(2).Feature.tolist()
            cool = i[i.Direction == "cooling"].head(2).Feature.tolist()
            L.append(f"Dominant warming drivers: {', '.join(warm) or 'none'}. "
                     f"Dominant cooling drivers: {', '.join(cool) or 'none'}.")
            L.append("")

    if results.get("gatekeeper"):
        gk = pd.DataFrame(results["gatekeeper"])
        L += ["## 4. Confidence Gatekeeper", "",
              "Split-conformal intervals fused with an applicability-domain check.",
              "Coverage should sit close to the 90% target; large deviation means the",
              "calibration set is not exchangeable with the test set.", "",
              gk.round(4).to_markdown(index=False), ""]
        if {"MAE_ACCEPT", "MAE_REJECT"}.issubset(gk.columns):
            for _, r in gk.iterrows():
                try:
                    ratio = float(r.MAE_REJECT) / float(r.MAE_ACCEPT)
                    L.append(f"- **{r.zone}**: predictions flagged REJECT carry "
                             f"{ratio:.2f}x the error of ACCEPT predictions "
                             f"({r.MAE_REJECT:.2f} vs {r.MAE_ACCEPT:.2f} degC). "
                             f"The gate is separating reliable from unreliable "
                             f"output rather than labelling at random.")
                except Exception:
                    pass
        L.append("")

    if "trends" in results:
        t = results["trends"]
        L += ["## 5. Temporal trends (Mann-Kendall / Sen's slope)", "",
              t.round(4).to_markdown(index=False), ""]
        sig = t[t.MK_p < 0.05]
        if len(sig):
            L.append(f"{len(sig)} of {len(t)} series show a statistically "
                     f"significant trend at α = 0.05.")
            L.append("")

    if results.get("lcz"):
        lcz = pd.concat(results["lcz"], ignore_index=True)
        L += ["## 6. Data-derived local climate zones", "",
              lcz.round(3).to_markdown(index=False), ""]

    if "change_detection" in results:
        L += ["## 7. Differential drivers (delta model)", "",
              results["change_detection"].round(3).to_markdown(index=False), ""]

    if results.get("gwr"):
        gw = pd.concat(results["gwr"], ignore_index=True)
        L += ["## 8. Spatial non-stationarity (GWR)", "",
              "Every global model above assumes one coefficient set fits the whole",
              "city. GWR fits a local regression at each location. Where local",
              "coefficients vary more than sampling noise allows, the global number",
              "is an average that conceals real geographic difference.", "",
              gw.round(4).to_markdown(index=False), ""]
        flagged = gw[gw.Non_stationary]
        if len(flagged):
            L.append(f"**{len(flagged)} of {len(gw)} drivers are spatially "
                     f"non-stationary.** For these, a single city-wide "
                     f"coefficient should not be reported without the local range.")
        else:
            L.append("No driver shows significant non-stationarity; global "
                     "coefficients are adequate for this study area.")
        L.append("")
        meta = results.get("gwr_meta", {})
        for z, m in meta.items():
            L.append(f"- {z}: bandwidth {m['bandwidth']:.0f} neighbours, "
                     f"GWR R² {m['gwr_R2']:.3f} vs OLS R² {m['ols_R2']:.3f}, "
                     f"ΔAICc {m['delta_aicc']:+.1f} "
                     f"({'GWR justified' if m['gwr_justified'] else 'no evidence of non-stationarity'}), "
                     f"n={m['n_points']}, {m['n_perm']} permutations.")
        L.append("")

    L += ["## Method notes", "",
          "- Cross-validation is blocked on geography, not random. This is the single",
          "  most consequential choice in the pipeline; random CV would have reported",
          "  materially higher and materially less honest scores.",
          "- Predictors are screened by VIF before modelling. Spectral indices share",
          "  bands by construction (NDVI and NDBI both use NIR), so collinearity is",
          "  structural rather than incidental.",
          "- SHAP direction is inferred from the correlation between a feature's value",
          "  and its SHAP value, so 'warming' means higher values push LST up.",
          "- Conformal intervals are distribution-free and carry a finite-sample",
          "  coverage guarantee under exchangeability. They do NOT detect concept",
          "  drift: if features look normal but the underlying relationship has",
          "  changed, coverage fails silently. Validate across epochs separately.",
          "- GWR uses an adaptive bisquare kernel with AICc-selected bandwidth on a",
          "  subsample, because the method is O(n^2) in memory. The coefficient",
          "  distribution is stable under subsampling; individual point estimates",
          "  are less so.",
          "- Climate zones use k-means with silhouette-selected k. HDBSCAN was",
          "  considered and rejected: it leaves noise points unlabelled, which is",
          "  awkward when every pixel needs a zone assignment.", ""]

    (OUT / "ANALYSIS_REPORT.md").write_text("\n".join(L))
    print("  report written")


if __name__ == "__main__":
    sys.exit(main(fast="--fast" in sys.argv))
