from __future__ import annotations
"""Allometric height model: fitting, prediction and held-out scoring.

The model is H = exp(A + B * ln(CPA_m2)), fitted by OLS of ln(H) ~ ln(CPA).

Before this module the fit was written out three times — shadow_analysis.ipynb cell 25,
baumkataster_analysis.ipynb cell 12 (byte-identical) and validation.ipynb cell 34 — so
editing one silently desynced the others. Everything here is the single source.

Two R-squared conventions are in circulation in this project and they are NOT comparable:
config.py's 0.531 is computed in ln-space, while validation.ipynb section 6 scores in
metres against a constant-at-the-mean baseline. `score()` returns both, labelled, so a
number can never be quoted against the wrong reference.
"""

import numpy as np
import pandas as pd

# The registry filter that produced the n=84,081 citywide fit quoted in config.py.
# Records at or below either floor carry placeholder zeros rather than survey values
# (the same floors as BK_MIN_HEIGHT_M / BK_MIN_CROWN_DIAM_M, restated as a query so a
# notebook can apply them to a plain DataFrame). Keep using it so any hold-out number
# stays comparable to the 0.531.
BK_FIT_FILTER = "Baumhoehe > 1 and Kronendurchmesser > 0.5"


def bk_crown_area(kronendurchmesser) -> np.ndarray:
    """Crown projected area (m²) from registry crown diameter (m).

    Matches the CPA definition used in src/shadow/cadastre.py:108 and in the original
    fit cells: a circular crown, area = pi * (diameter / 2)**2.
    """
    d = np.asarray(kronendurchmesser, dtype=float)
    return np.pi * (d / 2.0) ** 2


def fit_allometry(cpa_m2, height_m) -> tuple[float, float, int]:
    """OLS of ln(H) ~ ln(CPA). Returns (A, B, n).

    Rows with a non-positive area or height are dropped — ln is undefined there and the
    registry uses 0 as "not surveyed", so they carry no information either way.
    """
    cpa = np.asarray(cpa_m2, dtype=float)
    h = np.asarray(height_m, dtype=float)
    ok = (cpa > 0) & (h > 0) & np.isfinite(cpa) & np.isfinite(h)
    x, y = np.log(cpa[ok]), np.log(h[ok])
    A, B = np.linalg.lstsq(np.column_stack([np.ones(len(x)), x]), y, rcond=None)[0]
    return float(A), float(B), int(ok.sum())


def predict_height(cpa_m2, A: float, B: float) -> np.ndarray:
    """Estimated height (m) for each crown area.

    Note this is the conditional MEDIAN, not the mean — exp() of a least-squares fit in
    ln-space does not retransform to the mean. See `smearing_factor` before comparing it
    against a mean-based metric.
    """
    cpa = np.asarray(cpa_m2, dtype=float)
    return np.exp(A + B * np.log(np.maximum(cpa, 1e-6)))


def smearing_factor(cpa_m2, height_m, A: float, B: float) -> float:
    """Duan (1983) smearing estimate, computed on the TRAINING rows.

    Multiplying `predict_height` by this converts a median predictor into a mean
    predictor. Reported as a diagnostic only — applying it changes the estimand, so a
    smeared prediction must not be scored with median-based metrics.
    """
    cpa = np.asarray(cpa_m2, dtype=float)
    h = np.asarray(height_m, dtype=float)
    ok = (cpa > 0) & (h > 0) & np.isfinite(cpa) & np.isfinite(h)
    resid = np.log(h[ok]) - (A + B * np.log(cpa[ok]))
    return float(np.exp(resid).mean())


def score(height_m, pred_m, baseline_mean: float | None = None) -> dict:
    """Accuracy of estimated heights against measured heights.

    Parameters
    ----------
    baseline_mean : float, optional
        The constant predictor the metres-space R² is measured against — pass the TRAIN
        mean so folds stay comparable. Defaults to the mean of `height_m`, which makes
        each fold its own reference and is only appropriate for a single pooled score.

    Returns bias / MAE / RMSE and both R² conventions. `bias` is mean(pred - obs), so a
    positive value means the model over-predicts.
    """
    y = np.asarray(height_m, dtype=float)
    p = np.asarray(pred_m, dtype=float)
    ok = (y > 0) & (p > 0) & np.isfinite(y) & np.isfinite(p)
    y, p = y[ok], p[ok]

    base = float(np.mean(y)) if baseline_mean is None else float(baseline_mean)
    err = p - y
    ly, lp = np.log(y), np.log(p)

    return {
        "n": int(len(y)),
        "bias_m": float(err.mean()),
        "MAE_m": float(np.abs(err).mean()),
        "RMSE_m": float(np.sqrt((err ** 2).mean())),
        # ln-space: directly comparable to the 0.531 quoted in config.py
        "r2_ln": float(1 - ((ly - lp) ** 2).sum() / ((ly - ly.mean()) ** 2).sum()),
        # metres-space against a constant predictor; below 0 means it loses to that constant
        "r2_m": float(1 - (err ** 2).sum() / ((y - base) ** 2).sum()),
        "median_ratio": float(np.median(p / y)),
    }


def label_landuse(gdf) -> pd.Series:
    """Split registry rows into "street" and "green" on `Objektart lang`.

    AMT 66 is the road authority, so its trees are street trees: 30,231 of its 30,410
    rows carry "/SBG" (Straßenbegleitgrün) in Objektbezeichnung and no Öffentliches Grün
    row does. Spielplatz (n=1,303 citywide) is folded into green rather than left as a
    rare third level that no split could populate.
    """
    art = gdf["Objektart lang"].astype("string")
    return pd.Series(
        np.where(art == "AMT 66", "street", "green"),
        index=gdf.index,
        name="landuse",
    )


def noise_ceiling(cpa_m2, height_m, by=None) -> float:
    """Highest ln-space R² any function of the given covariates could reach.

    Registry crown diameter is field-estimated to the nearest metre and height takes only
    a few dozen distinct values, so two trees sharing a diameter are indistinguishable to
    any model reading only CPA — whatever height spread remains inside that bin is
    unlearnable. Grouping by rounded CPA and taking each group's mean ln(H) is therefore
    the best attainable predictor, and its R² is an exact ceiling rather than a guess.

    Pass `by` (e.g. a land-use label) to bin on (CPA, by) instead, and the increase over
    the CPA-only ceiling is exactly what that covariate could add.
    """
    cpa = np.asarray(cpa_m2, dtype=float)
    h = np.asarray(height_m, dtype=float)
    ok = (cpa > 0) & (h > 0) & np.isfinite(cpa) & np.isfinite(h)
    y = np.log(h[ok])

    # Round to the diameter grid the surveyor actually recorded, not to the derived area.
    diam = np.round(2 * np.sqrt(cpa[ok] / np.pi)).astype(int)
    keys = diam if by is None else list(zip(diam, np.asarray(by)[ok]))

    bin_mean = pd.Series(y).groupby(pd.Series(keys, dtype="object")).transform("mean")
    return float(1 - ((y - bin_mean.to_numpy()) ** 2).sum() / ((y - y.mean()) ** 2).sum())
