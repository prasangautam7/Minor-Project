"""
Feature extraction from 60-second windows of HR and SpO2 signals.
"""
import numpy as np
from scipy import stats


def _safe_nanstd(x):
    s = np.nanstd(x)
    return s if not np.isnan(s) else 0.0


def _rmssd(x):
    """Root Mean Square of Successive Differences (HRV proxy)."""
    valid = x[~np.isnan(x)]
    if len(valid) < 2:
        return 0.0
    diffs = np.diff(valid)
    return np.sqrt(np.mean(diffs ** 2))


def _slope(x):
    """Linear regression slope over the window."""
    valid_mask = ~np.isnan(x)
    if valid_mask.sum() < 2:
        return 0.0
    t = np.arange(len(x))[valid_mask]
    vals = x[valid_mask]
    slope, _, _, _, _ = stats.linregress(t, vals)
    return slope


def _count_hr_jumps(hr, threshold=5):
    """Count number of consecutive HR increases > threshold bpm."""
    valid = hr[~np.isnan(hr)]
    if len(valid) < 2:
        return 0
    diffs = np.diff(valid)
    return int(np.sum(diffs > threshold))


def _count_desaturations(spo2, drop_threshold=3):
    """Count SpO2 drops >= drop_threshold from running baseline."""
    valid = spo2[~np.isnan(spo2)]
    if len(valid) < 5:
        return 0
    baseline = np.nanmax(valid[:10]) if len(valid) >= 10 else np.nanmax(valid)
    count = 0
    in_desat = False
    for v in valid:
        if baseline - v >= drop_threshold:
            if not in_desat:
                count += 1
                in_desat = True
        else:
            in_desat = False
    return count


def _time_below_threshold(spo2, threshold=90):
    """Number of seconds SpO2 is below threshold."""
    valid = spo2[~np.isnan(spo2)]
    return int(np.sum(valid < threshold))


def _hr_spo2_correlation(hr, spo2):
    """Pearson correlation between HR and SpO2."""
    mask = ~(np.isnan(hr) | np.isnan(spo2))
    if mask.sum() < 5:
        return 0.0
    r, _ = stats.pearsonr(hr[mask], spo2[mask])
    return r if not np.isnan(r) else 0.0


def _time_lag_min_spo2_max_hr(hr, spo2):
    """Time difference between max HR and min SpO2 (brady-tachy pattern)."""
    hr_valid = hr.copy()
    spo2_valid = spo2.copy()
    hr_valid[np.isnan(hr_valid)] = np.nanmean(hr_valid)
    spo2_valid[np.isnan(spo2_valid)] = np.nanmean(spo2_valid)
    if np.all(np.isnan(hr)) or np.all(np.isnan(spo2)):
        return 0.0
    max_hr_idx = np.argmax(hr_valid)
    min_spo2_idx = np.argmin(spo2_valid)
    return float(max_hr_idx - min_spo2_idx)


def extract_features(window: dict) -> dict:
    """Extract all features from a single window."""
    hr = window["hr"].astype(float)
    spo2 = window["spo2"].astype(float)

    feats = {}

    # HR features (8)
    feats["hr_mean"] = np.nanmean(hr)
    feats["hr_std"] = _safe_nanstd(hr)
    feats["hr_min"] = np.nanmin(hr)
    feats["hr_max"] = np.nanmax(hr)
    feats["hr_range"] = feats["hr_max"] - feats["hr_min"]
    feats["hr_rmssd"] = _rmssd(hr)
    feats["hr_jumps"] = _count_hr_jumps(hr)
    feats["hr_slope"] = _slope(hr)

    # SpO2 features (8)
    feats["spo2_mean"] = np.nanmean(spo2)
    feats["spo2_std"] = _safe_nanstd(spo2)
    feats["spo2_min"] = np.nanmin(spo2)
    feats["spo2_max"] = np.nanmax(spo2)
    feats["spo2_desaturations"] = _count_desaturations(spo2)
    feats["spo2_time_below_90"] = _time_below_threshold(spo2, 90)
    feats["spo2_delta"] = feats["spo2_max"] - feats["spo2_min"]
    feats["spo2_slope"] = _slope(spo2)

    # Cross-signal features (3)
    feats["hr_spo2_corr"] = _hr_spo2_correlation(hr, spo2)
    feats["time_lag_spo2_hr"] = _time_lag_min_spo2_max_hr(hr, spo2)
    feats["combined_desat_idx"] = feats["spo2_desaturations"] * feats["hr_jumps"]

    return feats


FEATURE_NAMES = [
    "hr_mean", "hr_std", "hr_min", "hr_max", "hr_range",
    "hr_rmssd", "hr_jumps", "hr_slope",
    "spo2_mean", "spo2_std", "spo2_min", "spo2_max",
    "spo2_desaturations", "spo2_time_below_90", "spo2_delta", "spo2_slope",
    "hr_spo2_corr", "time_lag_spo2_hr", "combined_desat_idx",
]


def extract_all_features(windows: list) -> tuple:
    """
    Extract features for all windows.
    Returns (X: np.ndarray, y: np.ndarray, patient_ids: list)
    """
    features_list = []
    labels = []
    pids = []

    for w in windows:
        feats = extract_features(w)
        features_list.append([feats[name] for name in FEATURE_NAMES])
        labels.append(w["label"])
        pids.append(w["patient_id"])

    X = np.array(features_list, dtype=np.float64)
    y = np.array(labels, dtype=np.int32)

    # Replace any remaining NaN/inf in features
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    return X, y, pids
