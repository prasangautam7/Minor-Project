"""
Data preprocessing: load all patients, clean signals, handle missing values.
"""
import numpy as np
import pandas as pd
from utils import load_patient, PATIENT_IDS


def clean_signals(signals: pd.DataFrame, max_interp_gap: int = 30) -> pd.DataFrame:
    """
    Clean HR and SpO2 signals:
    - Interpolate short NaN gaps (<=max_interp_gap seconds)
    - Mark remaining NaNs for later filtering
    - Clip extreme values
    """
    df = signals.copy()

    # Clip physiologically implausible values
    df.loc[df["HR"] < 30, "HR"] = np.nan
    df.loc[df["HR"] > 200, "HR"] = np.nan
    df.loc[df["SpO2"] < 50, "SpO2"] = np.nan

    # Interpolate short gaps
    for col in ["HR", "SpO2"]:
        mask = df[col].isna()
        groups = mask.ne(mask.shift()).cumsum()
        gap_sizes = mask.groupby(groups).transform("sum")
        short_gaps = mask & (gap_sizes <= max_interp_gap)
        df.loc[short_gaps, col] = np.nan  # keep as NaN for interpolation
        df[col] = df[col].interpolate(method="linear", limit=max_interp_gap)

    df["hr_missing"] = df["HR"].isna().astype(int)
    df["spo2_missing"] = df["SpO2"].isna().astype(int)

    return df


def preprocess_all_patients() -> list:
    """Load and clean data for all 50 patients."""
    all_patients = []
    for pid in PATIENT_IDS:
        try:
            patient = load_patient(pid)
            patient["signals"] = clean_signals(patient["signals"])
            n_total = len(patient["signals"])
            n_valid = patient["signals"][["HR", "SpO2"]].dropna().shape[0]
            print(f"Patient {pid}: {n_total} samples, {n_valid} valid "
                  f"({100*n_valid/n_total:.1f}%), "
                  f"{len(patient['annotations']['events'])} events")
            all_patients.append(patient)
        except Exception as e:
            print(f"Patient {pid}: FAILED - {e}")
    return all_patients


if __name__ == "__main__":
    patients = preprocess_all_patients()
    print(f"\nSuccessfully loaded {len(patients)} patients")
