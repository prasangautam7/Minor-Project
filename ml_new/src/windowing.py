"""
Windowing and labeling: segment signals into fixed-size windows and assign apnea labels.
"""
import numpy as np
import pandas as pd


WINDOW_SIZE = 60   # seconds
STRIDE = 30        # seconds (50% overlap)
MIN_OVERLAP = 10   # minimum overlap with an apnea event to label window as apnea
MAX_MISSING_FRAC = 0.2  # skip windows with >20% missing values


def label_window(win_start: float, win_end: float, events: list) -> int:
    """
    Label a window as apnea (1) or normal (0).
    Apnea if any event overlaps with the window by >= MIN_OVERLAP seconds.
    Both 'hypo' and 'osa' count as apnea.
    """
    for ev in events:
        overlap_start = max(win_start, ev["event_start"])
        overlap_end = min(win_end, ev["event_end"])
        overlap = overlap_end - overlap_start
        if overlap >= MIN_OVERLAP:
            return 1
    return 0


def create_windows(patient: dict) -> list:
    """
    Create labeled windows from a single patient's data.
    Returns list of dicts with window info + raw signal arrays.
    """
    signals = patient["signals"]
    events = patient["annotations"]["events"]
    pid = patient["patient_id"]

    if signals.empty:
        return []

    t_min = signals["timestamp_sec"].iloc[0]
    t_max = signals["timestamp_sec"].iloc[-1]

    windows = []
    win_start = t_min

    while win_start + WINDOW_SIZE <= t_max:
        win_end = win_start + WINDOW_SIZE

        mask = (signals["timestamp_sec"] >= win_start) & (signals["timestamp_sec"] < win_end)
        segment = signals.loc[mask]

        if len(segment) < WINDOW_SIZE * 0.5:
            win_start += STRIDE
            continue

        hr_vals = segment["HR"].values
        spo2_vals = segment["SpO2"].values

        hr_missing_frac = np.isnan(hr_vals).sum() / len(hr_vals)
        spo2_missing_frac = np.isnan(spo2_vals).sum() / len(spo2_vals)

        if hr_missing_frac > MAX_MISSING_FRAC or spo2_missing_frac > MAX_MISSING_FRAC:
            win_start += STRIDE
            continue

        label = label_window(win_start, win_end, events)

        windows.append({
            "patient_id": pid,
            "win_start": win_start,
            "win_end": win_end,
            "hr": hr_vals,
            "spo2": spo2_vals,
            "label": label,
        })

        win_start += STRIDE

    return windows


def create_all_windows(patients: list) -> list:
    """Create windows for all patients."""
    all_windows = []
    for patient in patients:
        wins = create_windows(patient)
        n_apnea = sum(1 for w in wins if w["label"] == 1)
        n_normal = sum(1 for w in wins if w["label"] == 0)
        print(f"Patient {patient['patient_id']}: {len(wins)} windows "
              f"({n_apnea} apnea, {n_normal} normal)")
        all_windows.extend(wins)
    return all_windows
