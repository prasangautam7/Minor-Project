"""
Shared utilities for sleep apnea detection pipeline.
"""
import os
import json
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "Data"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"

PATIENT_IDS = [f"{i:02d}" for i in range(1, 51)]


def time_str_to_seconds(t: str) -> float:
    """Convert 'hh:mm:ss.ms' string to seconds since midnight."""
    parts = t.split(":")
    h, m = int(parts[0]), int(parts[1])
    s = float(parts[2])
    return h * 3600 + m * 60 + s


def _make_timestamps_continuous(timestamps: np.ndarray) -> np.ndarray:
    """
    Handle midnight crossings: when timestamps decrease, add 86400
    to make them monotonically increasing (continuous time).
    """
    result = timestamps.copy().astype(float)
    offset = 0.0
    for i in range(1, len(result)):
        if result[i] + offset < result[i - 1] + offset - 3600:
            offset += 86400.0
        result[i] += offset
    if offset == 0 and len(result) > 0:
        result += 0  # no crossing
    else:
        result[0] += 0
        for i in range(1, len(result)):
            pass  # already handled
    return result


def _parse_signal_timestamps(df: pd.DataFrame) -> np.ndarray:
    """Parse absolute time column and handle midnight crossings."""
    raw_secs = df.iloc[:, 1].apply(time_str_to_seconds).values
    offset = 0.0
    result = np.empty_like(raw_secs, dtype=float)
    result[0] = raw_secs[0]
    for i in range(1, len(raw_secs)):
        if raw_secs[i] < raw_secs[i - 1] - 3600:
            offset += 86400.0
        result[i] = raw_secs[i] + offset
    return result


def load_hr(patient_id: str) -> pd.DataFrame:
    path = DATA_DIR / patient_id / f"{patient_id}_HR.csv"
    df = pd.read_csv(path)
    df.columns = ["relative_time", "absolute_time", "HR"]
    df["timestamp_sec"] = _parse_signal_timestamps(df)
    df["HR"] = pd.to_numeric(df["HR"], errors="coerce")
    return df[["timestamp_sec", "HR"]]


def load_spo2(patient_id: str) -> pd.DataFrame:
    path = DATA_DIR / patient_id / f"{patient_id}_SpO2.csv"
    df = pd.read_csv(path)
    df.columns = ["relative_time", "absolute_time", "SpO2"]
    df["timestamp_sec"] = _parse_signal_timestamps(df)
    df["SpO2"] = pd.to_numeric(df["SpO2"], errors="coerce")
    return df[["timestamp_sec", "SpO2"]]


def load_annotations(patient_id: str) -> dict:
    path = DATA_DIR / patient_id / f"{patient_id}_annotation.json"
    with open(path) as f:
        data = json.load(f)

    events = []
    for e in data.get("events", []):
        events.append({
            "event_type": e["event_type"],
            "event_start": e["evnet_start"],
            "event_duration": e["event_duration"],
            "event_end": e["evnet_start"] + e["event_duration"],
        })
    return {
        "record_start": data["record_start"],
        "awake_intervals": data.get("awake_intervals", []),
        "events": events,
    }


def load_patient(patient_id: str) -> dict:
    """Load and merge HR + SpO2 for a single patient, plus annotations."""
    hr = load_hr(patient_id)
    spo2 = load_spo2(patient_id)
    annotations = load_annotations(patient_id)

    # Align signal timestamps with annotation time domain.
    # Annotations use continuous seconds (can exceed 86400 for post-midnight).
    # Signal timestamps are now also continuous after midnight fix.
    # If signal starts on the next day (small values) but annotations reference
    # the previous midnight, shift signals up by 86400.
    record_start = annotations["record_start"]
    sig_start_hr = hr["timestamp_sec"].iloc[0]
    sig_start_spo2 = spo2["timestamp_sec"].iloc[0]

    if record_start > 43200 and sig_start_hr < 43200:
        hr["timestamp_sec"] += 86400
    if record_start > 43200 and sig_start_spo2 < 43200:
        spo2["timestamp_sec"] += 86400

    merged = pd.merge(hr, spo2, on="timestamp_sec", how="inner")
    merged = merged.sort_values("timestamp_sec").reset_index(drop=True)

    return {
        "patient_id": patient_id,
        "signals": merged,
        "annotations": annotations,
    }
