import io
import os
import base64
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory, jsonify
from werkzeug.utils import secure_filename
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import joblib
from scipy import stats

app = Flask(__name__)
app.secret_key = "sleep-apnea-demo"

ROOT_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT_DIR / "models"
WEBAPP_DIR = Path(__file__).resolve().parent
TEST_DATA_DIR = WEBAPP_DIR / "test_data"
UPLOAD_DIR = WEBAPP_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_SIZE = 60
STRIDE = 30
MIN_OVERLAP = 10
FEATURE_NAMES = [
    "hr_mean", "hr_std", "hr_min", "hr_max", "hr_range",
    "hr_rmssd", "hr_jumps", "hr_slope",
    "spo2_mean", "spo2_std", "spo2_min", "spo2_max",
    "spo2_desaturations", "spo2_time_below_90", "spo2_delta", "spo2_slope",
    "hr_spo2_corr", "time_lag_spo2_hr", "combined_desat_idx",
]


# ---------------------------------------------------------------------------
# Feature extraction (mirrors src/features.py exactly)
# ---------------------------------------------------------------------------

def _rmssd(x):
    valid = x[~np.isnan(x)]
    if len(valid) < 2:
        return 0.0
    return np.sqrt(np.mean(np.diff(valid) ** 2))


def _slope(x):
    mask = ~np.isnan(x)
    if mask.sum() < 2:
        return 0.0
    t = np.arange(len(x))[mask]
    return stats.linregress(t, x[mask]).slope


def _count_hr_jumps(hr, threshold=5):
    valid = hr[~np.isnan(hr)]
    if len(valid) < 2:
        return 0
    return int(np.sum(np.diff(valid) > threshold))


def _count_desaturations(spo2, drop_threshold=3):
    valid = spo2[~np.isnan(spo2)]
    if len(valid) < 5:
        return 0
    baseline = np.nanmax(valid[:10]) if len(valid) >= 10 else np.nanmax(valid)
    count, in_desat = 0, False
    for v in valid:
        if baseline - v >= drop_threshold:
            if not in_desat:
                count += 1
                in_desat = True
        else:
            in_desat = False
    return count


def extract_window_features(hr, spo2):
    """Extract 19 features from a single window of HR and SpO2 arrays."""
    hr = hr.astype(float)
    spo2 = spo2.astype(float)
    f = {}

    f["hr_mean"] = np.nanmean(hr)
    f["hr_std"] = np.nanstd(hr) if not np.all(np.isnan(hr)) else 0.0
    f["hr_min"] = np.nanmin(hr)
    f["hr_max"] = np.nanmax(hr)
    f["hr_range"] = f["hr_max"] - f["hr_min"]
    f["hr_rmssd"] = _rmssd(hr)
    f["hr_jumps"] = _count_hr_jumps(hr)
    f["hr_slope"] = _slope(hr)

    f["spo2_mean"] = np.nanmean(spo2)
    f["spo2_std"] = np.nanstd(spo2) if not np.all(np.isnan(spo2)) else 0.0
    f["spo2_min"] = np.nanmin(spo2)
    f["spo2_max"] = np.nanmax(spo2)
    f["spo2_desaturations"] = _count_desaturations(spo2)
    f["spo2_time_below_90"] = int(np.sum(spo2[~np.isnan(spo2)] < 90))
    f["spo2_delta"] = f["spo2_max"] - f["spo2_min"]
    f["spo2_slope"] = _slope(spo2)

    mask = ~(np.isnan(hr) | np.isnan(spo2))
    if mask.sum() >= 5:
        r, _ = stats.pearsonr(hr[mask], spo2[mask])
        f["hr_spo2_corr"] = r if not np.isnan(r) else 0.0
    else:
        f["hr_spo2_corr"] = 0.0

    hr_c = hr.copy(); hr_c[np.isnan(hr_c)] = np.nanmean(hr_c)
    spo2_c = spo2.copy(); spo2_c[np.isnan(spo2_c)] = np.nanmean(spo2_c)
    f["time_lag_spo2_hr"] = float(np.argmax(hr_c) - np.argmin(spo2_c))
    f["combined_desat_idx"] = f["spo2_desaturations"] * f["hr_jumps"]

    return [f[name] for name in FEATURE_NAMES]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

_model_cache = {}

def get_model():
    if "model" not in _model_cache:
        path = MODELS_DIR / "random_forest.joblib"
        if not path.exists():
            raise FileNotFoundError(
                f"Model not found at {path}. Run src/train.py first."
            )
        _model_cache["model"] = joblib.load(path)
    return _model_cache["model"]


# ---------------------------------------------------------------------------
# Input normalization
# ---------------------------------------------------------------------------

def normalize_columns(df: pd.DataFrame):
    """Map various column names to standard names. Returns (df, has_spo2, resolution)."""
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Heart rate
    hr_mapped = False
    if "heart_rate_bpm" in df.columns:
        hr_mapped = True
    else:
        for c in df.columns:
            if any(k in c for k in ["heart_rate", "heartrate", "hr", "bpm"]):
                df.rename(columns={c: "heart_rate_bpm"}, inplace=True)
                hr_mapped = True
                break

    # SpO2
    spo2_mapped = False
    for c in df.columns:
        cl = c.lower()
        if any(k in cl for k in ["spo2", "oxygen", "o2_sat", "osat", "saturation"]):
            if c != "spo2":
                df.rename(columns={c: "spo2"}, inplace=True)
            spo2_mapped = True
            break

    # Time column - detect resolution
    resolution = "second"
    if "second" in df.columns:
        df.rename(columns={"second": "time"}, inplace=True)
    elif "minute" in df.columns:
        resolution = "minute"
        df.rename(columns={"minute": "time"}, inplace=True)
    elif "time" not in df.columns:
        df["time"] = np.arange(len(df))

    # Coerce numeric columns to numeric, invalid entries become NaN
    if hr_mapped:
        df["heart_rate_bpm"] = pd.to_numeric(df["heart_rate_bpm"], errors="coerce")
    if spo2_mapped:
        df["spo2"] = pd.to_numeric(df["spo2"], errors="coerce")

    # Time normalization
    if "time" in df.columns:
        df["time"] = pd.to_numeric(df["time"], errors="coerce")
        if df["time"].isna().any():
            df["time"] = df["time"].interpolate(method="linear").fillna(method="ffill").fillna(method="bfill")
    else:
        df["time"] = np.arange(len(df), dtype=float)

    # Label (optional)
    if "label" in df.columns:
        def map_label(val):
            if isinstance(val, str):
                v = val.strip().lower()
                if v in ("a", "apnea", "1", "true", "yes"):
                    return 1
                return 0
            try:
                return int(val)
            except Exception:
                return 0
        df["label"] = df["label"].apply(map_label)

    return df, hr_mapped, spo2_mapped, resolution


def expand_minute_to_second(df):
    """Expand per-minute data to per-second via interpolation."""
    rows = []
    for _, row in df.iterrows():
        minute = int(row["time"])
        for s in range(60):
            rows.append({
                "time": minute * 60 + s,
                "heart_rate_bpm": row["heart_rate_bpm"],
                "spo2": row.get("spo2", np.nan),
            })
    result = pd.DataFrame(rows)
    # Interpolate for smoother signal
    for col in ["heart_rate_bpm", "spo2"]:
        if col in result.columns:
            result[col] = result[col].astype(float).interpolate(method="linear")
    # Carry labels: each minute's label applies to all 60 seconds
    if "label" in df.columns:
        label_map = {}
        for _, row in df.iterrows():
            minute = int(row["time"])
            for s in range(60):
                label_map[minute * 60 + s] = row["label"]
        result["label"] = result["time"].map(label_map).fillna(0).astype(int)
    return result


# ---------------------------------------------------------------------------
# Windowing and prediction
# ---------------------------------------------------------------------------

def create_windows_and_predict(df, model):
    """
    Window the per-second data, extract features, predict.
    Returns per-second predictions and probabilities.
    """
    # safe numeric conversion (bad tokens -> NaN)
    hr = pd.to_numeric(df["heart_rate_bpm"], errors="coerce").values.astype(float)
    spo2 = (pd.to_numeric(df["spo2"], errors="coerce").values.astype(float)
            if "spo2" in df.columns else np.full(len(df), np.nan))
    times_series = pd.to_numeric(df["time"], errors="coerce").interpolate(method="linear")
    # pandas newer versions may not support fillna(method=...), use ffill/bfill directly.
    times = times_series.ffill().bfill().values.astype(float)
    n = len(df)

    features_list = []
    window_centers = []
    window_ranges = []

    i = 0
    while i + WINDOW_SIZE <= n:
        hr_win = hr[i:i + WINDOW_SIZE]
        spo2_win = spo2[i:i + WINDOW_SIZE]

        if np.isnan(hr_win).sum() / WINDOW_SIZE > 0.3:
            i += STRIDE
            continue

        feats = extract_window_features(hr_win, spo2_win)
        features_list.append(feats)
        window_centers.append(i + WINDOW_SIZE // 2)
        window_ranges.append((i, i + WINDOW_SIZE))
        i += STRIDE

    if not features_list:
        return np.zeros(n, dtype=int), np.zeros(n, dtype=float)

    X = np.array(features_list, dtype=np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    y_pred_win = model.predict(X)
    y_proba_win = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else None

    # Map window predictions back to per-second using voting
    pred_scores = np.zeros(n, dtype=float)
    pred_counts_arr = np.zeros(n, dtype=float)

    for idx, (start, end) in enumerate(window_ranges):
        prob = y_proba_win[idx] if y_proba_win is not None else float(y_pred_win[idx])
        pred_scores[start:end] += prob
        pred_counts_arr[start:end] += 1

    mask = pred_counts_arr > 0
    pred_scores[mask] /= pred_counts_arr[mask]

    y_pred_sec = (pred_scores >= 0.5).astype(int)
    return y_pred_sec, pred_scores


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def figure_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def plot_time_series(time_min, hr, spo2, y_pred, has_spo2):
    plt.style.use('seaborn-v0_8-whitegrid')
    n_axes = 2 if has_spo2 else 1
    fig, axes = plt.subplots(n_axes, 1, figsize=(14, 3.5 * n_axes), sharex=True)
    if n_axes == 1:
        axes = [axes]
    fig.patch.set_facecolor('#f8fafc')

    for ax in axes:
        ax.set_facecolor('#f8fafc')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#e2e8f0')
        ax.spines['bottom'].set_color('#e2e8f0')
        ax.grid(True, alpha=0.4, color='#e2e8f0')
        ax.tick_params(colors='#64748b')

    # HR plot
    axes[0].fill_between(time_min, hr, alpha=0.1, color="#6366f1")
    axes[0].plot(time_min, hr, color="#6366f1", linewidth=0.8, alpha=0.8)
    axes[0].set_ylabel("Heart Rate (BPM)", fontsize=11, color='#475569')
    axes[0].set_title("Heart Rate & SpO2 with Apnea Events" if has_spo2 else "Heart Rate with Apnea Events",
                       fontsize=12, fontweight='600', color='#1e293b')

    # Shade apnea
    apnea_mask = y_pred == 1
    if apnea_mask.any():
        for ax in axes:
            for i in range(len(time_min)):
                if apnea_mask[i]:
                    ax.axvspan(time_min[i], time_min[i] + 1/60, color="#ef4444", alpha=0.2, linewidth=0)

    # SpO2 plot
    if has_spo2 and len(axes) > 1:
        axes[1].fill_between(time_min, spo2, alpha=0.1, color="#10b981")
        axes[1].plot(time_min, spo2, color="#10b981", linewidth=0.8, alpha=0.8)
        axes[1].axhline(y=90, color='#ef4444', linestyle='--', alpha=0.5, linewidth=1)
        axes[1].set_ylabel("SpO2 (%)", fontsize=11, color='#475569')

    axes[-1].set_xlabel("Time (Minutes)", fontsize=11, color='#475569')

    from matplotlib.patches import Patch
    legend_elements = [
        plt.Line2D([0], [0], color='#6366f1', linewidth=2, label='Heart Rate'),
        Patch(facecolor='#ef4444', alpha=0.25, label='Predicted Apnea')
    ]
    if has_spo2:
        legend_elements.insert(1, plt.Line2D([0], [0], color='#10b981', linewidth=2, label='SpO2'))
    axes[0].legend(handles=legend_elements, loc='upper right', framealpha=0.9, fontsize=9)

    plt.tight_layout()
    return fig


def plot_probability(time_min, y_proba):
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 4))
    fig.patch.set_facecolor('#f8fafc')
    ax.set_facecolor('#f8fafc')

    ax.fill_between(time_min, y_proba, alpha=0.15, color="#10b981")
    ax.plot(time_min, y_proba, color="#10b981", linewidth=1.5)
    ax.axhline(0.5, color="#ef4444", linestyle="--", linewidth=2, alpha=0.8, label="Threshold (0.5)")

    high_risk = y_proba > 0.5
    if high_risk.any():
        ax.fill_between(time_min, 0.5, y_proba, where=high_risk,
                        alpha=0.2, color="#ef4444", interpolate=True)

    ax.set_xlabel("Time (Minutes)", fontsize=11, color='#475569')
    ax.set_ylabel("Apnea Probability", fontsize=11, color='#475569')
    ax.set_ylim(-0.02, 1.02)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#e2e8f0')
    ax.spines['bottom'].set_color('#e2e8f0')
    ax.grid(True, alpha=0.4, color='#e2e8f0')
    ax.tick_params(colors='#64748b')
    ax.legend(loc='upper right', framealpha=0.9)
    plt.tight_layout()
    return fig


def compute_severity(apnea_seconds, total_seconds):
    hours = total_seconds / 3600.0
    if hours <= 0:
        return 0, "Normal / None"
    # Count distinct apnea events (transitions from 0->1)
    # For AHI we use the apnea minute count as a proxy
    apnea_minutes = apnea_seconds / 60.0
    ahi = apnea_minutes / hours
    if ahi < 5:
        level = "Normal / None"
    elif ahi < 15:
        level = "Mild"
    elif ahi < 30:
        level = "Moderate"
    else:
        level = "Severe"
    return ahi, level


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/download_sample")
def download_sample():
    return send_from_directory(TEST_DATA_DIR, "sample_patient_35.csv", as_attachment=True)


@app.route("/files", methods=["GET"])
def files():
    rows = collection.find({}, {"_id": 0, "filename": 1}).sort("uploaded_at", -1)
    file_list = [r["filename"] for r in rows]
    return jsonify({"files": file_list})


@app.route("/analyze", methods=["POST"])
def analyze():
    file = request.files.get("data_file")
    selected_file = request.form.get("selected_file", "").strip()
    df = None

    if file and file.filename:
        try:
            df = pd.read_csv(file)
        except Exception:
            flash("Could not read the uploaded CSV file. Please check the format.")
            return redirect(url_for("index"))

    elif selected_file:
        entry = collection.find_one({"filename": selected_file})
        if not entry:
            flash("Selected file not found in database.")
            return redirect(url_for("index"))

        url = entry.get("url")
        if not url:
            flash("No valid URL found for selected file entry.")
            return redirect(url_for("index"))

        try:
            import requests
            r = requests.get(url)
            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text))
        except Exception:
            flash("Could not download/parse the selected file from cloud storage URL.")
            return redirect(url_for("index"))
    else:
        flash("Please upload a file or select a stored file.")
        return redirect(url_for("index"))

    df, hr_ok, has_spo2, resolution = normalize_columns(df)
    if not hr_ok:
        flash("CSV must include a heart rate column (e.g., heart_rate_bpm, hr, bpm).")
        return redirect(url_for("index"))

    if not has_spo2:
        flash("CSV must include an SpO2 column (e.g., spo2, oxygen_saturation). "
              "Both HR and SpO2 are required for accurate apnea detection.")
        return redirect(url_for("index"))

    has_labels = "label" in df.columns and df["label"].nunique() > 1

    # Expand per-minute data to per-second
    if resolution == "minute":
        df_sec = expand_minute_to_second(df)
        print(f"Expanded {len(df)} minute rows to {len(df_sec)} second rows")
    else:
        df_sec = df.copy()

    # Reject fully invalid signal data after coercion
    if df_sec["heart_rate_bpm"].isna().all():
        flash("No valid numeric heart rate values found after cleaning. Please check your CSV format.")
        return redirect(url_for("index"))
    if has_spo2 and df_sec["spo2"].isna().all():
        flash("No valid numeric SpO2 values found after cleaning. Please check your CSV format.")
        return redirect(url_for("index"))

    # Load model and predict
    model = get_model()
    y_pred, y_proba = create_windows_and_predict(df_sec, model)

    total_seconds = len(df_sec)
    apnea_seconds = int(y_pred.sum())
    total_minutes_display = total_seconds / 60.0

    ahi, severity = compute_severity(apnea_seconds, total_seconds)

    # Downsample to per-minute for visualization
    n_minutes = int(np.ceil(total_seconds / 60))
    time_min = np.arange(n_minutes).astype(float)
    hr_min = np.array([np.nanmean(df_sec["heart_rate_bpm"].values[i*60:min((i+1)*60, total_seconds)])
                        for i in range(n_minutes)])
    spo2_min = np.array([np.nanmean(df_sec["spo2"].values[i*60:min((i+1)*60, total_seconds)])
                          for i in range(n_minutes)]) if has_spo2 else np.zeros(n_minutes)
    pred_min = np.array([int(y_pred[i*60:min((i+1)*60, total_seconds)].mean() >= 0.5)
                          for i in range(n_minutes)])
    proba_min = np.array([float(y_proba[i*60:min((i+1)*60, total_seconds)].mean())
                           for i in range(n_minutes)])

    # Per-minute labels for metrics
    if has_labels:
        if resolution == "minute":
            y_true_min = df["label"].values[:n_minutes]
        else:
            y_true_min = np.array([int(df_sec["label"].values[i*60:min((i+1)*60, total_seconds)].mean() >= 0.5)
                                    for i in range(n_minutes)])

    # Metrics (only with ground truth)
    metrics = None
    if has_labels:
        metrics = {
            "accuracy": accuracy_score(y_true_min, pred_min),
            "precision": precision_score(y_true_min, pred_min, zero_division=0),
            "recall": recall_score(y_true_min, pred_min, zero_division=0),
            "f1": f1_score(y_true_min, pred_min, zero_division=0),
        }

    # Plots
    ts_fig = plot_time_series(time_min, hr_min, spo2_min, pred_min, has_spo2)
    prob_fig = plot_probability(time_min, proba_min)
    ts_img = figure_to_base64(ts_fig)
    prob_img = figure_to_base64(prob_fig)

    apnea_min_count = int(pred_min.sum())
    normal_min_count = int((pred_min == 0).sum())

    return render_template(
        "results.html",
        model_name="Random Forest (HR + SpO2)",
        total_minutes=n_minutes,
        apnea_minutes=apnea_min_count,
        apnea_percent=(apnea_min_count / n_minutes * 100) if n_minutes else 0,
        ahi=ahi,
        severity=severity,
        metrics=metrics,
        pred_counts={"normal": normal_min_count, "apnea": apnea_min_count},
        proba_stats={
            "mean": float(np.mean(proba_min)),
            "median": float(np.median(proba_min)),
            "p90": float(np.percentile(proba_min, 90)),
            "p95": float(np.percentile(proba_min, 95)),
        },
        ts_img=ts_img,
        prob_img=prob_img,
        has_spo2=has_spo2,
    )

import cloudinary
from cloudinary import uploader
from pymongo import MongoClient

client = MongoClient("mongodb+srv://prasan:prasan123@cluster0.iecuydb.mongodb.net/?appName=Cluster0")
db = client["Minor_Project"]
collection = db["files"]

cloudinary.config(
    cloud_name="dfdkmr4wu",
    api_key="927846171834547",
    api_secret="fpHRe4SGWIhl-ncpZxnFP4qaJEI"
)

@app.route('/upload', methods=['POST'])
def upload_csv():
    # 1) Multipart/form-data upload field (standard browser/form method)
    if 'file' in request.files:
        uploaded_file = request.files['file']
        if uploaded_file.filename == '':
            return jsonify({'error': 'filename missing'}), 400
        safe_name = secure_filename(uploaded_file.filename)
        if not safe_name.lower().endswith('.csv'):
            safe_name += '.csv'
        target_path = UPLOAD_DIR / safe_name
        uploaded_file.save(target_path)
        result = uploader.upload(
            str(target_path),
            resource_type="raw",
            public_id=uploaded_file.filename,
            overwrite=True
        )

        file_url = result.get('secure_url')
        collection.insert_one({
            "filename": uploaded_file.filename,
            "url": file_url,
            "path": str(target_path),
            "uploaded_at": datetime.utcnow()
        })
        return jsonify({'status': 'ok', 'path': str(target_path), 'file_url': file_url}), 200

    # 2) Raw body upload (ESP32 may send text/csv or octet-stream)
    content_type = (request.content_type or '').lower()
    raw = request.get_data() or b''
    if raw:
        if 'text/csv' in content_type or 'application/csv' in content_type or 'application/octet-stream' in content_type:
            filename = request.args.get('filename', None) or request.args.get('name', None)
            if not filename:
                filename = f"uploaded_{datetime.utcnow():%Y%m%d_%H%M%S_%f}.csv"
            safe_name = secure_filename(filename)
            if not safe_name.lower().endswith('.csv'):
                safe_name += '.csv'
            target_path = UPLOAD_DIR / safe_name
            with open(target_path, 'wb') as f:
                f.write(raw)
            return jsonify({'status': 'ok', 'path': str(target_path)}), 200

    return jsonify({'error': 'No CSV data received or unsupported content type'}), 400

if __name__ == "__main__":
    app.run(
        host='0.0.0.0', 
        port=int(os.environ.get("PORT", 5000))
    )
