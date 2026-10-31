"""ML functions for the Wine clustering DAG.

Every function reads its inputs from files and writes its outputs to files in
WORK_DIR/runs/<run_id>/. Each returns only a small JSON-safe dict (paths and
metrics), so no datasets or pickled objects pass through XCom.
"""
import base64
import io
import json
import logging
import os
import re

import joblib
import matplotlib

matplotlib.use("Agg")  # no display inside the container
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from kneed import KneeLocator
from sklearn.cluster import KMeans
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)

WORK_DIR = os.environ.get("WORK_DIR", "/opt/airflow/working_data")
LABEL = "cultivar"
RANDOM_STATE = 42


def _run_dir(run_id: str) -> str:
    # Airflow run ids contain ':' and '+', which are not valid in Windows file names
    # (working_data is mounted from the Windows host), so keep only safe characters.
    safe = re.sub(r"[^A-Za-z0-9_-]", "_", run_id)
    path = os.path.join(WORK_DIR, "runs", safe)
    os.makedirs(path, exist_ok=True)
    return path


def _feature_columns(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c != LABEL]


def extract_data(run_id: str) -> dict:
    """Load the Wine dataset (178 samples, 13 features, 3 cultivars) and save it as raw.csv."""
    wine = load_wine(as_frame=True)
    df = wine.data.copy()
    df[LABEL] = wine.target
    path = os.path.join(_run_dir(run_id), "raw.csv")
    df.to_csv(path, index=False)
    log.info("Saved %d rows x %d columns to %s", len(df), df.shape[1], path)
    return {"path": path, "rows": int(len(df)), "features": len(_feature_columns(df))}


def validate_data(run_id: str, min_rows: int) -> dict:
    """Data quality gate: raise (and fail the run) if the raw data breaks any check."""
    df = pd.read_csv(os.path.join(_run_dir(run_id), "raw.csv"))
    features = _feature_columns(df)
    checks = {
        f"at least {min_rows} rows (found {len(df)})": len(df) >= int(min_rows),
        f"13 feature columns (found {len(features)})": len(features) == 13,
        "label column present": LABEL in df.columns,
        "no missing values": not df.isna().any().any(),
        "all features numeric": all(pd.api.types.is_numeric_dtype(df[c]) for c in features),
        "no negative feature values": bool((df[features] >= 0).all().all()),
    }
    failed = [name for name, ok in checks.items() if not ok]
    for name, ok in checks.items():
        log.info("%s %s", "PASS" if ok else "FAIL", name)
    if failed:
        raise ValueError(f"Data validation failed: {failed}")
    return {"rows": int(len(df)), "checks_passed": len(checks)}


def preprocess(run_id: str) -> dict:
    """Standardize the 13 features and save both the scaled data and the fitted scaler."""
    run_dir = _run_dir(run_id)
    df = pd.read_csv(os.path.join(run_dir, "raw.csv"))
    features = _feature_columns(df)
    # Feature ranges differ by ~1000x (proline in the hundreds, hue around 1),
    # so without scaling proline alone would decide the clusters.
    scaler = StandardScaler()
    scaled = pd.DataFrame(scaler.fit_transform(df[features]), columns=features)
    scaled.to_csv(os.path.join(run_dir, "features_scaled.csv"), index=False)
    joblib.dump(scaler, os.path.join(run_dir, "scaler.joblib"))
    log.info("Scaled %d features; scaler saved for use at prediction time", len(features))
    return {"rows": int(len(scaled)), "features": len(features)}


def tune_k(run_id: str, k_min: int, k_max: int) -> dict:
    """Fit K-Means for each k, pick k by silhouette score, and report the elbow k for comparison."""
    k_min, k_max = int(k_min), int(k_max)
    if k_min < 2 or k_max <= k_min:
        raise ValueError(f"Need 2 <= k_min < k_max, got k_min={k_min}, k_max={k_max}")

    run_dir = _run_dir(run_id)
    X = pd.read_csv(os.path.join(run_dir, "features_scaled.csv")).values
    y = pd.read_csv(os.path.join(run_dir, "raw.csv"))[LABEL].values

    rows = []
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(X)
        rows.append({
            "k": k,
            "sse": round(float(km.inertia_), 2),
            "silhouette": round(float(silhouette_score(X, km.labels_)), 4),
            # ARI is used only to evaluate the choice, never to make it.
            "ari": round(float(adjusted_rand_score(y, km.labels_)), 4),
        })
        log.info("k=%d sse=%.1f silhouette=%.4f ari=%.4f", k, rows[-1]["sse"], rows[-1]["silhouette"], rows[-1]["ari"])

    ks = [r["k"] for r in rows]
    best = max(rows, key=lambda r: r["silhouette"])
    elbow = KneeLocator(ks, [r["sse"] for r in rows], curve="convex", direction="decreasing").elbow
    elbow_k = int(elbow) if elbow is not None else None
    elbow_ari = next((r["ari"] for r in rows if r["k"] == elbow_k), None)

    result = {
        "best_k": best["k"],
        "best_silhouette": best["silhouette"],
        "best_ari": best["ari"],
        "elbow_k": elbow_k,
        "elbow_ari": elbow_ari,
        "k_range": [k_min, k_max],
        "table": rows,
    }
    with open(os.path.join(run_dir, "tuning.json"), "w") as f:
        json.dump(result, f, indent=2)
    log.info("Silhouette picks k=%d (ARI %.4f); elbow picks k=%s (ARI %s)",
             best["k"], best["ari"], elbow_k, elbow_ari)
    return {k: result[k] for k in ("best_k", "best_silhouette", "best_ari", "elbow_k", "elbow_ari")}


def train_final(run_id: str) -> dict:
    """Train the final model with the chosen k, save it, and record cluster assignments."""
    run_dir = _run_dir(run_id)
    with open(os.path.join(run_dir, "tuning.json")) as f:
        best_k = json.load(f)["best_k"]

    X = pd.read_csv(os.path.join(run_dir, "features_scaled.csv")).values
    y = pd.read_csv(os.path.join(run_dir, "raw.csv"))[LABEL].values
    model = KMeans(n_clusters=best_k, n_init=10, random_state=RANDOM_STATE).fit(X)
    joblib.dump(model, os.path.join(run_dir, "model.joblib"))

    pd.DataFrame({"cluster": model.labels_, LABEL: y}).to_csv(
        os.path.join(run_dir, "assignments.csv"), index=False
    )
    sizes = {int(c): int(n) for c, n in zip(*np.unique(model.labels_, return_counts=True))}
    metrics = {
        "k": int(best_k),
        "silhouette": round(float(silhouette_score(X, model.labels_)), 4),
        "ari": round(float(adjusted_rand_score(y, model.labels_)), 4),
        "cluster_sizes": sizes,
    }
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    log.info("Final model: %s", metrics)
    return metrics


def predict_samples(run_id: str) -> dict:
    """Predict one raw sample per cultivar using the saved scaler and the chosen-k model."""
    run_dir = _run_dir(run_id)
    raw = pd.read_csv(os.path.join(run_dir, "raw.csv"))
    features = _feature_columns(raw)
    scaler = joblib.load(os.path.join(run_dir, "scaler.joblib"))
    model = joblib.load(os.path.join(run_dir, "model.joblib"))

    # Name each cluster after the cultivar most of its members belong to.
    assign = pd.read_csv(os.path.join(run_dir, "assignments.csv"))
    majority = assign.groupby("cluster")[LABEL].agg(lambda s: int(s.mode()[0])).to_dict()

    samples = raw.groupby(LABEL).head(1)
    # Raw values go through the same scaler the model was trained with.
    clusters = model.predict(scaler.transform(samples[features]))
    results = []
    for true_label, cluster in zip(samples[LABEL], clusters):
        predicted = majority[int(cluster)]
        results.append({
            "true_cultivar": int(true_label),
            "cluster": int(cluster),
            "cluster_majority_cultivar": int(predicted),
            "correct": int(true_label) == int(predicted),
        })
        log.info("cultivar %d -> cluster %d (majority cultivar %d)", true_label, cluster, predicted)

    with open(os.path.join(run_dir, "predictions.json"), "w") as f:
        json.dump(results, f, indent=2)
    correct = sum(r["correct"] for r in results)
    return {"correct": correct, "total": len(results)}


def _png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def build_report(run_id: str) -> dict:
    """Write a self-contained HTML dashboard for this run."""
    run_dir = _run_dir(run_id)
    with open(os.path.join(run_dir, "tuning.json")) as f:
        tuning = json.load(f)
    with open(os.path.join(run_dir, "metrics.json")) as f:
        metrics = json.load(f)
    raw = pd.read_csv(os.path.join(run_dir, "raw.csv"))
    X = pd.read_csv(os.path.join(run_dir, "features_scaled.csv")).values
    assign = pd.read_csv(os.path.join(run_dir, "assignments.csv"))
    table = pd.DataFrame(tuning["table"])

    # 1. SSE curve with the elbow k and the silhouette k marked
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ax.plot(table["k"], table["sse"], marker="o")
    if tuning["elbow_k"]:
        ax.axvline(tuning["elbow_k"], linestyle="--", color="tab:orange", label=f"elbow k={tuning['elbow_k']}")
    ax.axvline(tuning["best_k"], linestyle="-", color="tab:green", label=f"silhouette k={tuning['best_k']}")
    ax.set(xlabel="k", ylabel="SSE (inertia)", title="Elbow curve")
    ax.legend()
    sse_png = _png(fig)

    # 2. Silhouette and ARI per k
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ax.plot(table["k"], table["silhouette"], marker="o", label="silhouette")
    ax.plot(table["k"], table["ari"], marker="s", label="ARI vs true cultivar")
    ax.axvline(tuning["best_k"], linestyle="-", color="tab:green")
    ax.set(xlabel="k", ylabel="score", title="Silhouette and ARI by k")
    ax.legend()
    score_png = _png(fig)

    # 3. Clusters projected to 2D with PCA, marker shape = true cultivar
    coords = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(X)
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "h", "p"]
    for cultivar in sorted(assign[LABEL].unique()):
        mask = assign[LABEL] == cultivar
        ax.scatter(coords[mask, 0], coords[mask, 1], c=assign.loc[mask, "cluster"], cmap="viridis",
                   vmin=0, vmax=max(metrics["k"] - 1, 1), marker=markers[cultivar % len(markers)],
                   edgecolor="black", linewidth=0.3, label=f"cultivar {cultivar}")
    ax.set(xlabel="PC1", ylabel="PC2", title="Clusters (color) vs true cultivar (shape)")
    ax.legend(fontsize=8)
    pca_png = _png(fig)

    # Cluster profiles in original units for a few interpretable features
    profile_cols = ["alcohol", "malic_acid", "flavanoids", "color_intensity", "proline"]
    profile = raw[profile_cols].assign(cluster=assign["cluster"]).groupby("cluster").mean().round(2)
    profile.insert(0, "size", assign.groupby("cluster").size())

    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Wine clustering report</title>
<style>
 body {{ font-family: Arial, sans-serif; margin: 24px; color: #222; }}
 .cards {{ display: flex; gap: 12px; flex-wrap: wrap; }}
 .card {{ border: 1px solid #ddd; border-radius: 8px; padding: 10px 16px; min-width: 140px; }}
 .card b {{ font-size: 22px; display: block; }}
 .plots {{ display: flex; gap: 12px; flex-wrap: wrap; margin-top: 16px; }}
 table {{ border-collapse: collapse; margin-top: 8px; }}
 td, th {{ border: 1px solid #ddd; padding: 4px 10px; text-align: right; }}
</style></head><body>
<h1>Wine clustering report</h1>
<p>Run: <code>{run_id}</code></p>
<div class="cards">
 <div class="card">Chosen k (silhouette)<b>{tuning['best_k']}</b></div>
 <div class="card">Elbow k<b>{tuning['elbow_k']}</b></div>
 <div class="card">Silhouette<b>{metrics['silhouette']}</b></div>
 <div class="card">ARI vs true cultivar<b>{metrics['ari']}</b></div>
 <div class="card">Samples<b>{len(raw)}</b></div>
</div>
<div class="plots">
 <img src="data:image/png;base64,{sse_png}">
 <img src="data:image/png;base64,{score_png}">
 <img src="data:image/png;base64,{pca_png}">
</div>
<h2>Cluster profiles (mean, original units)</h2>
{profile.to_html()}
<h2>All k tried</h2>
{table.to_html(index=False)}
</body></html>"""

    path = os.path.join(run_dir, "report.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    # Also keep a copy at a fixed path, so the latest report is always easy to find.
    with open(os.path.join(WORK_DIR, "latest_report.html"), "w", encoding="utf-8") as f:
        f.write(html)
    log.info("Report written to %s", path)
    return {"report": path}