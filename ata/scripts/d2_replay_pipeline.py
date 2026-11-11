"""Deliverable 2 - end-to-end software replay on HELD-OUT test streams (W/F saved artifact).
Pipeline per window: saved feature schema + imputation -> RF probabilities -> confidence gate -> saved decision policy
-> bounded reversible correction -> paper Eq.4 trust update -> logged row with latency.
No retraining, no threshold changes. Ground-truth columns are read ONLY after the pipeline has run, for scoring.

Run from the repo root after notebooks 04 and 06 (needs the saved NB6 model, feature table and predictions):
  python scripts/d2_replay_pipeline.py
Latency is specific to the machine that runs it.
"""
import argparse, json, platform, time, warnings
import numpy as np, pandas as pd, joblib
warnings.filterwarnings("ignore")
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]   # repo root (this script lives in scripts/)
ap = argparse.ArgumentParser()
ap.add_argument("--artifact", default=str(ROOT / "outputs/notebook06_v2/models/nb6_W_F.joblib"))
ap.add_argument("--feat", default=str(ROOT / "data/processed/feature_dataset_AF_v2.parquet"))
ap.add_argument("--pred", default=str(ROOT / "outputs/notebook06_v2/nb6_v2_predictions.parquet"))   # used only for a parity check
ap.add_argument("--out", default=str(ROOT / "outputs/deliverable2"))
ap.add_argument("--latency_windows", type=int, default=300)           # extra timing block with the saved n_jobs setting
a = ap.parse_args()
from pathlib import Path
OUT = Path(a.out); OUT.mkdir(parents=True, exist_ok=True)

# ---- declared, NOT tuned (paper gives no values for the Eq.4 step sizes) ----
ETA_D, ETA_R, T0 = 0.05, 0.01, 1.0
PROB_FLOOR = 1e-4   # NB6 config prob_floor

art = joblib.load(a.artifact)
MODEL, FEATS, CLASSES = art["model"], art["features"], list(art["classes"])
TAU, TCORR = art["tau"], art["t_corr"]
assert art["calibration_method"] == "raw"
MED = pd.Series(art["imputation_medians"])[FEATS].to_numpy(float)
SAVED_NJOBS = MODEL.n_jobs
IDX = {c: i for i, c in enumerate(CLASSES)}

T = pd.read_parquet(a.feat)
# train-only correction bound, same definition as NB7 (95th pct of |dev_mean_adj| on train windows)
BOUND = float(T[(T.split == "train") & T.dev_mean_adj.notna()].dev_mean_adj.abs().quantile(.95))

# ---------------- the pipeline (no ground truth inside) ----------------
def process(x_feat, dev_mean_adj, t_mean, trust, timing):
    t0 = time.perf_counter_ns()
    x = np.where(np.isnan(x_feat), MED, x_feat).reshape(1, -1)               # saved imputation
    t1 = time.perf_counter_ns()
    p = MODEL.predict_proba(x)[0]
    t2 = time.perf_counter_ns()
    p = np.maximum(p, PROB_FLOOR); p = p / p.sum()                            # NB6 prob floor
    k = int(np.argmax(p)); pred = CLASSES[k]; conf = float(p[k])
    gate = conf >= TAU
    if pred == "DRIFT" and p[IDX["DRIFT"]] >= TCORR: decision = "correct"     # saved policy: bypasses tau
    elif gate and pred == "NORMAL": decision = "no_action"
    elif gate and pred == "EVENT": decision = "preserve"
    else: decision = "defer_flag"
    est = applied = np.nan; corrected = t_mean
    if decision == "correct":
        if np.isfinite(dev_mean_adj):                                         # bounded, reversible (raw value kept)
            est = float(np.clip(dev_mean_adj, -BOUND, BOUND)); applied = est; corrected = t_mean - est
        else: decision_note = "correct_skipped_no_peer_evidence"
    # paper Eq.4 with gated attribution: y_hat = argmax if conf>=tau else UNCERTAIN
    y_hat = pred if gate else "UNCERTAIN"
    nxt = trust - ETA_D * conf * (y_hat in ("DRIFT", "EVENT_DRIFT")) + ETA_R * (y_hat == "NORMAL")
    nxt = float(min(1.0, max(0.0, nxt)))
    t3 = time.perf_counter_ns()
    timing.append(((t3 - t0) / 1e6, (t2 - t1) / 1e6))
    return dict(pred=pred, confidence=conf, **{"p_" + c: float(p[IDX[c]]) for c in CLASSES}, gate_pass=bool(gate),
                decision=decision, correction_applied=bool(np.isfinite(applied)), estimated_drift=est, raw_value=t_mean,
                corrected_value=corrected, trust_before=trust, trust_after=nxt, y_hat_gated=y_hat), nxt

def run(df, label):
    log, timing = [], []
    for (_, _), g in df.sort_values("window_start").groupby(["episode_id", "variant"], sort=True):
        trust = T0                                   # one trust state per sensor stream, reset between streams
        X = g[FEATS].to_numpy(float); dv = g.dev_mean_adj.to_numpy(float); tm = g.t_mean.to_numpy(float)
        for i in range(len(g)):
            r, trust = process(X[i], dv[i], tm[i], trust, timing)
            r.update(window_id=g.window_id.iloc[i], episode_id=g.episode_id.iloc[i], variant=g.variant.iloc[i])
            log.append(r)
    L = pd.DataFrame(log); L["latency_ms_end_to_end"] = [t[0] for t in timing]; L["latency_ms_model_only"] = [t[1] for t in timing]
    return L

def stats(v):
    v = np.asarray(v); return {"n": len(v), "median_ms": float(np.median(v)), "p95_ms": float(np.percentile(v, 95)), "p99_ms": float(np.percentile(v, 99))}

test = T[T.split == "test"].copy()
# warm-up (not logged)
_w = []
for i in range(20): process(test[FEATS].to_numpy(float)[i], np.nan, 0., T0, _w)

# main replay: n_jobs=1 (isolates inference cost from thread-pool overhead)
MODEL.set_params(n_jobs=1)
L = run(test, "n_jobs=1")
# secondary timing block with the artifact's saved n_jobs
MODEL.set_params(n_jobs=SAVED_NJOBS)
Ls = run(test.sort_values("window_start").head(a.latency_windows), f"n_jobs={SAVED_NJOBS}")
MODEL.set_params(n_jobs=SAVED_NJOBS)

# ---------------- scoring (truth used only here) ----------------
G = L.merge(test[["window_id", "label", "window_pure", "gt_event_component", "gt_drift_component", "minutes_since_onset"]], on="window_id")
G["applied"] = G.estimated_drift.fillna(0.0)
G["residual_drift"] = G.gt_drift_component - G.applied
G["event_distortion"] = np.sign(G.applied * G.gt_event_component) * np.minimum(G.applied.abs(), G.gt_event_component.abs())
G.to_csv(OUT / "replay_log.csv", index=False)
P = G[G.window_pure]                                          # same windows NB6/D1 evaluate

# parity with saved NB6 parquet (decision + probabilities)
sv = pd.read_parquet(a.pred); sv = sv[(sv.regime == "W") & (sv.set == "F") & (sv.part == "test")]
M = P.merge(sv[["window_id", "decision", "p_DRIFT", "p_EVENT_DRIFT"]], on="window_id", suffixes=("", "_saved"))
parity = {"windows_compared": len(M), "decision_match_rate": float((M.decision == M.decision_saved).mean()),
          "max_abs_prob_diff": float(max((M.p_DRIFT - M.p_DRIFT_saved).abs().max(), (M.p_EVENT_DRIFT - M.p_EVENT_DRIFT_saved).abs().max()))}

# checks of the safe-adaptation behaviours
checks = {
    "no_correction_when_deferred_or_preserved": bool((G.loc[G.decision.isin(["defer_flag", "preserve", "no_action"]), "correction_applied"] == False).all()),
    "every_applied_correction_within_bound": bool((G.loc[G.correction_applied, "estimated_drift"].abs() <= BOUND + 1e-12).all()),
    "raw_value_always_retained": bool(G.raw_value.notna().all()),
    "trust_within_0_1": bool(G.trust_after.between(0, 1).all()),
    "corrections_with_conf_below_tau": int(((G.decision == "correct") & ~G.gate_pass & G.window_pure).sum()),
}

shares = (P.groupby("label").decision.value_counts(normalize=True).unstack(fill_value=0).reindex(
    ["NORMAL", "DRIFT", "EVENT", "EVENT_DRIFT"]).reindex(columns=["no_action", "preserve", "correct", "defer_flag"], fill_value=0))
shares["n"] = P.groupby("label").size()
shares.to_csv(OUT / "behaviour_by_true_class.csv")
beh = P.groupby("label").agg(correction_applied_rate=("correction_applied", "mean"),
                              mean_abs_applied=("applied", lambda s: s.abs().mean()),
                              mean_abs_residual_drift=("residual_drift", lambda s: s.abs().mean()),
                              mean_abs_event_distortion=("event_distortion", lambda s: s.abs().mean())).reindex(["NORMAL", "DRIFT", "EVENT", "EVENT_DRIFT"])
beh.loc[["NORMAL", "EVENT"], "mean_abs_residual_drift"] = np.nan
beh.to_csv(OUT / "adaptation_by_true_class.csv")

fam = G.assign(stream=G.variant.map(lambda v: "NORMAL" if v == "NORMAL" else v.split("_")[0].replace("ED", "EVENT_DRIFT")))
final = G.sort_values("window_start" if "window_start" in G else "window_id").groupby(["episode_id", "variant"]).trust_after.last().reset_index()
final["stream"] = final.variant.map(lambda v: "NORMAL" if v == "NORMAL" else ("EVENT_DRIFT" if v.startswith("ED_") else ("DRIFT" if v.startswith("DRIFT") else "EVENT")))
trust = final.groupby("stream").trust_after.agg(["mean", "min", "max"]).round(3).reindex(["NORMAL", "DRIFT", "EVENT", "EVENT_DRIFT"])
trust.to_csv(OUT / "final_trust_by_stream_type.csv")

# example rows for the report/demo
ex = []
def pick(name, mask):
    z = P[mask]
    if len(z): r = z.iloc[0].to_dict(); r["example"] = name; ex.append(r)
pick("gate_bypass: corrected although confidence < tau (true DRIFT)", (P.decision == "correct") & ~P.gate_pass & (P.label == "DRIFT"))
pick("gate_bypass: corrected although confidence < tau (true EVENT_DRIFT)", (P.decision == "correct") & ~P.gate_pass & (P.label == "EVENT_DRIFT"))
pick("event_distortion: EVENT_DRIFT corrected with largest event distortion", P.event_distortion.abs().eq(P[(P.label == "EVENT_DRIFT") & P.correction_applied].event_distortion.abs().max()) & (P.label == "EVENT_DRIFT"))
pick("event preserved", (P.decision == "preserve") & (P.label == "EVENT"))
pick("normal, no action", (P.decision == "no_action") & (P.label == "NORMAL"))
pick("deferred", (P.decision == "defer_flag") & (P.label == "EVENT_DRIFT"))
EX = pd.DataFrame(ex)[["example", "window_id", "episode_id", "variant", "label", "pred", "confidence", "p_DRIFT", "decision", "gate_pass",
                       "raw_value", "corrected_value", "estimated_drift", "gt_event_component", "gt_drift_component", "event_distortion", "trust_before", "trust_after"]]
EX.to_csv(OUT / "demo_examples.csv", index=False)

summary = {
    "environment": {"platform": platform.platform(), "python": platform.python_version(), "processor": platform.processor(),
                    "logical_cpus": __import__("os").cpu_count(), "note": "Latency applies only to the machine that ran this script; a desktop/sandbox is not gateway evidence."},
    "artifact": {"path": a.artifact, "tau": TAU, "t_corr": TCORR, "saved_n_jobs": SAVED_NJOBS, "correction_bound_train_q95": BOUND},
    "declared_untuned_constants": {"eta_d": ETA_D, "eta_r": ETA_R, "T0": T0},
    "windows_replayed_total": len(G), "windows_scored_pure": len(P), "streams": int(G.groupby(["episode_id", "variant"]).ngroups),
    "parity_with_saved_nb6_parquet": parity, "behaviour_checks": checks,
    "latency_n_jobs_1_all_windows": {"end_to_end": stats(L.latency_ms_end_to_end), "model_only": stats(L.latency_ms_model_only)},
    f"latency_saved_n_jobs_{SAVED_NJOBS}_first_{len(Ls)}_windows": {"end_to_end": stats(Ls.latency_ms_end_to_end), "model_only": stats(Ls.latency_ms_model_only)},
}
try:
    import psutil; summary["process_rss_mb_after"] = psutil.Process().memory_info().rss / 2**20
except Exception: summary["process_rss_mb_after"] = None
json.dump(summary, open(OUT / "replay_summary.json", "w"), indent=2)

pd.set_option("display.width", 220, "display.max_columns", 30)
print(json.dumps({k: summary[k] for k in ["environment", "windows_replayed_total", "windows_scored_pure", "streams", "parity_with_saved_nb6_parquet", "behaviour_checks"]}, indent=1))
print("\nlatency:", json.dumps({k: v for k, v in summary.items() if k.startswith("latency")}, indent=1))
print("\nDecision share by TRUE class (pure test windows):"); print(shares.round(3).to_string())
print("\nCorrection effect by true class:"); print(beh.round(3).to_string())
print("\nFinal trust by stream type (mean/min/max):"); print(trust.to_string())
print("\nDemo examples:"); print(EX[["example", "label", "pred", "confidence", "decision", "gate_pass", "event_distortion", "trust_after"]].round(3).to_string(index=False))
