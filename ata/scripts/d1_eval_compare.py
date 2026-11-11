"""Deliverable 1 - held-out evaluation + comparison with baselines.
Uses ONLY saved artifacts (NB6 predictions, feature table, NB6 config). No retraining, no threshold changes.
Baselines follow NB7 definitions on dev_mean_adj (peer-median deviation). Test outcomes are never used to set parameters.

Usage: python d1_eval_compare.py [--pred P] [--feat F] [--cfg C] [--out DIR]
"""
import argparse, json, numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support, f1_score

ap = argparse.ArgumentParser()
U = "/mnt/user-data/uploads/"
ap.add_argument("--pred", default=U + "nb6_v2_predictions.parquet")
ap.add_argument("--feat", default=U + "feature_dataset_AF_v2.parquet")
ap.add_argument("--cfg", default=U + "nb6_v2_config.json")
ap.add_argument("--out", default="/mnt/user-data/outputs/d1")
a = ap.parse_args()
OUT = Path(a.out); OUT.mkdir(parents=True, exist_ok=True)

CLASSES = ["NORMAL", "DRIFT", "EVENT", "EVENT_DRIFT"]
T = pd.read_parquet(a.feat); P = pd.read_parquet(a.pred); cfg = json.load(open(a.cfg))
assert T.window_id.is_unique
meta = T[["window_id", "window_start", "dev_mean_adj"]]

# ---------- baseline detectors (NB7 definitions) ----------
def cusum(x, k, h):
    x = np.where(np.isfinite(x), x, 0.0); out = np.zeros(len(x), bool); pos = neg = 0.0
    for i, v in enumerate(x):
        pos = max(0., pos + v - k); neg = min(0., neg + v + k); out[i] = pos > h or neg < -h
    return out

def page_hinkley(x, delta, lam):
    x = np.where(np.isfinite(x), x, 0.0); out = np.zeros(len(x), bool); mean = cum = mn = 0.0
    for i, v in enumerate(x):
        mean += (v - mean) / (i + 1); cum += v - mean - delta; mn = min(mn, cum); out[i] = (cum - mn) > lam
    return out

def per_stream(df, fn):
    """Run a stateful detector separately on each (episode, variant) stream, in time order; reset between streams."""
    flag = pd.Series(False, index=df.index)
    for _, g in df.sort_values("window_start").groupby(["episode_id", "variant"]):
        x = g.dev_mean_adj.to_numpy(float); f = fn(x) & np.isfinite(x); flag.loc[g.index] = f
    return flag

def concat_run(df, fn):
    """NB7 protocol: one detector run over concatenated rows, no reset. Kept for traceability only."""
    x = df.dev_mean_adj.to_numpy(float); return pd.Series(fn(x) & np.isfinite(x), index=df.index)

# ---------- design set (train+validation episodes only) ----------
DEV = T[T.split.isin(["train", "validation"])].copy()
DEV["dev_abs"] = DEV.dev_mean_adj.abs()

def objective(flag, df):
    pure = df.label.isin(["NORMAL", "EVENT"]).to_numpy(); dr = df.label.eq("DRIFT").to_numpy(); f = np.asarray(flag)
    return (f[dr].mean() if dr.any() else 0.), (f[pure].mean() if pure.any() else 0.)

def choose_threshold(d):  # identical rule to NB7
    x = d.dev_abs.to_numpy(float); y = d.label.to_numpy(); v = np.isfinite(x)
    cand = np.unique(np.quantile(x[v], np.linspace(.05, .99, 40))); pure = np.isin(y, ["NORMAL", "EVENT"]); dr = y == "DRIFT"
    best = (-1., float(cand.max()))
    for th in cand:
        pr = v & (x >= th); u = pr[pure].mean(); r = pr[dr].mean()
        if u <= .05 and r > best[0]: best = (r, float(th))
    return best[1]
TH = choose_threshold(DEV)

# CUSUM / Page-Hinkley: scale-aware parameters picked on design streams with the SAME objective as the threshold baseline
sig = float(DEV[DEV.label.eq("NORMAL")].dev_mean_adj.std())
def tune(kind):
    best = (-1., None)
    for m in [1, 2, 4, 8, 16, 32, 64]:
        h = m * sig; k = .5 * sig
        fn = (lambda x, h=h, k=k: cusum(x, k, h)) if kind == "cusum" else (lambda x, h=h, k=k: page_hinkley(x, k, h))
        r, u = objective(per_stream(DEV, fn), DEV)
        if u <= .05 and r > best[0]: best = (r, h)
    return best[1] if best[1] is not None else 64 * sig
H_C, H_P = tune("cusum"), tune("ph")
print(f"design-only parameters: threshold={TH:.3f} | sigma_NORMAL={sig:.3f} | CUSUM h={H_C:.3f} | PH lambda={H_P:.3f}")

# ---------- evaluation helpers ----------
COND = cfg["conditions"]
def rates(z, flag):
    f = pd.Series(flag).loc[z.index]; r = {}
    for c in CLASSES:
        m = z.label.eq(c); r["corr_" + c] = f[m].mean() if m.any() else np.nan
    ev = z.label.isin(["EVENT", "EVENT_DRIFT"]); r["paper_FCR_event_windows"] = f[ev].mean()
    return r

def table(z, methods, tag):
    rows = []
    for name, flag in methods.items():
        flag = pd.Series(flag); row = {"method": name, "scope": tag, "n": len(z), **rates(z, flag)}
        eps = {e: rates(g, flag.loc[g.index]) for e, g in z.groupby("episode_id")}
        for k in ["corr_DRIFT", "paper_FCR_event_windows"]:
            v = [eps[e][k] for e in eps]; row[k + "_ep_min"], row[k + "_ep_max"] = min(v), max(v)
        rows.append(row)
    return rows

clf_rows, cmp_rows, ep_rows = [], [], []
for reg in ["W", "ALL"]:
    pr = P[(P.regime == reg) & (P.set == "F") & (P.part == "test")].merge(meta, on="window_id")
    pr["variant_"] = pr.variant
    tau = cfg["chosen"][f"{reg}/F"]["tau"]; tc = cfg["chosen"][f"{reg}/F"]["t_corr"]
    pr["dev_mean_adj"] = pr["dev_mean_adj"].astype(float)
    # detectors run on the FULL test streams (incl. transition windows), then scored on the NB6-evaluated windows
    full = T[T.split == "test"].copy()
    fl = {
        "fixed_threshold (peer-median dev)": pd.Series(np.isfinite(full.dev_mean_adj) & (full.dev_mean_adj.abs() >= TH), index=full.index),
        "CUSUM (design-tuned, per-stream)": per_stream(full, lambda x: cusum(x, .5 * sig, H_C)),
        "Page-Hinkley (design-tuned, per-stream)": per_stream(full, lambda x: page_hinkley(x, .5 * sig, H_P)),
        "CUSUM (NB7 fixed k=.05,h=.5; concatenated)": concat_run(full, lambda x: cusum(x, .05, .5)),
        "Page-Hinkley (NB7 fixed; concatenated)": concat_run(full, lambda x: page_hinkley(x, .005, .5)),
        "always_correct": pd.Series(np.isfinite(full.dev_mean_adj), index=full.index),
    }
    fl = {k: pd.Series(v.values, index=full.window_id) for k, v in fl.items()}
    methods = {
        "PROPOSED gated policy (saved decisions)": pr.decision.eq("correct"),
        "same classifier, no gate (argmax==DRIFT)": pr.pred.eq("DRIFT"),
    }
    for k, v in fl.items(): methods[k] = pd.Series(v.reindex(pr.window_id).to_numpy(), index=pr.index)
    methods["no_correction"] = pd.Series(False, index=pr.index)
    assert (pr.decision.eq("correct") == (pr.pred.eq("DRIFT") & (pr.p_DRIFT >= tc))).all(), "saved decisions != saved policy"

    # classification metrics (argmax class, pooled test) + per-episode macro-F1
    p, r_, f, s = precision_recall_fscore_support(pr.label, pr.pred, labels=CLASSES, zero_division=0)
    for c, a1, b1, c1, d1 in zip(CLASSES, p, r_, f, s):
        clf_rows.append({"regime": reg, "scope": "pooled_test", "class": c, "precision": a1, "recall": b1, "f1": c1, "support": d1})
    clf_rows.append({"regime": reg, "scope": "pooled_test", "class": "MACRO", "f1": f1_score(pr.label, pr.pred, labels=CLASSES, average="macro", zero_division=0), "support": len(pr)})
    for e, g in pr.groupby("episode_id"):
        clf_rows.append({"regime": reg, "scope": f"episode_{e}", "class": "MACRO", "f1": f1_score(g.label, g.pred, labels=CLASSES, average="macro", zero_division=0), "support": len(g)})
    cm = pd.crosstab(pr.label, pr.pred).reindex(index=CLASSES, columns=CLASSES, fill_value=0); cm.to_csv(OUT / f"confusion_{reg}_F.csv")

    # abstention + accuracy on accepted (proposed gate)
    acc = pr.confidence >= tau
    clf_rows.append({"regime": reg, "scope": "gate", "class": "coverage(conf>=tau)", "f1": acc.mean(), "support": int(acc.sum())})
    clf_rows.append({"regime": reg, "scope": "gate", "class": "accuracy_on_accepted", "f1": (pr.label[acc] == pr.pred[acc]).mean(), "support": int(acc.sum())})
    clf_rows.append({"regime": reg, "scope": "gate", "class": "defer_rate", "f1": pr.decision.eq("defer_flag").mean(), "support": len(pr)})

    # comparison: pooled + per condition
    for r in table(pr, methods, "pooled"): r["regime"] = reg; cmp_rows.append(r)
    for cn, variants in COND.items():
        z = pr[pr.variant.isin(variants)]
        for r in table(z, methods, cn): r["regime"] = reg; cmp_rows.append(r)

CLF = pd.DataFrame(clf_rows); CMP = pd.DataFrame(cmp_rows)
CLF.to_csv(OUT / "d1_classification_gate.csv", index=False); CMP.to_csv(OUT / "d1_comparison.csv", index=False)
cols = ["method", "corr_DRIFT", "corr_NORMAL", "corr_EVENT", "corr_EVENT_DRIFT", "paper_FCR_event_windows"]
pd.set_option("display.width", 250, "display.max_columns", 30)
for reg in ["W", "ALL"]:
    print(f"\n===== regime {reg}, feature set F, pooled test (n={int(CMP[(CMP.regime==reg)&(CMP.scope=='pooled')].n.iloc[0])}) =====")
    print(CMP[(CMP.regime == reg) & (CMP.scope == "pooled")][cols].round(3).to_string(index=False))
    for cn in ["cool_ramp", "cool_step"]:
        print(f"--- {reg} / {cn}"); print(CMP[(CMP.regime == reg) & (CMP.scope == cn)][cols].round(3).to_string(index=False))
print("\n===== classification / gate =====")
print(CLF[CLF["class"].isin(CLASSES + ["MACRO"]) | (CLF.scope == "gate")].round(3).to_string(index=False))
