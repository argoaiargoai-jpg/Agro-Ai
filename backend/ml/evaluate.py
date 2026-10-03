"""Evaluate a trained run on the HELD-OUT data, one dataset at a time (never blended).

    python evaluate.py logits    --run v1      # run the model once on val+test sets, cache the raw scores
    python evaluate.py calibrate --run v1      # temperature + thresholds, from VALIDATION scores only
    python evaluate.py report    --run v1      # final numbers on TEST sets -> reports/<run>/
"""
import argparse
import json

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch.optim import LBFGS
from torch.utils.data import DataLoader

import decision
from mldata import *  # noqa: F401,F403
from train import build_model, pick_device, predict_logits


def cmd_logits(run: str, workers: int = 4, limit: int = 0):
    ck = torch.load(RUNS / run / "best.pt", map_location="cpu"); classes = ck["classes"]
    dev = pick_device(); model = build_model(len(classes), pretrained=False); model.load_state_dict(ck["model"]); model.to(dev)
    df = load_manifest(); df = df[df.split.isin(["val", "test"])]
    if limit: df = df.groupby(["source", "split", "role"], group_keys=False).head(limit)   # smoke tests only
    df = df.reset_index(drop=True); ds = ImageSet(df, classes, eval_transform())
    logits, y = predict_logits(model, DataLoader(ds, batch_size=128, num_workers=workers), dev)
    np.savez(RUNS / run / "logits.npz", logits=logits, y=y)
    df.to_csv(RUNS / run / "eval_manifest.csv", index=False); print("saved", logits.shape, "from epoch", ck["epoch"])


def load_scores(run):
    z = np.load(RUNS / run / "logits.npz"); df = pd.read_csv(RUNS / run / "eval_manifest.csv")
    return z["logits"], z["y"], df, json.load(open(RUNS / run / "classes.json"))


# ------------------------------------------------------------------ calibration (validation only)
def fit_temperature(logits, y):
    lg = torch.tensor(logits, dtype=torch.float32); yt = torch.tensor(y)
    logT = torch.zeros(1, requires_grad=True); opt = LBFGS([logT], lr=0.1, max_iter=100)
    nll = torch.nn.CrossEntropyLoss()
    def closure():
        opt.zero_grad(); l = nll(lg / logT.exp(), yt); l.backward(); return l
    opt.step(closure); return float(logT.exp())


def ece(prob, y, bins=15):
    conf, pred = prob.max(1), prob.argmax(1); acc = (pred == y).astype(float); e = 0.0
    for lo in np.linspace(0, 1, bins, endpoint=False):
        m = (conf > lo) & (conf <= lo + 1 / bins)
        if m.any(): e += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(e)


def cmd_calibrate(run: str):
    logits, y, df, classes = load_scores(run)
    val = (df.split == "val").values
    i_unk, i_np = classes.index(decision.UNKNOWN), classes.index(decision.NON_PLANT)
    known = val & (df.role == "known").values & (df.source != "coco").values
    # 1) temperature on validation known samples (domain-balanced: PlantDoc counts as much as PlantVillage)
    pv, pdoc = known & (df.source == "plantvillage").values, known & (df.source == "plantdoc").values
    T = fit_temperature(np.concatenate([logits[pv], logits[pdoc]]), np.concatenate([y[pv], y[pdoc]]))
    p_raw, p_cal = decision.softmax(logits[known]), decision.softmax(logits[known], T)
    cal = dict(ece_before=ece(p_raw, y[known]), ece_after=ece(p_cal, y[known]), temperature=T)
    print(f"temperature {T:.3f} | ECE {cal['ece_before']:.4f} -> {cal['ece_after']:.4f}")
    P = decision.softmax(logits, T)

    # 2) NO_PLANT threshold: smallest threshold that wrongly calls <=1% of REAL PLANT validation images "not a plant"
    plant_val = val & np.isin(df.source, ["plantvillage", "plantdoc"])
    coco_val = val & (df.source == "coco").values
    cands = np.linspace(0.30, 0.995, 140)
    # false 'not a plant' must be low in BOTH domains (lab and field), so take the worse of the two
    def fpr(t, dom): m = plant_val & (df.domain == dom).values; return float((P[m][:, i_np] >= t).mean())
    ok = [t for t in cands if max(fpr(t, "lab"), fpr(t, "field")) <= 0.01]
    t_np = float(ok[0]) if ok else 0.9
    np_recall = float((P[coco_val][:, i_np] >= t_np).mean())
    print(f"t_non_plant {t_np:.3f} -> val recall on COCO {np_recall:.3f}; false NO_PLANT lab {fpr(t_np,'lab'):.4f} field {fpr(t_np,'field'):.4f}")

    # 3) confidence threshold: maximise Youden J = (known accepted-and-correct) - (novel unknown wrongly accepted)
    unk_novel = val & (df.role == "unknown_novel").values
    def outcome_known(t, m):
        r = decision.decide(logits[m], classes, dict(temperature=T, t_non_plant=t_np, t_confidence=t))
        return np.mean([(o["cls"] is not None) and classes.index(o["cls"]) == yy for o, yy in zip(r, y[m])])
    def accepted_unknown(t):
        r = decision.decide(logits[unk_novel], classes, dict(temperature=T, t_non_plant=t_np, t_confidence=t))
        return np.mean([o["outcome"] in ("DISEASE", "HEALTHY") for o in r])
    rows = []
    for t in np.round(np.arange(0.20, 0.991, 0.02), 3):
        lab, fld, acc_unk = outcome_known(t, pv), outcome_known(t, pdoc), accepted_unknown(t)
        j = 0.5 * (lab + fld) - acc_unk                       # domain-balanced known TPR minus unknown FPR
        rows.append(dict(t=float(t), known_lab=float(lab), known_field=float(fld), unknown_wrongly_accepted=float(acc_unk), youden_J=float(j)))
    best = max(rows, key=lambda r: r["youden_J"])
    print("threshold sweep (validation):")
    for r in rows[::4]: print("  t=%.2f lab-correct %.3f field-correct %.3f unknown-accepted %.3f J %.3f" % (r["t"], r["known_lab"], r["known_field"], r["unknown_wrongly_accepted"], r["youden_J"]))
    print("CHOSEN t_confidence", best["t"], best)
    cfg = dict(temperature=T, t_non_plant=t_np, t_confidence=best["t"], method=dict(
        temperature="LBFGS on validation NLL (PlantVillage+PlantDoc known)", t_non_plant="lowest threshold with <=1% false NO_PLANT on validation plant images (both domains)",
        t_confidence="max Youden J on validation: domain-balanced known-correct rate minus novel-unknown (Raspberry) accepted rate"),
        calibration=cal, validation_sweep=rows)
    json.dump(cfg, open(RUNS / run / "thresholds.json", "w"), indent=1)


# ------------------------------------------------------------------ report (test sets, separately)
def wilson(k, n, z=1.96):
    if n == 0: return (None, None)
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(c - h, 4), round(c + h, 4))


def closed_set_metrics(df, logits, y, classes, cfg, mask, name, outdir):
    """Known-class test: raw argmax (36-way) and the FINAL pipeline (rejections count as wrong)."""
    idx = np.where(mask)[0]; n = len(idx); K = len(classes) - 2
    raw = logits[idx].argmax(1); acc_raw = float((raw == y[idx]).mean())
    res = decision.decide(logits[idx], classes, cfg)
    final_cls = np.array([classes.index(r["cls"]) if r["cls"] else -1 for r in res])
    acc_final = float((final_cls == y[idx]).mean()); coverage = float((final_cls >= 0).mean())
    acc_cov = float((final_cls[final_cls >= 0] == y[idx][final_cls >= 0]).mean()) if coverage else None
    hd_true = np.array([classes[t].endswith("___healthy") for t in y[idx]])
    hd_pred = np.array([r["outcome"] == "HEALTHY" for r in res]); answered = final_cls >= 0
    hd_acc = float((hd_true[answered] == hd_pred[answered]).mean()) if answered.any() else None
    labels = list(range(K)); p, r, f, s = precision_recall_fscore_support(y[idx], raw, labels=labels, zero_division=0)
    present = s > 0; cm = confusion_matrix(y[idx], raw, labels=labels)
    per = {classes[i]: dict(precision=round(float(p[i]), 4), recall=round(float(r[i]), 4), f1=round(float(f[i]), 4), support=int(s[i])) for i in labels if present[i]}
    off = [(int(cm[i, j]), classes[i], classes[j]) for i in labels for j in range(len(labels)) if i != j and cm[i, j] > 0]
    off.sort(reverse=True)
    wp, wr, wf, _ = precision_recall_fscore_support(y[idx], raw, labels=[i for i in labels if present[i]], average="weighted", zero_division=0)
    mp, mr, mf, _ = precision_recall_fscore_support(y[idx], raw, labels=[i for i in labels if present[i]], average="macro", zero_division=0)
    out = dict(dataset=name, n=n, accuracy_raw_argmax=round(acc_raw, 4), accuracy_ci95=wilson(int((raw == y[idx]).sum()), n),
               accuracy_final_pipeline=round(acc_final, 4), pipeline_coverage=round(coverage, 4), accuracy_when_answered=None if acc_cov is None else round(acc_cov, 4),
               healthy_vs_disease_accuracy_when_answered=None if hd_acc is None else round(hd_acc, 4),
               precision_macro=round(float(mp), 4), recall_macro=round(float(mr), 4), f1_macro=round(float(mf), 4),
               precision_weighted=round(float(wp), 4), recall_weighted=round(float(wr), 4), f1_weighted=round(float(wf), 4),
               rejected_as=dict(__import__("collections").Counter(r["outcome"] for r in res if r["cls"] is None)),
               per_class=per, top_confusions=[dict(count=c, true=a, predicted=b) for c, a, b in off[:12]],
               best_classes=sorted(per, key=lambda k: (-per[k]["f1"], -per[k]["support"]))[:5], weakest_classes=sorted(per, key=lambda k: per[k]["f1"])[:6])
    save_cm(cm, [classes[i] for i in labels], f"{name} confusion matrix (n={n})", outdir / f"confusion_{name}.png", present)
    return out


def save_cm(cm, names, title, path, present):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    keep = np.where(present)[0]; cm = cm[np.ix_(keep, keep)]; names = [names[i] for i in keep]
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(13, 11)); im = ax.imshow(norm, cmap="Greens", vmin=0, vmax=1)
    short = [n.replace("_(maize)", "").replace("_(including_sour)", "").replace("___", ": ")[:30] for n in names]
    ax.set_xticks(range(len(short))); ax.set_xticklabels(short, rotation=90, fontsize=7); ax.set_yticks(range(len(short))); ax.set_yticklabels(short, fontsize=7)
    ax.set_xlabel("predicted"); ax.set_ylabel("true (row-normalised)"); ax.set_title(title); fig.colorbar(im, fraction=0.03); fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def outcome_counts(df, logits, cfg, classes, mask):
    res = decision.decide(logits[mask], classes, cfg); n = len(res)
    c = __import__("collections").Counter(r["outcome"] for r in res)
    return dict(n=n, **{k: round(v / n, 4) for k, v in c.items()}, counts=dict(c))


def cmd_report(run: str):
    logits, y, df, classes = load_scores(run); cfg = json.load(open(RUNS / run / "thresholds.json"))
    outdir = REPORTS / run; outdir.mkdir(parents=True, exist_ok=True)
    te = (df.split == "test").values; K = len(classes) - 2
    R = dict(run=run, thresholds={k: cfg[k] for k in ("temperature", "t_non_plant", "t_confidence")}, calibration=cfg["calibration"], note="Each dataset is reported on its own; no blended accuracy.")
    pv_known = te & (df.source == "plantvillage").values & (df.role == "known").values
    R["A_plantvillage_test_known_classes"] = closed_set_metrics(df, logits, y, classes, cfg, pv_known, "plantvillage", outdir)
    # leakage audit: images covered by the official leaf-map vs images grouped heuristically
    mapped = pv_known & df.leaf.notna().values
    for tag, m in (("leaf_map_covered", mapped), ("not_in_leaf_map", pv_known & ~mapped)):
        R["A_plantvillage_test_known_classes"][f"accuracy_raw_{tag}"] = round(float((logits[m].argmax(1) == y[m]).mean()), 4) if m.any() else None
        R["A_plantvillage_test_known_classes"][f"n_{tag}"] = int(m.sum())
    pd_known = te & (df.source == "plantdoc").values & (df.role == "known").values
    R["B_plantdoc_test_real_world_known_classes"] = closed_set_metrics(df, logits, y, classes, cfg, pd_known, "plantdoc", outdir)
    co = te & (df.source == "coco").values
    R["C_coco_test_non_plant"] = outcome_counts(df, logits, cfg, classes, co)
    R["C_coco_test_non_plant"]["no_plant_recall"] = R["C_coco_test_non_plant"].get("NO_PLANT", 0.0)
    plants = te & np.isin(df.source, ["plantvillage", "plantdoc"]) & (df.role == "known").values
    R["C_false_no_plant_rate_on_real_plant_test_images"] = {d: round(float((decision.softmax(logits[plants & (df.source == d).values], cfg["temperature"])[:, classes.index(decision.NON_PLANT)] >= cfg["t_non_plant"]).mean()), 4) for d in ("plantvillage", "plantdoc")}
    U = {}
    for name, m in (("plantvillage_Blueberry_novel_crop", te & (df.source == "plantvillage").values & (df.role == "unknown_novel").values),
                    ("plantvillage_Orange_Squash_seen_unsupported_crops", te & (df.source == "plantvillage").values & (df.role == "unknown_seen").values),
                    ("plantdoc_unsupported_crops(Blueberry/Raspberry novel+Squash)", te & (df.source == "plantdoc").values & df.role.isin(["unknown_novel", "unknown_seen"]).values)):
        o = outcome_counts(df, logits, cfg, classes, m); o["correctly_refused(UNKNOWN or NO_PLANT)"] = round(1 - o.get("DISEASE", 0) - o.get("HEALTHY", 0), 4); U[name] = o
    R["D_unknown_unsupported_plants"] = U
    json.dump(R, open(outdir / "metrics.json", "w"), indent=1)
    for k in ("A_plantvillage_test_known_classes", "B_plantdoc_test_real_world_known_classes"):
        r = R[k]; print(f"\n== {k}  n={r['n']}\n  raw 36-way argmax acc {r['accuracy_raw_argmax']} CI95 {r['accuracy_ci95']} | macro P/R/F1 {r['precision_macro']}/{r['recall_macro']}/{r['f1_macro']}")
        print(f"  final pipeline acc {r['accuracy_final_pipeline']} (coverage {r['pipeline_coverage']}, acc when answered {r['accuracy_when_answered']}) | healthy-vs-disease {r['healthy_vs_disease_accuracy_when_answered']}")
        print("  weakest:", r["weakest_classes"], "\n  top confusions:", r["top_confusions"][:4])
    print("\n== A leakage audit", {k: v for k, v in R["A_plantvillage_test_known_classes"].items() if "leaf_map" in k})
    print("\n== C COCO NO_PLANT", R["C_coco_test_non_plant"], "\n   false NO_PLANT on real plants", R["C_false_no_plant_rate_on_real_plant_test_images"])
    for k, v in U.items(): print("== D", k, v)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["logits", "calibrate", "report"]); ap.add_argument("--run", default="v1")
    ap.add_argument("--workers", type=int, default=4); ap.add_argument("--limit", type=int, default=0, help="smoke test: use only N images per group")
    a = ap.parse_args()
    if a.cmd == "logits": cmd_logits(a.run, a.workers, a.limit)
    else: {"calibrate": cmd_calibrate, "report": cmd_report}[a.cmd](a.run)
