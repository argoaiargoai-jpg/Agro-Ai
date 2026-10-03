"""Train MobileNetV3-Small (transfer learning) on the 36 outputs.

    python train.py --name v1                 # full run
    python train.py --name smoke --bench      # speed test only
Works on Apple GPU (mps), CUDA (Colab) or CPU: the device is picked automatically.
"""
import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import models

from mldata import *  # noqa: F401,F403


def pick_device():
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


def build_model(n_classes: int, pretrained=True):
    m = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None)
    m.classifier[3] = nn.Linear(m.classifier[3].in_features, n_classes)
    return m


@torch.no_grad()
def predict_logits(model, loader, device):
    model.eval(); out, ys = [], []
    for x, y in loader:
        out.append(model(x.to(device)).float().cpu()); ys.append(y)
    return torch.cat(out).numpy(), torch.cat(ys).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="v1"); ap.add_argument("--epochs", type=int, default=14)
    ap.add_argument("--warm-epochs", type=int, default=2); ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--samples", type=int, default=20000, help="images drawn per epoch")
    ap.add_argument("--patience", type=int, default=3); ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--lr", type=float, default=6e-4); ap.add_argument("--pd-boost", type=float, default=3.0)
    ap.add_argument("--bench", action="store_true"); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", action="store_true", help="continue from runs/<name>/last.pt after a Colab disconnect")
    ap.add_argument("--smoke", action="store_true", help="tiny pipeline check (a few batches). Produces NO usable model.")
    a = ap.parse_args()
    if a.smoke: a.epochs, a.warm_epochs, a.samples, a.workers, a.bs = 2, 1, 128, 2, 32
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = pick_device(); print("device:", dev)
    run = RUNS / a.name; run.mkdir(parents=True, exist_ok=True)

    df = load_manifest(); classes = class_names(df); json.dump(classes, open(run / "classes.json", "w"), indent=1)
    tr = split_frame(df, "train"); va = split_frame(df, "val", exclude_novel_from_train=False)
    va = va[va.role != "unknown_novel"]      # novel crops are reserved for threshold calibration, not model selection
    if a.smoke: va = va.groupby("label_out", group_keys=False).head(4)
    train_ds = ImageSet(tr, classes, train_transform()); val_ds = ImageSet(va, classes, eval_transform())

    # Balanced sampling: sqrt-inverse class frequency (full balancing over-weights tiny classes), PlantDoc boosted.
    cnt = np.bincount(train_ds.y, minlength=len(classes)).astype(float)
    w = (1.0 / np.sqrt(cnt[train_ds.y])) * np.where(tr.source.values == "plantdoc", a.pd_boost, 1.0)
    sampler = WeightedRandomSampler(w, num_samples=a.samples, replacement=True)
    mk = dict(num_workers=a.workers, persistent_workers=a.workers > 0, **({"prefetch_factor": 4} if a.workers > 0 else {}))
    tl = DataLoader(train_ds, batch_size=a.bs, sampler=sampler, drop_last=True, **mk)
    vl = DataLoader(val_ds, batch_size=128, shuffle=False, **mk)
    print(f"train pool {len(train_ds)} | val {len(val_ds)} | classes {len(classes)}")
    print("sampled source mix (expected):", {s: round(float(w[tr.source.values == s].sum() / w.sum()), 3) for s in tr.source.unique()})

    model = build_model(len(classes)).to(dev)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    sched_total = a.epochs * (a.samples // a.bs)
    opt = None; best, bad, hist, start = -1, 0, [], 1
    amp = dev.type == "cuda"; scaler = torch.amp.GradScaler(enabled=amp)   # mixed precision: ~2x faster on a Colab T4
    saved = None
    if a.resume and (run / "last.pt").exists():
        saved = torch.load(run / "last.pt", map_location="cpu"); model.load_state_dict(saved["model"])
        best, bad, hist, start = saved["best"], saved["bad"], saved["hist"], saved["epoch"] + 1
        print(f"resuming after epoch {saved['epoch']} (best macro-F1 so far {best:.4f})")

    for ep in range(start, a.epochs + 1):
        warm = ep <= a.warm_epochs
        if opt is None or ep == a.warm_epochs + 1:     # phase A: head only | phase B: everything
            for p in model.features.parameters(): p.requires_grad = not warm
            params = [p for p in model.parameters() if p.requires_grad]
            opt = torch.optim.AdamW(params, lr=a.lr if warm else a.lr / 3, weight_decay=1e-4)
            steps = (a.warm_epochs if warm else a.epochs - a.warm_epochs) * (a.samples // a.bs)
            sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=opt.param_groups[0]["lr"], total_steps=steps, pct_start=0.15)
            if saved and saved["phase_warm"] == warm and saved["epoch"] + 1 == ep:        # resume mid-phase: restore optimizer + LR schedule
                opt.load_state_dict(saved["opt"]); sched.load_state_dict(saved["sched"])
        model.train(); t0 = time.time(); tot = 0; n = 0
        for it, (x, y) in enumerate(tl):
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=amp):
                loss = crit(model(x), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item() * len(y); n += len(y)
            if a.bench and it == 8: tb = time.time()
            if a.bench and it == 28: print(f"bench (steady state): {20*a.bs/(time.time()-tb):.0f} img/s"); return
        logits, yv = predict_logits(model, vl, dev)
        pred = logits.argmax(1); acc = float((pred == yv).mean()); f1 = float(f1_score(yv, pred, average="macro"))
        hist.append(dict(epoch=ep, phase="head" if warm else "full", train_loss=tot / n, val_acc=acc, val_macro_f1=f1, sec=time.time() - t0))
        print(f"ep {ep:2d} [{hist[-1]['phase']}] loss {tot/n:.4f} | val acc {acc:.4f} macroF1 {f1:.4f} | {time.time()-t0:.0f}s", flush=True)
        json.dump(hist, open(run / "history.json", "w"), indent=1)
        stop = False
        if f1 > best:
            best, bad = f1, 0; torch.save(dict(model=model.state_dict(), classes=classes, epoch=ep, val_macro_f1=f1, smoke=a.smoke), run / "best.pt")
        elif not warm:
            bad += 1
            stop = bad >= a.patience
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), epoch=ep, best=best, bad=bad, hist=hist, phase_warm=warm), run / "last.pt")
        if stop: print("early stopping"); break
    print("best val macro-F1:", round(best, 4))


if __name__ == "__main__":
    main()
