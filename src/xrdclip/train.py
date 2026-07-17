"""Contrastive training loop for XRD-CLIP.

Runs on the Vast RTX 5090 for real; runs on CPU for the smoke test (`--smoke`). The overfit
sanity check (SPEC §6, week 1) is built in as `--overfit N`: on a tiny fixed subset with no
augmentation, in-batch top-1 must reach ~1.0, or the model/plumbing is broken and scaling up is
pointless.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from xrdclip.data.dataset import ContrastiveXRDDataset, StickStore, collate
from xrdclip.data import augment as A
from xrdclip.models.clip import XRDClip, batch_topk_accuracy, clip_loss


def load_refcodes(path: str) -> "list[str]":
    with open(path) as fh:
        return [ln.strip() for ln in fh if ln.strip()]


def build_loader(store, refcodes, batch_size, augment, workers, shuffle=True, ranges=A.Ranges(),
                 batch_sampler=None):
    ds = ContrastiveXRDDataset(store, refcodes, augment=augment, ranges=ranges)
    if batch_sampler is not None:
        # batch_sampler already fixes batch composition + size; batch_size/shuffle/drop_last are
        # mutually exclusive with it.
        loader = DataLoader(ds, batch_sampler=batch_sampler, num_workers=workers,
                            collate_fn=collate, pin_memory=torch.cuda.is_available())
    else:
        loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
                            collate_fn=collate, drop_last=True, pin_memory=torch.cuda.is_available())
    return loader, ds


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sticks", default="cache/sticks")
    ap.add_argument("--train_refs", default="refcodes/scaffold_train.txt")
    ap.add_argument("--val_refs", default="refcodes/scaffold_val.txt")
    ap.add_argument("--out", default="runs/xrdclip")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=0.05)
    ap.add_argument("--workers", type=int, default=8, help="cap this on many-core Vast boxes")
    ap.add_argument("--embed_dim", type=int, default=256)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--poly_families", default="",
                    help="polymorph family map (e.g. refcodes/polymorph_families.json); when set, "
                         "co-locate sibling polymorphs in each batch as hard negatives (pivot)")
    ap.add_argument("--smoke", action="store_true", help="tiny CPU run to verify the loop end to end")
    ap.add_argument("--overfit", type=int, default=0, help="overfit N structures, no aug (sanity)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(0)

    if args.smoke:
        args.epochs, args.batch_size, args.workers = 2, 16, 0

    # --- data ---
    train_sampler = None
    if args.overfit:
        store = StickStore(args.sticks)
        refs = store.refcodes[: args.overfit]
        train_loader, _ = build_loader(store, refs, min(args.batch_size, len(refs)),
                                       augment=False, workers=args.workers)
        val_loader = None
        print(f"[train] OVERFIT MODE: {len(refs)} structures, no augmentation")
    else:
        train_refs = load_refcodes(args.train_refs)
        val_refs = load_refcodes(args.val_refs)
        if args.smoke:
            train_refs, val_refs = train_refs[:200], val_refs[:64]
        need = set(train_refs) | set(val_refs)
        store = StickStore(args.sticks, refcodes=need)
        if args.poly_families:
            from xrdclip.data.polymorph import load_families, PolymorphBatchSampler
            fams = load_families(args.poly_families)
            train_sampler = PolymorphBatchSampler(train_refs, fams, args.batch_size, seed=0)
            print(f"[train] polymorph hard-negative batches: {train_sampler.n_families:,} families / "
                  f"{train_sampler.n_family_structs:,} structs, {len(train_sampler):,} batches/epoch")
            train_loader, _ = build_loader(store, train_refs, args.batch_size, True, args.workers,
                                           batch_sampler=train_sampler)
        else:
            train_loader, _ = build_loader(store, train_refs, args.batch_size, True, args.workers)
        val_loader, _ = build_loader(store, val_refs, min(args.batch_size, len(val_refs)),
                                     False, args.workers, shuffle=False)
        print(f"[train] {len(train_refs):,} train / {len(val_refs):,} val on {device}")

    # --- model ---
    model = XRDClip(embed_dim=args.embed_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] model has {n_params/1e6:.1f}M params")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps = max(1, len(train_loader))
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=steps,
        pct_start=min(0.3, args.warmup / max(args.epochs, 1)),
    )

    history = []
    best_val = -1.0
    for epoch in range(args.epochs):
        model.train()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)      # fresh polymorph-aware shuffle each epoch
        t0, losses, accs = time.time(), [], []
        for batch in train_loader:
            pattern = batch["pattern"].to(device)
            az, af = batch["atom_z"].to(device), batch["atom_frac"].to(device)
            lat, mask = batch["lattice"].to(device), batch["mask"].to(device)

            logits, _, _ = model(pattern, az, af, lat, mask)
            loss = clip_loss(logits)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()

            losses.append(loss.item())
            accs.append(batch_topk_accuracy(logits)["top1"])

        tr_loss, tr_acc = float(np.mean(losses)), float(np.mean(accs))

        val_acc = float("nan")
        if val_loader is not None:
            val_acc = evaluate_inbatch(model, val_loader, device)
            if val_acc > best_val:
                best_val = val_acc
                torch.save({"model": model.state_dict(), "args": vars(args), "epoch": epoch},
                           os.path.join(args.out, "best.pt"))

        dt = time.time() - t0
        print(f"[e{epoch:03d}] loss={tr_loss:.4f} train_top1={tr_acc:.3f} "
              f"val_top1={val_acc:.3f} tau={1/model.log_temp.exp().item():.3f} ({dt:.1f}s)")
        history.append({"epoch": epoch, "loss": tr_loss, "train_top1": tr_acc, "val_top1": val_acc})

        if args.overfit and tr_acc > 0.99:
            print(f"[train] overfit reached top1={tr_acc:.3f} at epoch {epoch} — plumbing OK")
            break

    torch.save({"model": model.state_dict(), "args": vars(args)}, os.path.join(args.out, "last.pt"))
    with open(os.path.join(args.out, "history.json"), "w") as fh:
        json.dump(history, fh, indent=2)
    print(f"[train] done. best val_top1={best_val:.3f}. wrote {args.out}/")
    return 0


@torch.no_grad()
def evaluate_inbatch(model, loader, device) -> float:
    model.eval()
    accs = []
    for batch in loader:
        logits, _, _ = model(
            batch["pattern"].to(device), batch["atom_z"].to(device),
            batch["atom_frac"].to(device), batch["lattice"].to(device), batch["mask"].to(device),
        )
        accs.append(batch_topk_accuracy(logits)["top1"])
    return float(np.mean(accs)) if accs else float("nan")


if __name__ == "__main__":
    raise SystemExit(main())
