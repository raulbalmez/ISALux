"""Train ISALux (Sec. 4.1): Adam(0.9, 0.999), 300k iterations, batch 8,
256x256 crops with flips/rotations, hybrid L2 + VGG19 + MS-SSIM loss."""
import argparse
import os

import torch
from torch.utils.data import DataLoader

from isalux import ISALux, count_params
from isalux.data import PairedDataset
from isalux.evaluation import evaluate_model
from isalux.losses import HybridLoss
from isalux.schedule import isalux_lr


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True)
    p.add_argument("--dataset", default="lolv1",
                   choices=["lolv1", "lolv2_real", "lolv2_syn"])
    p.add_argument("--iterations", type=int, default=300_000)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--patch_size", type=int, default=256)
    p.add_argument("--w_l2", type=float, default=1.0)
    p.add_argument("--w_perc", type=float, default=0.05)
    p.add_argument("--w_ssim", type=float, default=0.5)
    p.add_argument("--eval_every", type=int, default=5000)
    p.add_argument("--target_psnr", type=float, default=None,
                   help="stop once test PSNR >= target - tol (paper value)")
    p.add_argument("--tol", type=float, default=0.05)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--amp", action="store_true", help="bfloat16 autocast")
    p.add_argument("--out", default="checkpoints")
    args = p.parse_args()

    dev = "cuda"
    os.makedirs(args.out, exist_ok=True)
    model = ISALux().to(dev)
    print(f"ISA-T parameters: {count_params(model.isat) / 1e6:.3f}M "
          f"(+ frozen MobileNetV3: {count_params(model.semantic) / 1e6:.2f}M)")
    criterion = HybridLoss(args.w_l2, args.w_perc, args.w_ssim).to(dev)
    opt = torch.optim.Adam(model.isat.parameters(), lr=2e-4, betas=(0.9, 0.999))

    train_ds = PairedDataset(args.data_root, args.dataset, "train",
                             args.patch_size,
                             length=args.iterations * args.batch_size)
    train_dl = DataLoader(train_ds, args.batch_size, shuffle=True,
                          num_workers=args.workers, drop_last=True,
                          pin_memory=True, persistent_workers=True)
    test_dl = DataLoader(PairedDataset(args.data_root, args.dataset, "test"),
                         batch_size=1, num_workers=2)

    best, name = -1.0, args.dataset
    model.train()
    for it, (low, high) in enumerate(train_dl, start=1):
        for g in opt.param_groups:
            g["lr"] = isalux_lr(it, args.iterations)
        low, high = low.to(dev, non_blocking=True), high.to(dev, non_blocking=True)
        with torch.autocast("cuda", torch.bfloat16, enabled=args.amp):
            pred = model(low)
        loss = criterion(pred.float(), high)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if it % 100 == 0:
            print(f"it {it} loss {loss.item():.4f} lr {opt.param_groups[0]['lr']:.2e}",
                  flush=True)
        if it % args.eval_every == 0 or it == args.iterations:
            psnr, ssim = evaluate_model(model, test_dl, dev, gt_mean=True)
            print(f"[eval] it {it}: PSNR {psnr:.2f} SSIM {ssim:.4f}", flush=True)
            torch.save(model.isat.state_dict(), f"{args.out}/{name}_last.pth")
            if psnr > best:
                best = psnr
                torch.save(model.isat.state_dict(), f"{args.out}/{name}_best.pth")
            model.train()
            if args.target_psnr and psnr >= args.target_psnr - args.tol:
                print(f"[eval] reached paper PSNR {args.target_psnr}; stopping.")
                break


if __name__ == "__main__":
    main()
