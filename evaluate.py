"""Evaluate an ISA-T checkpoint with PSNR / SSIM on a LOL test split."""
import argparse

import torch
from torch.utils.data import DataLoader

from isalux import ISALux
from isalux.data import PairedDataset
from isalux.evaluation import evaluate_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True)
    p.add_argument("--dataset", default="lolv1",
                   choices=["lolv1", "lolv2_real", "lolv2_syn"])
    p.add_argument("--weights", required=True)
    args = p.parse_args()

    model = ISALux().cuda()
    model.isat.load_state_dict(torch.load(args.weights, map_location="cuda"))
    dl = DataLoader(PairedDataset(args.data_root, args.dataset, "test"),
                    batch_size=1, num_workers=2)
    for gm in (False, True):
        psnr, ssim = evaluate_model(model, dl, "cuda", gt_mean=gm)
        print(f"{args.dataset} (gt_mean={gm}): PSNR {psnr:.2f}  SSIM {ssim:.4f}")


if __name__ == "__main__":
    main()
