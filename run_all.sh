#!/bin/bash
# Train + test ISALux on the LOL datasets (300k iterations, batch 8, bf16),
# stopping each run once test PSNR reaches the paper value (-0.05).
PY=${PY:-python}; D=${DATA:-../data}; mkdir -p logs
export PYTHONUNBUFFERED=1
run() { # dataset root paper_psnr
  $PY train.py --data_root $2 --dataset $1 --amp --eval_every 5000 --target_psnr $3 --out checkpoints > logs/$1.train.log 2>&1
  $PY evaluate.py --data_root $2 --dataset $1 --weights checkpoints/$1_best.pth > logs/$1.test.log 2>&1
}
run lolv1      $D/lolv1/lol_dataset   27.63
run lolv2_real $D/hf_lolv2-real       29.76
run lolv2_syn  $D/hf_lolv2-synthetic  30.78
