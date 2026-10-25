<div align="center">

# ISALux

### Illumination and Semantics-Aware Transformer Employing Mixture of Experts for Low Light Image Enhancement

Raul Balmez, Alexandru Brateanu, Ciprian Orhei, Codruta Ancuti, Cosmin Ancuti

[![WACV 2026](https://img.shields.io/badge/WACV-2026-1f6feb.svg?style=flat-square)](https://openaccess.thecvf.com/content/WACV2026/html/Balmez_ISALux_Illumination_and_Semantics-Aware_Transformer_Employing_Mixture_of_Experts_for_WACV_2026_paper.html)
[![arXiv](https://img.shields.io/badge/arXiv-2508.17885-b31b1b.svg?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2508.17885)
[![Project Page](https://img.shields.io/badge/Project-Page-0a7d5a.svg?style=flat-square)](https://raulbalmez.github.io/ISALux/)
[![Python](https://img.shields.io/badge/Python-3.13-3776AB.svg?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.14-EE4C2C.svg?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-PolyForm%20NC%201.0.0-orange.svg?style=flat-square)](LICENSE)

[**Paper (WACV 2026)**](https://openaccess.thecvf.com/content/WACV2026/html/Balmez_ISALux_Illumination_and_Semantics-Aware_Transformer_Employing_Mixture_of_Experts_for_WACV_2026_paper.html) · [**Project Page**](https://raulbalmez.github.io/ISALux/) · [**Supplementary**](https://openaccess.thecvf.com/content/WACV2026/supplemental/Balmez_ISALux_Illumination_and_WACV_2026_supplemental.pdf) · [**Checkpoints**](#-checkpoints) · [**Results**](#-results) · [**Citation**](#-citation)

</div>

---

**ISALux** is a transformer for low-light image enhancement (LLIE) that injects illumination and semantic priors directly into self-attention, and uses a sparse mixture-of-experts feed-forward network to specialise in different low-light degradations.

## 📌 Highlights

- **HISA-MSA** — Hybrid Illumination and Semantics-Aware Multi-Headed Self-Attention: two parallel attention modules whose outputs are modulated by an illumination map and a semantic segmentation map, then fused with learnable weights.
- **LoRA on Q, K, V** — low-rank adaptations inside the attention block to reduce overfitting to dataset-specific light patterns.
- **MoE-FFN** — a sparse mixture of 4 CNN experts with a gating network that activates the top 2 per image.
- **Efficient** — **1.85M parameters** (plus a frozen 11.03M MobileNetV3 segmenter) and **105 ms** per 400×600 image.
- **State of the art (2024 – early 2025)** — the highest PSNR among the compared methods on LOL-v2-real, LOL-v2-synthetic, SDSD-in, SDSD-out and LOL-Blur, with a **30.08 dB / 0.910** average over LOL and SDSD.
- **Ranking** — based on self-reported numbers from recent papers, ISALux has the highest PSNR on LOL-v2-real and LOL-v2-synthetic under ground-truth-mean alignment, and on SDSD-in, SDSD-out and LOL-Blur among frame-based methods. Many 2025 – 2026 papers report only unaligned scores, which are not directly comparable.

## 📊 Results

### Low-Light Image Enhancement

| Method | Params (M) ↓ | LOL-v1 PSNR / SSIM ↑ | LOL-v2-real PSNR / SSIM ↑ | LOL-v2-syn PSNR / SSIM ↑ | SDSD-in PSNR / SSIM ↑ | SDSD-out PSNR / SSIM ↑ | Average PSNR / SSIM ↑ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Restormer | 26.1 | 22.43 / 0.823 | 19.94 / 0.827 | 21.41 / 0.830 | 25.67 / 0.827 | 24.79 / 0.802 | 22.85 / 0.822 |
| SNR-Net | 4.01 | 24.61 / 0.842 | 21.48 / 0.849 | 24.14 / 0.928 | 29.44 / 0.894 | 28.66 / 0.866 | 25.87 / 0.876 |
| LLFlow | 37.7 | 25.13 / 0.872 | 26.20 / 0.888 | 24.80 / 0.919 | – | – | 25.38 / 0.893 |
| LLFormer | 24.6 | 25.75 / 0.823 | 26.19 / 0.819 | 28.00 / 0.927 | – | – | 26.64 / 0.856 |
| LL-SKF | 39.91 | 26.79 / 0.879 | 28.45 / 0.905 | 29.11 / 0.953 | – | – | 28.12 / **0.912** |
| Retinexformer | **1.61** | 27.18 / 0.850 | 27.71 / 0.856 | 29.04 / 0.939 | 29.77 / 0.896 | 29.84 / 0.877 | 28.71 / 0.884 |
| GLARE | 44.04 | 27.35 / **0.883** | 28.98 / 0.905 | 29.84 / **0.958** | 30.10 / 0.896 | 30.85 / 0.884 | 29.42 / 0.905 |
| **ISALux (ours)** | 1.85 | **27.63** / 0.881 | **29.76 / 0.908** | **30.78** / 0.956 | **30.67 / 0.909** | **31.58 / 0.895** | **30.08** / 0.910 |

<details>
<summary><b>LOL-Blur and no-reference results</b></summary>

| Method | LOL-Blur PSNR / SSIM ↑ |
| :--- | :---: |
| FourierDiff | 20.53 / 0.710 |
| DeblurGAN-v2 | 22.30 / 0.745 |
| Retinexformer | 22.90 / 0.824 |
| LEDNet | 25.74 / 0.850 |
| ASP-LED | 26.73 / 0.866 |
| VQCNIR | 27.79 / 0.875 |
| **ISALux (ours)** | **28.01 / 0.903** |

| Method | MEF | LIME | DICM | NPE | Avg NIQE ↓ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| SNR-Net | 4.14 | 5.51 | 4.62 | 4.36 | 4.63 |
| LLFlow | 3.92 | 5.29 | 3.78 | 4.16 | 4.25 |
| LL-SKF | 4.03 | 5.15 | 3.70 | 4.08 | 4.24 |
| RFR | 3.92 | **3.81** | 3.75 | 4.13 | 3.92 |
| GLARE | 3.66 | 4.52 | 3.61 | 4.19 | 3.99 |
| **ISALux (ours)** | **3.58** | 3.91 | **3.21** | **3.40** | **3.34** |

</details>

Numbers are taken from Tables 1 – 3 of the [WACV paper](https://openaccess.thecvf.com/content/WACV2026/html/Balmez_ISALux_Illumination_and_Semantics-Aware_Transformer_Employing_Mixture_of_Experts_for_WACV_2026_paper.html).

## ⚙️ Installation

**Prerequisite:** [Anaconda or Miniconda](https://docs.anaconda.com/miniconda/).

**1. Clone the repository**

```bash
git clone https://github.com/raulbalmez/ISALux.git
cd ISALux
```

**2. Create the environment** (Python 3.13 was used for development and is recommended)

```bash
conda create -n isalux python=3.13 -y
conda activate isalux
```

**3. Install the requirements**

```bash
pip install -r requirements.txt
```

The pretrained MobileNetV3 segmenter and VGG-19 weights are downloaded automatically by torchvision on first use.

## 📂 Data

Download LOL-v1, LOL-v2-real and LOL-v2-synthetic with:

```bash
python scripts/download_data.py --out data
```

The datasets are placed under the `data/` folder.

## 📦 Checkpoints

Pretrained ISA-T weights are available on **[Google Drive](https://drive.google.com/drive/folders/REPLACE_WITH_FOLDER_ID)**.

| Dataset | File |
| :--- | :--- |
| LOL-v1 | `lolv1_best.pth` |
| LOL-v2-real | `lolv2_real_best.pth` |
| LOL-v2-synthetic | `lolv2_syn_best.pth` |

Place them under `checkpoints/` and pass the file to `evaluate.py` with `--weights` (see [Inference](#-inference)). The frozen MobileNetV3 and VGG-19 weights are downloaded automatically by torchvision.

## 🏋️ Training

```bash
python train.py --data_root <path/to/dataset> --dataset <lolv1|lolv2_real|lolv2_syn>
```

Training follows the paper: 300k iterations, Adam (β₁=0.9, β₂=0.999), batch size 8, random 256×256 crops with flips and rotations, and a learning rate of 2e-4 rising to 3e-4 at 92k iterations, decaying to 2e-4 at 208k and to 1e-6 at the end. Add `--amp` for bfloat16 mixed precision.

## 🔍 Inference

```bash
python evaluate.py \
    --data_root <path/to/dataset> \
    --dataset <lolv1|lolv2_real|lolv2_syn> \
    --weights <path/to/weights.pth>
```

PSNR and SSIM are reported with and without the ground-truth-mean correction of Retinexformer.

## 🔧 Configuration

The loss is a weighted sum of L2, VGG-19 perceptual and MS-SSIM terms. Each term can be switched off to reproduce the ablation in the paper:

```bash
python train.py ... --w_l2 1.0 --w_perc 0.05 --w_ssim 0.5
python train.py ... --w_perc 0          # L2 + MS-SSIM only
```

Model width and expert settings are arguments of `ISAT` in [`isalux/model.py`](isalux/model.py).

## 🗒️ Things to Follow

- [ ] Support for the SDSD-indoor and SDSD-outdoor datasets
- [ ] Support for the LOL-Blur dataset
- [ ] Checkpoints for SDSD-in, SDSD-out and LOL-Blur
- [x] Video demonstration of ISALux (on the [project page](https://raulbalmez.github.io/ISALux/))

## 📄 Citation

If you find ISALux useful in your research, please consider citing:

```bibtex
@inproceedings{balmez2026isalux,
  title     = {ISALux: Illumination and Semantics-Aware Transformer Employing Mixture of Experts for Low Light Image Enhancement},
  author    = {Balmez, Raul and Brateanu, Alexandru and Orhei, Ciprian and Ancuti, Codruta O. and Ancuti, Cosmin},
  booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  year      = {2026},
  eprint    = {2508.17885},
  archivePrefix = {arXiv}
}
```

## 📜 License

ISALux is source-available for non-commercial research and educational use under the [PolyForm Noncommercial License 1.0.0](LICENSE). It is not open-source software in the OSI sense, and commercial use requires a separate written license. This applies to the source code, model definitions, documentation and pretrained weights. Third-party components remain under their original licenses. See [LICENSE](LICENSE) for the full terms.

## Acknowledgements

Thanks to the authors of [Retinexformer](https://github.com/caiyuanhao1998/Retinexformer) for the evaluation protocol, and to the authors of [LOL](https://daooshee.github.io/BMVC2018website/) and [LOL-v2](https://github.com/flyywh/CVPR-2020-Semi-Low-Light) for making their datasets publicly available.

<div align="center">

If you find this work useful, please consider starring the repository.

</div>
