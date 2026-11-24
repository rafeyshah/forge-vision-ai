# Data

This repository does **not** contain any dataset files. Every dataset below is downloaded from its official source. Read each license before using the data. Never commit images, masks or archives to Git.

## Datasets

| Dataset | Role in this project | Size | License |
|---|---|---|---|
| [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) | Main dataset. The MVP uses the `bottle` and `metal_nut` categories. | 148 MB + 157 MB (full set 4.9 GB) | CC BY-NC-SA 4.0 |
| [VisA](https://github.com/amazon-science/spot-diff) | Second evaluation dataset, for cross-dataset generalization | ~1.8 GB (single archive) | CC BY 4.0 |
| [KolektorSDD2](https://vicos.si/resources/kolektorsdd2/) | Supervised segmentation, using its own training masks | see source page | CC BY-NC-SA 4.0 |
| [MVTec AD 2](https://www.mvtec.com/company/research/datasets/mvtec-ad-2) | Optional harder benchmark | ~8,000 images | CC BY-NC-SA 4.0 |

The CC BY-NC-SA datasets are for **non-commercial use only**. This project is a research and portfolio prototype. Do not redistribute the data or ship models trained on it in a commercial product without permission from the dataset owners.

## Download (Google Colab)

Model training and evaluation run in Colab. Download the data into the Colab VM's local disk (`/content`), not into Google Drive. The download takes seconds there, and reading thousands of small files from Drive is slow. Re-run the download cell at the start of every Colab session.

### MVTec AD: MVP categories

```bash
mkdir -p /content/data/mvtec_ad && cd /content/data/mvtec_ad
wget -q --show-progress -O bottle.tar.xz \
  "https://www.mydrive.ch/shares/163640/9a565b9a02f45cd135e46b1c36e01ec5/download/420937370-1629958698/bottle.tar.xz"
wget -q --show-progress -O metal_nut.tar.xz \
  "https://www.mydrive.ch/shares/163647/6d5a833cdee9035a45d5ce0921ecd24d/download/420937637-1629959294/metal_nut.tar.xz"
for f in *.tar.xz; do tar -xf "$f" && rm "$f"; done
```

To add more categories, copy their links from the [official downloads page](https://www.mvtec.com/company/research/datasets/mvtec-ad/downloads). The full dataset is also listed there.

### VisA (when you reach cross-dataset evaluation)

```bash
mkdir -p /content/data/visa && cd /content/data/visa
wget -q --show-progress https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar
tar -xf VisA_20220922.tar && rm VisA_20220922.tar
```

Convert VisA to the standard train/test layout with the official `1cls` split. Use the `prepare_data.py` script and `split_csv/1cls.csv` from the [spot-diff repository](https://github.com/amazon-science/spot-diff):

```bash
python ./utils/prepare_data.py --split-type 1cls --data-folder ./VisA \
  --save-folder ./VisA_pytorch --split-file ./split_csv/1cls.csv
```

### KolektorSDD2 (supervised segmentation milestone)

Download it from the button on the [official page](https://vicos.si/resources/kolektorsdd2/), then extract it to `/content/data/ksdd2/`.

## Expected layout

```text
/content/data/
  mvtec_ad/
    bottle/
      train/good/            # defect-free training images
      test/good/             # defect-free test images
      test/<defect_type>/    # defective test images
      ground_truth/<defect_type>/   # pixel masks for defective test images
    metal_nut/ ...
  visa/VisA_pytorch/1cls/<category>/ ...
  ksdd2/ ...
```

For local development, put the data under `data/raw/`, which Git ignores.

## Splits and evaluation rules

- **Use the official splits.** For MVTec AD, `train/` contains only defect-free images. `test/` mixes defect-free and defective images.
- **Validation set:** MVTec AD has no official validation split, and `train/` contains no defects. Use this protocol and commit the file lists to `configs/splits/`:
  - `train`: 90% of `train/good`
  - `val_normal`: 10% of `train/good`
  - `val`: a stratified 30% of the official `test/` (fixed seed), used for thresholds and hyperparameters
  - `test`: the remaining 70% of `test/`, used **only** for the final reported metrics
- **Paper comparison:** also report image and pixel AUROC on the full official `test/`, clearly labeled. AUROC uses no threshold, so this doesn't leak. See [docs/IMPLEMENTATION_GUIDE.md](../docs/IMPLEMENTATION_GUIDE.md) §1.2.
- **Never train on test masks.** Supervised segmentation uses KolektorSDD2's own training split. MVTec `ground_truth/` masks are used for evaluation only.
- **MVTec AD 2:** report metrics on `test_public`. The `test_private` splits have no public ground truth.
- Record the dataset version, archive checksum (`sha256sum`), seed and split file for every experiment.

## Citations

- Bergmann et al., *MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection*, CVPR 2019.
- Zou et al., *SPot-the-Difference Self-Supervised Pre-training for Anomaly Detection and Segmentation*, ECCV 2022 (VisA).
- Božič et al., *Mixed supervision for surface-defect detection: from weakly to fully supervised learning*, Computers in Industry 2021 (KolektorSDD2).
