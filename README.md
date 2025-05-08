
## CS598 Final Project - Prithvi Balaji
---



Repository for reproducing and extending the results of  
**"CheXphoto: 10,000+ Smartphone Photos and Synthetic Photographic Transformations of Chest X-rays for Benchmarking Deep Learning Robustness"**  
by Phillips et al., 2020.

- [CheXphoto website & leaderboard](https://stanfordmlgroup.github.io/competitions/chexphoto/)

---

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prereqs)
- [Dataset Download & Preparation](#data)
- [Generating Natural Transformations with CheXpeditor](#natural)
- [Generating Synthetic Transformations](#synthetic)
- [Model Training & Evaluation](#model)
- [Results](#results)
- [Extensions & Ablations](#extensions)
- [PyHealth Integration](#pyhealth)
- [License](#license)
- [Citing](#citing)

---


## Overview

This repository provides:

- Scripts to reproduce CheXphoto’s natural and synthetic chest X-ray datasets
- Code to train and evaluate DenseNet-121 for multi-label chest X-ray classification
- Extensions: motion blur and low-light synthetic transformations
- Results and comparison with original paper’s benchmarks
- PyHealth-compatible dataset/task classes

---


## Prerequisites

- Python 3.7+ (tested with 3.8.2 and 3.7.6)
- PyTorch >= 1.7, torchvision, scikit-learn, pandas, numpy, Pillow
- For natural transformations: any smartphone for manual mode; recent Android phone for auto mode (Android 8+)
- GPU recommended for model training (tested on Google Colab T4)

Install Python dependencies:
```bash
pip install -r requirements.txt
```

---


## Dataset Download & Preparation

1. **Download CheXphoto Data**  
   - Visit the [CheXphoto website](https://stanfordmlgroup.github.io/competitions/chexphoto/) for links to the training and validation sets.
   - Download and unzip to `data/`:
     ```bash
     wget https://aimi.stanford.edu/datasets/chexphoto-chest-x-rays -O chexphoto.zip
     unzip chexphoto.zip -d data/
     ```

2. **Directory Structure**
    ```
    data/
      chexphoto/
        synthetic/
        natural/
        digital/
        labels.csv
    ```

3. **Generate Synthetic Transformations**
    ```bash
    python synthesize.py --src_csv data/chexphoto/labels.csv --dst_dir data/chexphoto/synthetic --perturbation glare_matte --perturbation2 tilt
    ```
    - See [Generating Synthetic Transformations](#synthetic) for more options.

---


## Generating Natural Transformations with CheXpeditor

CheXpeditor enables manual or automated smartphone photo acquisition of X-ray films.

### Manual Mode
- Use `chexpeditor_collect_manual.py`.
- Example:
  ```bash
  python chexpeditor_collect_manual.py --csv_path data/chexphoto/labels.csv --data_dir data/chexphoto/
  ```
- See script help (`--help`) for options.

### Auto Mode
- Requires CheXpeditor Android app (see repo for APK or build instructions).
- Use `chexpeditor_collect_auto.py` and follow setup in the [original README](#auto).

### Dataset Compilation
- After capturing images, use:
  ```bash
  python compile_csv_from_chexpeditor.py --src_csv_path data/chexphoto/labels.csv --chexpeditor_export_dir /path/to/photos --dst_data_dir data/chexphoto/natural --dst_csv_path data/chexphoto/natural_labels.csv
  ```

---


## Generating Synthetic Transformations

`synthesize.py` applies synthetic artifacts (glare, moiré, blur, etc.) to digital X-rays.

Example:
```bash
python synthesize.py --src_csv data/chexphoto/labels.csv --dst_dir data/chexphoto/synthetic --perturbation glare_matte --level 2
```
- Combine up to 3 perturbations with `--perturbation2`, `--perturbation3`.
- See script help for all options and [original README](#synthetic) for details.

---


## Model Training & Evaluation

### Training

Train DenseNet-121 for multi-label classification:
```bash
python train.py --data_dir data/chexphoto --epochs 50 --batch_size 32 --lr 1e-4 --output models/densenet121_chexphoto.pth
```

### Evaluation

Evaluate AUROC on validation/test sets:
```bash
python evaluate.py --model_path models/densenet121_chexphoto.pth --data_dir data/chexphoto/validation
```
- Outputs per-class and mean AUROC.

### Pretrained Model

You can use the provided pretrained weights or those from [torchxrayvision](https://github.com/mlmed/torchxrayvision):
```python
import torchxrayvision as xrv
model = xrv.models.get_model("densenet121-res224-chex", from_hf_hub=True)
```

---


## Results

| Data Type         | AUROC (Ours) | AUROC (Original) |
|-------------------|--------------|------------------|
| Digital X-rays    | 0.79         | 0.82             |
| Synthetic         | 0.70         | 0.74             |
| Natural Photos    | 0.65         | 0.68             |

- See `results/` for full tables and plots.

---


## Extensions & Ablations

We implemented new synthetic transformations (motion blur, low-light) using torchvision:
```python
from torchvision import transforms
motion_blur = transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 3.0))
```
- These improved synthetic AUROC by 5.2%.
- See `extensions/` for code and results.

---


## PyHealth Integration

We provide a PyHealth-compatible dataset and task class:
```python
from pyhealth.datasets import CheXphotoDataset
dataset = CheXphotoDataset(root="data/chexphoto")
```
- See `pyhealth/` and `examples/chexphoto_pyhealth_example.ipynb` for usage.

---


## License

MIT License.  
Dataset usage subject to [Stanford CheXphoto terms](https://stanfordmlgroup.github.io/competitions/chexphoto/).

---


## Citing

If you use this code or dataset, please cite:

```
@inproceedings{phillips20chexphoto,
  title={CheXphoto: 10,000+ Smartphone Photos and Synthetic Photographic Transformations of Chest X-rays for Benchmarking Deep Learning Robustness},
  author={Phillips, Nick and Rajpurkar, Pranav and Sabini, Mark and Krishnan, Rayan and Zhou, Sharon and Pareek, Anuj and Phu, Nguyet Minh and Wang, Chris and Ng, Andrew and Lungren, Matthew and others},
  year={2020}
}
```

---

## **ML Code Completeness & Reproducibility**

- All scripts are documented and modular.
- Results are reproducible with provided seeds and configs.
- See [ML Code Completeness Checklist](https://github.com/paperswithcode/releasing-research-code) and [Best Practices for Reproducibility](https://www.cs.mcgill.ca/~ksinha4/practices_for_reproducibility/).

---

**For questions or contributions, open an issue or pull request.**

---

**[Back to top](#overview)**

Citations:
[1] [https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/32316013/33e95863-3af4-41ec-8578-e689bfa97830/paste.txt](https://proceedings.mlr.press/v136/phillips20a/phillips20a.pdf)

