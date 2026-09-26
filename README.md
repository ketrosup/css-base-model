# CNN-Based Visual Defect Detection — Casting Products

Automated visual inspection of submersible-pump impeller castings: classify each top-down image as **OK** or **Defective**.

**Business goal:** minimise **false negatives** (defective parts shipped to customers) while keeping **false positives** (good parts sent for manual re-check) low. Recall on the *Defective* class is therefore the headline metric.

---

## Results (held-out test set, 715 images)

Every model is scored on the same test split. The "tuned" threshold is picked on the **validation set only**: it's the threshold with the highest precision that still reaches **≥ 99.5 % recall**.

| Model | Params | Threshold | Accuracy | Precision | Recall | F1 | Missed defects (FN) | False alarms (FP) |
|---|---|---|---|---|---|---|---|---|
| Baseline CNN | 1.29 M | 0.50 | 0.9944 | 1.0000 | 0.9912 | 0.9956 | 4 | 0 |
| Baseline CNN | | tuned 0.58 | 0.9930 | 1.0000 | 0.9890 | 0.9945 | 5 | 0 |
| **Cand A: Regularised CNN** ✅ | 2.35 M | 0.50 | 0.9986 | 1.0000 | 0.9978 | 0.9989 | 1 | 0 |
| **Cand A: Regularised CNN** ✅ | | tuned 0.20 | 0.9986 | 1.0000 | 0.9978 | 0.9989 | **1** | **0** |
| Cand B: ResNet18 (transfer learning) | 11.18 M | 0.50 | 0.9972 | 1.0000 | 0.9956 | 0.9978 | 2 | 0 |
| Cand B: ResNet18 (transfer learning) | | tuned 0.31 | 0.9986 | 1.0000 | 0.9978 | 0.9989 | 1 | 0 |

**Selected model: Candidate A (regularised CNN from scratch).** It ranked highest on validation ROC-AUC. On the test set it misses 1 of 453 defects and raises no false alarms. It ties ResNet18 at the tuned threshold with ~5× fewer parameters.

**Inference latency (single image):**

| Model | CPU ms/image | GPU ms/image (RTX 4050 Laptop) |
|---|---|---|
| Baseline CNN | 3.5 | 2.3 |
| Cand A: Regularised CNN | 21.6 | 4.4 |
| Cand B: ResNet18 | 19.4 | 4.6 |

All three models are fast enough for real-time line-side inspection. The full numbers are in [results.json](results.json) and the plots are in [figures/](figures/).

![Test-set confusion matrices](figures/cm_all.png)

---

## What was done

The main notebook is **[defect_detection_pytorch.ipynb](defect_detection_pytorch.ipynb)**.

1. **Data.** The dataset uses the `casting_data/` version: 300×300 grayscale images with an official train/test split (6,633 train / 715 test, ~57 % defective). 15 % of `train/` is split off, stratified, as a validation set (995 images). That validation set drives early stopping, model selection and threshold tuning. The test set is only touched for the final evaluation.
2. **Phase 1: Baseline CNN.** 4 × [Conv → ReLU → MaxPool] at 128×128 with no augmentation or batch-norm. This is the reference point. Diagnosis: it overfits, and at 128 px the small pinholes become only a few pixels wide.
3. **Phase 2: Two improvement candidates**, trained with the same loss, splits, early-stopping rule and metrics:
   - **A. Regularised scratch CNN:** 224 px input, double-conv blocks with BatchNorm, a global-average-pooling head, weight decay and GPU augmentation. The augmentation uses rotations and flips (the impeller is rotationally symmetric), small affine zoom/shift, and brightness/contrast jitter.
   - **B. ResNet18 transfer learning:** ImageNet weights, full fine-tune with discriminative learning rates (backbone 3e-4, head 3e-3), one-cycle schedule and the same augmentation.
4. **Threshold tuning.** The recall-first operating point (≥ 99.5 % recall) is chosen on validation.
5. **Deployment checks.** The notebook measures CPU/GPU latency, shows the remaining errors, and runs Grad-CAM to check the model is looking at the defect rather than background artefacts.
6. **Outputs.** The notebook saves `best_model_cnn.pt` (Candidate A weights), `results.json` and `figures/*.png`.

The TensorFlow notebook **[defect_detection_cnn.ipynb](defect_detection_cnn.ipynb)** is an earlier version. It uses a baseline CNN plus MobileNetV2 transfer learning on 1,000 images from `casting_512x512/`. It's kept for reference, and the results above come from the PyTorch notebook.

---

## Repository layout

```text
├── defect_detection_pytorch.ipynb   # main notebook (PyTorch) — produces all results
├── defect_detection_cnn.ipynb       # earlier TensorFlow/Keras version
├── best_model_cnn.pt                # trained weights of the selected model (Cand A, state_dict)
├── results.json                     # metrics, thresholds, latency, epochs
├── figures/                         # saved plots (samples, curves, confusion matrices, ROC, Grad-CAM)
├── archive/                         # dataset (Kaggle casting product images)
│   ├── casting_data/casting_data/{train,test}/{ok_front,def_front}/
│   └── casting_512x512/casting_512x512/{ok_front,def_front}/
├── MODEL_COMPARISON_REPORT.md       # written report / reflection
├── reflective-question.pdf          # assignment brief
└── requirements.txt
```

---

## How to run

**Prerequisites:** Python 3.12, and ideally an NVIDIA GPU. The notebook also runs on CPU, but training is much slower.

```powershell
# 1. Clone
git clone https://github.com/ketrosup/css-base-model.git
cd css-base-model

# 2. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate

# 3. (GPU only) install CUDA-enabled PyTorch first
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128

# 4. Install the rest
pip install -r requirements.txt
```

Then:

- Open `defect_detection_pytorch.ipynb` in VS Code, select the `.venv` kernel, and click **Run All**. To use the browser instead, run `pip install notebook` and then `jupyter notebook`.
- The notebook regenerates `figures/`, `results.json` and `best_model_cnn.pt`. Seeds are fixed (`SEED = 42`), but GPU nondeterminism can shift results slightly.

**Dataset:** it's already included under `archive/`. If it's missing, download it from [Kaggle — Casting product image data](https://www.kaggle.com/datasets/ravirajsingh45/real-life-industrial-dataset-of-casting-product) and unzip it into `archive/`. The notebook also finds the dataset automatically at the standard Kaggle (`/kaggle/input/...`) and Colab (`/content/...`) paths.

### Using the saved model

```python
import torch
# Define/import the ImprovedCNN class from the notebook first (section 3.3)
model = ImprovedCNN()
model.load_state_dict(torch.load("best_model_cnn.pt", map_location="cpu"))
model.eval()

# x: (1, 1, 224, 224) grayscale tensor in [0,1], normalised with the train mean/std (PIX_MEAN, PIX_STD from the notebook)
p_defect = torch.sigmoid(model((x - PIX_MEAN) / PIX_STD)).item()
is_defective = p_defect >= 0.2035   # validation-tuned, recall-first threshold (results.json)
```

---

## Caveats

- The test set is small (453 defective images), so a 1-error difference is ~0.2 pp of recall. Candidates A and B are effectively tied.
- The `casting_data` images were augmented by the dataset author, so near-duplicates may exist across the train and test splits. Real production data will likely be harder, so re-validate on fresh line images before deployment.
- Monitor recall and false-positive rate in production, and retrain when defect types or lighting drift.

---

## References

- Dataset: <https://www.kaggle.com/datasets/ravirajsingh45/real-life-industrial-dataset-of-casting-product>
- ResNet: <https://arxiv.org/abs/1512.03385>
- Grad-CAM: <https://arxiv.org/abs/1610.02391>
- PyTorch: <https://pytorch.org/>
