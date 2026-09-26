# Automated Visual Defect Detection with CNNs

*Reflective Question (industry relevant): Manufacturing Quality Control*

---

## ⚠️ Assignment brief — key points

From [reflective-question.pdf](reflective-question.pdf):

**Scenario:** manual inspection of cast metal parts is slow, inconsistent and expensive. The task is to build a CNN-based visual inspection system that can run on the production line and flag defective units **in real time**.

**Business goal:** **minimise false negatives** (missed defects escaping to customers) while keeping **false positives low enough** that the line isn't flooded with manual re-checks.

**Dataset:** [Kaggle — Casting Product Image Data](https://www.kaggle.com/datasets/ravirajsingh45/real-life-industrial-dataset-of-casting-product) (submersible-pump impeller castings, labelled *defective* / *ok*).

### Required tasks and where they're covered

| Phase | Requirement | Status |
|---|---|---|
| **1 — Baseline** | Build a CNN **from scratch** that classifies defective vs non-defective images | ✅ `BaselineCNN` (notebook §2) |
| | Report accuracy, precision, recall, F1 **and the confusion matrix** | ✅ notebook §2, [results](#results-same-test-set-715-images) |
| | **Key lessons and reflection** on the learning journey | ✅ [Reflection](#key-lessons--reflection) |
| **2 — Improve** | Improve the baseline with justified technique(s) | ✅ Candidates A and B (notebook §3) |
| | **Justify** the chosen approach over the alternatives | ✅ [Options considered](#phase-2--options-considered) |
| | Compare against Phase 1 on the **same test set and metrics** | ✅ [Results](#results-same-test-set-715-images) |
| | **Key lessons and reflection** on the approach vs the baseline | ✅ [Reflection](#key-lessons--reflection) |

### Deliverables to submit

1. **Executed Jupyter notebook** containing data prep, model development, training, evaluation, graphs, meaningful comments, outputs and observations → [defect_detection_pytorch.ipynb](defect_detection_pytorch.ipynb)
2. **Model comparison report and reflection** discussing the models, their performance, and the modelling process, learning and outcomes → this README (the sections below)

### Academic integrity rules (read before submitting)

- The submission **must be original**.
- The reflection must be **unique and personalised**, based on *your own* experiments and understanding.
- Copied, shared or jointly prepared work gets **zero marks**.
- The reflection must show *your own* interpretation, not generic or reproduced content.
- **Acknowledge any external resources** you used.
- Digital tools are allowed only for **limited proofreading or basic support**. The ideas, structure and execution must be your own.
- Complete the work **before the assessment day**.

> **Note:** the reflection section below summarises what the experiments showed. Rewrite it in your own words, with your own experience of the process, before you submit.

---

## Results (same test set, 715 images)

Positive class = **Defective**. The "tuned" threshold is chosen on the **validation set only**: it's the highest-precision threshold that keeps recall ≥ **99.5 %**.

| Model | Params | Threshold | Accuracy | Precision | Recall | F1 | Missed defects (FN) | False alarms (FP) |
|---|---|---|---|---|---|---|---|---|
| Baseline CNN | 1.29 M | 0.50 | 0.9944 | 1.0000 | 0.9912 | 0.9956 | 4 | 0 |
| Baseline CNN | | tuned 0.58 | 0.9930 | 1.0000 | 0.9890 | 0.9945 | 5 | 0 |
| **Cand A: Regularised CNN** ✅ | 2.35 M | 0.50 | 0.9986 | 1.0000 | 0.9978 | 0.9989 | 1 | 0 |
| **Cand A: Regularised CNN** ✅ | | tuned 0.20 | 0.9986 | 1.0000 | 0.9978 | 0.9989 | **1** | **0** |
| Cand B: ResNet18 (transfer learning) | 11.18 M | 0.50 | 0.9972 | 1.0000 | 0.9956 | 0.9978 | 2 | 0 |
| Cand B: ResNet18 (transfer learning) | | tuned 0.31 | 0.9986 | 1.0000 | 0.9978 | 0.9989 | 1 | 0 |

**Selected model: Candidate A.**
- **Result:** it misses 1 of 453 defects and raises 0 false alarms, down from 4 missed defects for the baseline (−75 %).
- **Why A over B:** it ties ResNet18 at the tuned threshold and is ~5× smaller.

![Test-set confusion matrices](figures/cm_all.png)

**Latency (single image):** all three models are real-time capable.

| Model | CPU ms/image | GPU ms/image (RTX 4050 Laptop) |
|---|---|---|
| Baseline CNN | 3.5 | 2.3 |
| Cand A | 21.6 | 4.4 |
| Cand B | 19.4 | 4.6 |

Full numbers are in [results.json](results.json) and the plots are in [figures/](figures/).

---

## What was done

**Data**
- Uses the `casting_data/` version of the dataset: 300×300 images, loaded as grayscale, with the official train/test split (6,633 train / 715 test, ~57 % defective).
- 15 % of train is split off, stratified, as **validation** (995 images). It's used for early stopping, model selection and threshold tuning.
- The test set is touched only once, for the final numbers.

**Phase 1 — Baseline CNN (from scratch)**
- **Architecture:** 128×128 input, 4 × [Conv → ReLU → MaxPool] (32→64→128→128), then Dense(128) → Dropout(0.5) → output.
- **Training setup:** no batch-norm and no augmentation.
- **Result:** already strong, with 99.1 % recall.
- **Limitation:** at 128 px, the small pinholes and blow-holes shrink to a few pixels.

### Phase 2 — Options considered

| Option | Decision | Reason |
|---|---|---|
| **A. Regularised scratch CNN** (224 px, BatchNorm, GAP head, weight decay, augmentation) | **Tried** | Small and fast, needs no external weights, and isolates the effect of resolution plus regularisation |
| **B. ResNet18 transfer learning** (ImageNet, full fine-tune, discriminative learning rates) | **Tried** | Pre-trained edge/texture features transfer well, and it's still light enough for edge inference |
| Bigger backbones (ResNet50, EfficientNet, ViT) | Rejected | 2–8× slower, with little headroom left |
| Class re-weighting / oversampling | Rejected | Imbalance is mild. Threshold tuning handles the FN/FP trade-off more transparently |
| Anomaly detection (train on OK only) | Rejected | Plenty of labelled defects are available, so supervised learning is stronger |

**Augmentation (Candidates A and B)**
- 90° rotations and flips. These preserve the label because the impeller is rotationally symmetric.
- ±15° rotation, ±10 % zoom and ±5 % shift, to simulate positioning on the conveyor.
- ±15 % brightness/contrast jitter, to simulate lighting drift.

**Fair comparison**
- All models share the same splits, loss (BCE-with-logits), early-stopping rule and metrics code.
- The winner is chosen on **validation**, then its threshold is tuned on validation, and only then is it scored on test.

**Error analysis and explainability**
- Candidate A's only miss scored p(defective) = 0.107, and the defect is barely visible at 224 px.
- Grad-CAM shows attention on the rim and edge chips where the defects are ([figures/gradcam.png](figures/gradcam.png)).
- ⚠️ The Grad-CAM **code cell is missing** from the notebook; the figure comes from an earlier run. Restore that cell before submitting.

---

## Key lessons & reflection

1. **Build a strong baseline first:**
   - A 1.3 M-parameter CNN already reached 99.1 % recall.
   - Most of the later gain came from **higher resolution (128 → 224 px) plus augmentation**, not from a fancier architecture.
2. **Check the diagnosis against the logs:** the baseline's validation loss kept falling, so it was *not* over-fitting. It was limited by input resolution.
   - ⚠️ The notebook's §3.1 markdown still says "over-fitting". Correct it so it matches.
3. **Transfer learning converges fastest:**
   - ResNet18 hit > 99 % validation accuracy after one epoch.
   - A well-regularised scratch CNN matched it on this narrow, consistent domain at ~1/5 of the size.
4. **The threshold is a business decision:**
   - Tuning it on validation for a recall target makes the FN/FP trade-off explicit.
   - With few borderline examples it's noisy, though: the baseline's tuned threshold went from 4 to 5 misses on test.
5. **Report differences honestly:**
   - 1 error is only 0.22 pp of test recall, so A, B and the baseline are close.
   - The choice of A rests on size, simplicity and speed as much as on the metric.

**Limitations**
- **Small test set:** 453 defective images is too few to separate models reliably at > 99.5 % recall.
- **Possible near-duplicates:** the dataset author augmented the images, so train and test may share near-copies, which inflates scores.
- **Single run:** results come from one training run (seed 42).
- **Next step:** before real deployment, re-validate on fresh line images and monitor recall and false alarms in production.

---

## How to run

**Prerequisites:** Python 3.12, and ideally an NVIDIA GPU. The notebook also runs on CPU, but training is much slower.

```powershell
git clone https://github.com/ketrosup/css-base-model.git
cd css-base-model
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate

# GPU only: install CUDA-enabled PyTorch first
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128

pip install -r requirements.txt
```

Then:

- Open [defect_detection_pytorch.ipynb](defect_detection_pytorch.ipynb) in VS Code, select the `.venv` kernel, and click **Run All**. For the browser, run `pip install notebook` and then `jupyter notebook`.
- The run regenerates `figures/`, `results.json` and `best_model_cnn.pt`. Seeds are fixed (`SEED = 42`).
- **Dataset:** it's already included in `archive/`. If it's missing, download it from Kaggle and unzip it into `archive/`. The standard Kaggle and Colab paths are detected automatically.

### Using the saved model

```python
import torch
# Define the ImprovedCNN class from the notebook (section 3.3) first
model = ImprovedCNN()
model.load_state_dict(torch.load("best_model_cnn.pt", map_location="cpu"))
model.eval()

# x: (1, 1, 224, 224) grayscale tensor in [0,1]; PIX_MEAN/PIX_STD from the notebook
p_defect = torch.sigmoid(model((x - PIX_MEAN) / PIX_STD)).item()
is_defective = p_defect >= 0.2035   # validation-tuned, recall-first threshold
```

---

## Repository layout

```text
├── defect_detection_pytorch.ipynb   # main notebook (PyTorch): the deliverable, produces all results
├── defect_detection_cnn.ipynb       # earlier TensorFlow/Keras version (reference only)
├── best_model_cnn.pt                # trained weights of the selected model (Candidate A)
├── results.json                     # metrics, thresholds, latency, epochs
├── figures/                         # saved plots
├── archive/                         # dataset
├── reflective-question.pdf          # assignment brief
└── requirements.txt
```

---

## References

- Dataset: <https://www.kaggle.com/datasets/ravirajsingh45/real-life-industrial-dataset-of-casting-product>
- ResNet: He et al., 2015: <https://arxiv.org/abs/1512.03385>
- Grad-CAM: Selvaraju et al., 2017: <https://arxiv.org/abs/1610.02391>
- PyTorch: <https://pytorch.org/>
