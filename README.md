# CNN-Based Visual Defect Detection
## Manufacturing Quality Control - Casting Products

Complete implementation of automated visual inspection system using deep learning.

---

## Files Overview

### **Notebooks** (Choose One)

- **`defect_detection_pytorch.ipynb`** ✅ **RECOMMENDED**
  - Uses PyTorch (compatible with Python 3.14+)
  - ResNet50 transfer learning architecture
  - All dependencies already installed
  - Run this notebook if you have Python 3.14

- **`defect_detection_cnn.ipynb`** (TensorFlow version)
  - Requires TensorFlow (needs Python 3.9-3.12)
  - MobileNetV2 transfer learning architecture
  - Alternative if you install Python 3.11 or 3.12

### **Documentation**

- **`MODEL_COMPARISON_REPORT.md`**
  - Detailed analysis of both models
  - Business context and deployment strategies
  - Performance metrics and comparisons
  - Key lessons and insights

---

## Quick Start

### Prerequisites
✅ Python 3.14 (already installed)  
✅ PyTorch (already installed)  

### Run the PyTorch Notebook

1. **Open the notebook** in your IDE/Jupyter:
   ```bash
   jupyter notebook defect_detection_pytorch.ipynb
   ```

2. **Run all cells** - The notebook includes:
   - Data loading and exploration
   - Phase 1: Baseline CNN implementation
   - Phase 2: Improved model with transfer learning
   - Detailed comparisons and visualizations

3. **Review results** - Each section outputs:
   - Accuracy, Precision, Recall, F1-Score
   - Confusion matrices
   - Performance visualizations
   - Key insights

---

## Project Structure

### Phase 1: Baseline CNN Classifier
- **Architecture**: 3-layer CNN with progressive filters (32→64→128)
- **Purpose**: Establish baseline performance and identify limitations
- **Output**: Baseline metrics for comparison

### Phase 2: Improved Model with Transfer Learning
- **Architecture**: ResNet50 pre-trained on ImageNet + custom top layers
- **Justification**: 
  - Pre-trained features capture general visual patterns
  - Better generalization with less training data
  - Faster convergence and improved recall (critical for defect detection)
- **Training Strategy**:
  - Phase 2a: Train custom layers (frozen base)
  - Phase 2b: Fine-tune base model layers
- **Output**: Improved metrics with significant gains in recall

---

## Key Metrics Explained

| Metric | Importance | Formula |
|--------|-----------|---------|
| **Accuracy** | Overall correctness | (TP + TN) / Total |
| **Precision** | False alarm rate | TP / (TP + FP) |
| **Recall** | **CRITICAL** - Defect catch rate | TP / (TP + FN) |
| **F1-Score** | Balance of precision & recall | 2 × (P × R) / (P + R) |

### Why Recall Matters Most
- **False Negative (FN)**: Defect escapes to customer → warranty claim + reputation damage
- **False Positive (FP)**: Good product flagged → manual review (minor cost)
- **Therefore**: Maximize recall, accept moderate false positives

---

## Dataset Info

**Source**: Kaggle - Casting Product Image Data  
**Link**: https://www.kaggle.com/datasets/ravirajsingh45/real-life-industrial-dataset-of-casting-product  
**Size**: 512×512 RGB images  
**Classes**: 
- Defective (1): Casting impellers with defects
- OK (0): Good quality castings

**Expected Usage**:
- ~1000 images (500 per class)
- Train: 70%, Validation: 15%, Test: 15%

---

## Results Summary

| Model | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|-----|
| Baseline CNN | [Metric] | [Metric] | [Metric] | [Metric] |
| Improved (ResNet50) | [Metric] | [Metric] | [Metric] | [Metric] |

*Metrics will be populated when you run the notebook*

---

## Deployment Recommendations

### Production Integration

```python
# Example inference code
model = torch.load('best_model.pt')
model.eval()

image = load_and_preprocess_image('product.jpg')
probability = model(image)[0].item()

if probability > 0.7:      # High confidence defect
    reject_product()
elif probability > 0.3:    # Uncertain - manual review
    flag_for_inspection()
else:                       # High confidence OK
    pass_product()
```

### Confidence Thresholds

- **>0.7**: Automatically reject (reduce false negatives)
- **0.3-0.7**: Manual inspection (safety margin)
- **<0.3**: Pass to customer (reduce false positives)

### Monitoring Strategy

Track over time:
- Daily true positive rate (defects caught)
- Daily false positive rate (unnecessary rejections)
- Model drift indicators
- Retraining triggers

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'torch'"
✅ PyTorch is already installed. Restart your Jupyter kernel.

### "ModuleNotFoundError: No module named 'tensorflow'"
→ This is expected. Use the PyTorch notebook instead.
→ Or install Python 3.11 and use TensorFlow notebook.

### Notebook runs slowly
→ This is normal on CPU. Consider using GPU if available.
→ Reduce dataset size in data loading cells for faster testing.

### CUDA not available (GPU)
→ Model will automatically use CPU. This is fine for inference.
→ PyTorch handles CPU/GPU transparently.

---

## Key Learning Points

### Technical Insights
1. **CNNs** effectively detect visual defects through hierarchical feature learning
2. **Transfer Learning** accelerates training and improves generalization
3. **Data Augmentation** (rotation, shifts, flips) prevents overfitting
4. **Regularization** (dropout, batch norm) essential for deep networks
5. **Fine-tuning** with lower learning rates adapts pre-trained models

### Business Context
1. **Automation Value**: 
   - Replaces slow, inconsistent manual inspection
   - Provides 24/7 consistent quality checks
   - Scales to multiple production lines
   
2. **Quality Impact**:
   - Catches defects before reaching customers
   - Reduces warranty claims and reputation damage
   - Enables data-driven quality decisions

3. **Continuous Improvement**:
   - Log predictions and corrections
   - Quarterly retraining with new data
   - Monitor for model drift over time

---

## Next Steps

1. ✅ **Review** reflective-question.pdf for full requirements
2. ✅ **Run** defect_detection_pytorch.ipynb to execute models
3. ✅ **Analyze** MODEL_COMPARISON_REPORT.md for insights
4. 📊 **Visualize** confusion matrices and metric comparisons
5. 🎯 **Document** your findings and reflections

---

## References

- PyTorch: https://pytorch.org/
- ResNet50: https://arxiv.org/abs/1512.03385
- Transfer Learning: https://cs231n.github.io/transfer-learning/
- Dataset: https://www.kaggle.com/datasets/ravirajsingh45/real-life-industrial-dataset-of-casting-product

---

## Support

For issues or questions:
- Check the notebook comments and cell outputs
- Review MODEL_COMPARISON_REPORT.md for detailed explanations
- Refer to PyTorch documentation: https://pytorch.org/docs

---

**Status**: ✅ Ready to Run  
**Last Updated**: 2026-09-21  
**Python Version**: 3.14+  
**Framework**: PyTorch 2.14+
