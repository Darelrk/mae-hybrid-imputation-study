# Heart Disease Imputation using Masked Autoencoder

## 🏆 Winner: MAE (Deep Learning)

**Performance:**
- **RMSE**: 1.0380 ± 0.0402
- **Best RMSE**: 0.9461 (fold_2_repeat_2_mask_0.2)
- **Dataset**: UCI Heart Disease (920 rows, target: fbs)
- **Success Rate**: 100% (45/45 runs completed)

## 📁 Project Structure

```
github_upload/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── imputation_research_plan.md  # Complete research plan
├── combined_research_notebook.py # 📓 Combined Research Notebook (All Pipelines)
├── src/
│   ├── hybrid_pipeline.py       # 🔄 Hybrid Pipeline Source
│   ├── mae_residuals.png        # 📊 MAE residual analysis
│   ├── mae_true_vs_pred.png     # 📊 MAE predictions vs actual
│   ├── final_comparison_complete.png # 📊 Complete method comparison
│   └── final_comparison_rmse.png # 📊 RMSE comparison
├── data/
│   ├── data_cleaned.csv         # 📊 Complete dataset (920 rows)
│   └── masks/                   # 🎭 Mask files
├── artifacts/
│   ├── scaler.pkl               # StandardScaler parameters
│   ├── encoders.json            # LabelEncoder mappings
│   ├── masks_manifest.json      # Mask metadata
│   └── results/                 # 📊 Analysis results
└── notebooks/                   # 📓 Jupyter notebooks
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Choose Your Pipeline

#### 🏆 Winner: MAE (Deep Learning)
```bash
cd src
python mae_pipeline.py
```

#### 🔄 Alternative: Hybrid Methods
```bash
cd src
python hybrid_pipeline.py        # Hybrid V1 (MAE inputs)
python pure_classical_pipeline.py  # Pure Classical
python classical_pipeline.py      # Classical baseline
```

### 3. View Results & Visualizations

**Performance Results:**
- Check `artifacts/results/final_aggregate_summary.json` for detailed metrics
- See `artifacts/results/final_comparison_rmse.png` for performance chart

**Data Analysis Charts:**
- `data/charts/data_correlation.png` - Feature correlations
- `data/charts/data_missingness.png` - Missingness patterns  
- `data/charts/data_target_dist.png` - Target distribution

**Model Performance Charts:**
- `src/mae_residuals.png` - MAE residual analysis
- `src/mae_true_vs_pred.png` - MAE predictions vs actual
- `src/final_comparison_complete.png` - Complete method comparison

## 📊 Dataset Information

**Source:** UCI Heart Disease Repository
```
Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1989). 
Heart Disease [Dataset]. UCI Machine Learning Repository. 
https://doi.org/10.24432/C52P4X.
```

**Note:** The original dataset structure has been modified according to research requirements:
- Schema harmonization across 4 source files (Cleveland, Hungarian, Switzerland, VA)
- Added missing indicators for features with missing values
- Applied StandardScaler to numeric features
- Encoded categorical variables with LabelEncoder
- Generated synthetic masks for imputation experiments

**Target Variable:** `fbs` (fasting blood sugar)
- **Missingness Rate:** 9.8% (90/920 rows)
- **Clinical Relevance:** Important cardiovascular risk factor
- **Type:** Binary (0/1 after preprocessing)

**Features:** 25 columns including:
- Clinical measurements (age, sex, cp, trestbps, chol, thalach, etc.)
- Missing indicators for all features with missing values
- Source file encoding for batch effects

## 🎯 Method Comparison

| Rank | Method | Category | RMSE | Best RMSE |
|------|--------|----------|------|-----------|
| 🥇 | **MAE (Baseline)** | Deep Learning | **1.0380** | **0.9461** |
| 🥈 | Residual-RF | Hybrid V2 | 1.1467 | 1.1239 |
| 🥉 | Pure-KMeans | Pure Classical | 1.1472 | 1.1472 |
| 4 | Hybrid-KMeans | Hybrid V1 | 1.1492 | 1.1492 |

## 🔬 Technical Details

### MAE Architecture (Optimal Configuration)
```python
# Encoder: 64 → 32 → 16 (latent)
# Decoder: 16 → 32 → 64
# Mask Rate: 0.4 (fixed)
# Learning Rate: 1e-3
# Training: 100 epochs with early stopping
```

### Key Findings
- **Simple architecture works best** for this medical dataset
- **Dynamic masking** hurt performance
- **Baseline configuration** was already well-optimized
- **Deep learning outperforms** classical methods for this dataset

## 📈 Results Summary

- **Total Runs:** 45 (5-fold × 3 repeats × 3 mask rates)
- **Runtime:** ~4.4 minutes total
- **Stability:** Standard deviation 0.0402 (very consistent)
- **Best Configuration:** fold_2_repeat_2_mask_0.2

## 🔧 Reproducibility

This repository contains all essential components to reproduce the winning results:
- ✅ Complete MAE implementation
- ✅ Complete dataset and masks
- ✅ Trained scaler and encoders
- ✅ Performance metrics and visualizations
- ✅ Detailed documentation

## 📚 Research Context

This project implements and compares multiple imputation strategies for missing medical data using the UCI Heart Disease dataset. The original dataset has been extensively preprocessed and structured for imputation research:

**Dataset Modifications:**
- **Schema Harmonization:** Unified 4 different source files into consistent format
- **Feature Engineering:** Added missing indicators and encoded categorical variables  
- **Scaling:** Applied StandardScaler to normalize numeric features
- **Target Selection:** Chose `fbs` (fasting blood sugar) with optimal 9.8% missingness
- **Mask Generation:** Created 45 synthetic mask configurations for evaluation

**Imputation Strategies Compared:**
1. **Deep Learning:** Masked Autoencoder (MAE) - 🏆 WINNER
2. **Hybrid Methods:** Classical algorithms enhanced with MAE features
3. **Pure Classical:** Traditional imputation methods without MAE inputs

**Key Finding:** Simple, well-tuned MAE architecture outperforms complex hybrid methods for medical data imputation.

## 🤝 Citation

If you use this code or results, please cite both the original dataset and this research:

**Original Dataset:**
```
Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1989). 
Heart Disease [Dataset]. UCI Machine Learning Repository. 
https://doi.org/10.24432/C52P4X.
```

**This Research:**
```
Heart Disease Imputation using Masked Autoencoder
MAE Winner: RMSE 1.0380 ± 0.0402
Dataset: UCI Heart Disease Repository
Target: fbs (fasting blood sugar) - 9.8% missingness
```

## 📄 License

This project is for research and educational purposes.

## 🔍 Future Work

- Multi-target imputation
- Uncertainty quantification
- Clinical validation studies
- Real-time deployment

---

**🎯 Key Takeaway:** Simple, well-tuned MAE architecture outperforms complex hybrid methods for medical data imputation.
