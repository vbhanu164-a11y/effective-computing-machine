# Effective-computing-machine
End-to-end loan default prediction pipeline: Exploratory Data Analysis (EDA) on mixed-type features and a PyTorch-based binary classification neural network with preprocessing, training, evaluation, and inference.

# 🧠 Loan Default Prediction – EDA & Binary Classification with PyTorch

## 📌 Project Overview
This project demonstrates a **complete machine learning workflow** to tackle a binary classification problem — predicting **loan default risk** based on a mixed-type dataset containing both numerical and categorical features.  

The workflow includes:
- **Exploratory Data Analysis (EDA)** to understand the dataset and extract insights.
- **Data preprocessing** for mixed features (numeric + categorical).
- **Model development**: A simple but effective **binary classification neural network** using PyTorch.
- **Model training & evaluation** with performance metrics.
- **Inference** on unseen data.

---

## 1️⃣ Exploratory Data Analysis (EDA)
EDA is crucial for understanding the data and identifying preprocessing needs.  
The following steps were performed:

**Goals**

Understand schema, class balance, and data quality.

Inspect distributions, outliers, and correlation structure.

Identify feature engineering needs before modeling.

**What the EDA does**

Initial inspection: shape, dtypes, sample head

Target distribution: confirms class balance/imbalance

Missingness: saves artifacts/missingness.csv (sorted)

Duplicates: count of duplicate rows (ex-ID)

Numeric distributions: top histograms

Categorical frequencies: bar charts of frequent labels

Correlation heatmap (numeric)

Outliers: IQR-based table artifacts/outliers_iqr.csv

Signal sniff test: optional Mutual Information (numeric→target)
artifacts/mutual_info_numeric.csv

**Key takeaways**

The dataset has N rows × M columns. Target positive rate ≈ P%.

Top missing: feature_a, feature_b, feature_c.
→ Use median (numeric) and mode (categorical) imputation.

Skewed numerics (e.g., loan_amount) → consider robust scaling or log transform.

High-cardinality categoricals (e.g., zip, employer) → rare-category bucketing and/or hashing; MLP with embeddings is a future improvement.

Correlated numerics (e.g., debt_ratio ~ utilization) → watch for redundancy.


### 📂 Data Inspection
- **Shape & Structure**: Checked number of rows, columns, and data types.
- **Target Distribution**: Verified class balance (loan default vs. non-default).
- **Sample View**: Displayed head/tail of dataset for initial impression.

### 🧹 Data Quality Checks
- **Missing Value Analysis**: Calculated per-column missingness, visualized using a bar chart.
- **Duplicate Detection**: Identified and counted duplicate rows (excluding ID).
- **Outlier Detection**: Used Interquartile Range (IQR) method to detect anomalies.

### 🔍 Feature Analysis
- **Numerical Features**:
  - Distribution plots for top numeric columns.
  - Summary statistics (mean, median, skewness).
  - Correlation heatmap to detect multicollinearity.
- **Categorical Features**:
  - Frequency counts for top categories.
  - Cardinality check for high-unique features.

### 📊 Example EDA Outputs
- Missingness heatmap.
- Target variable distribution.
- Correlation matrix heatmap.
- Example histograms and bar charts.

---

## 2️⃣ Data Preprocessing
Before model training, the dataset underwent preprocessing:
- **Handling Missing Values**:
  - Numerical: Median imputation.
  - Categorical: Mode imputation.
- **Encoding**:
  - Categorical features → One-Hot Encoding.
- **Scaling**:
  - Numerical features → StandardScaler.
- **Train/Validation Split**:
  - Stratified 80/20 split to maintain class balance.

---

## 3️⃣ Model Architecture – PyTorch Neural Network
A simple **fully-connected feedforward neural network** was implemented for binary classification.

**Training details**

Optimizer: Adam (lr=1e-3, weight_decay=1e-4)

Loss: BCEWithLogitsLoss with pos_weight for class imbalance

Split: Stratified 80/20 train/val

Metrics: ROC-AUC, Accuracy, Precision, Recall, F1

Batching: sparse→dense per batch (memory-safe)

Device: CPU or CUDA (auto)

**Why this baseline?**

Fast, readable, and strong for tabular data with good preprocessing.

Provides a clear scaffold for future upgrades (embeddings, deeper nets, CatBoost/etc. comparison).


## 4️⃣ Model Training & Evaluation
**Training loop** with real-time validation loss tracking.

**Early stopping** to avoid overfitting.

Final metrics reported on validation set:

Accuracy: XX%

Precision: XX%

Recall: XX%

F1 Score: XX%

ROC-AUC: XX%

## 5️⃣ Inference on Test Data
Loaded the trained model.

Applied the same preprocessing pipeline to unseen data.

Generated probability scores and binary predictions.

## 6️⃣ Key Insights & Recommendations
**Class imbalance:** Slight skew in target distribution → consider SMOTE or class weighting for improvement.

**High-cardinality categorical features:** Need embedding-based approaches in future.

**Model performance:** Good baseline; can be enhanced with hyperparameter tuning or more complex architectures.

## 📦 Tech Stack
**Python:** Data manipulation, modeling

**Pandas / NumPy:** Data analysis

**Matplotlib / Seaborn:** Visualization

**Scikit-learn:** Preprocessing & metrics

**PyTorch:** Neural network implementation


### 🏗 Architecture

Input Layer (num_features)  
↓  
Linear Layer → ReLU → Dropout(0.3)  
↓  
Linear Layer → ReLU → Dropout(0.3)  
↓  
Output Layer (1 neuron, Sigmoid activation)  

⚙️ Training Configuration
Loss Function: BCELoss (Binary Cross-Entropy)

Optimizer: Adam (lr=0.001)

Batch Size: 64

Epochs: 20

Metrics: Accuracy, Precision, Recall, F1-score, ROC-AUC 


---

## 🚀 How to Run
### 1. Install dependencies
pip install torch pandas scikit-learn matplotlib seaborn joblib

### 2. Train the model
python train_binary.py --train_csv "path/to/training_data.csv" --target_col "TARGET"

### 3. Run inference
python infer_binary.py --test_csv "path/to/testing_data.csv" --model_path "model.pth"

### Environment

python -m venv .venv

### Windows
.venv\Scripts\activate

### macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
1) Put data locally (not committed)

data/
  training_loan_data.csv
  testing_loan_data.csv
If your CSVs start with a non-header note line, pass --skiprows 1.

### Train (saves model + preprocessor + metrics)

python train_binary.py --train_csv data/training_loan_data.csv --target bad_flag --skiprows 1 --artifacts artifacts

### Inference (create predictions for the test set)

python infer_binary.py --test_csv data/testing_loan_data.csv --model artifacts/simple_mlp.pt --preproc 

## ✅ Why this repo stands out

Memory-safe tabular pipeline (sparse end-to-end; no accidental 70+ GB allocations).

Readable, modular code using standard tools recruiters expect (pandas / sklearn / PyTorch).

Documented EDA with real artifacts and succinct insights.

Clear instructions, so anyone can clone and run in minutes.

## 🔒 Data Policy
The repository does not include any datasets. Place CSVs locally under data/ and keep them private.
