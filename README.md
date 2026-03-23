### **Project Title:**
**The Artificial Judge: Evaluating Legal Reasoning and Spurious Correlations in NLP Models of the European Court of Human Rights**

### **1. Project Overview & Motivation**
The European Court of Human Rights (ECHR) faces a significant backlog of cases, prompting interest in AI-driven tools to triage or predict judicial outcomes. While recent studies demonstrate that Natural Language Processing (NLP) models can predict outcomes with high accuracy (79%–82%), significant questions remain regarding *how* these models reach their conclusions.

This project investigates the "black box" of Legal Judgment Prediction (LJP). Specifically, we aim to determine if predictive models are learning genuine legal principles or merely exploiting linguistic artifacts and spurious correlations (e.g., the presence of specific procedural keywords or metadata). Drawing on the course’s focus on **data provenance** and **data generation**, we critique the reliance on post-hoc judicial texts as input data, acknowledging that the "facts" in these documents are often constructed narratives rather than objective reality.

### **2. Key Research Questions**
1.  **Predictive Accuracy:** How do traditional machine learning models (Linear SVM with N-grams) compare against modern Transformer-based models (Legal-BERT) in predicting ECHR violation outcomes?
2.  **Explainability & Legal Reasoning:** Do these models rely on legally substantive text (e.g., specific fact patterns) or spurious correlations (e.g., procedural terms like "represented by," or country names)?
3.  **Generalization:** Does the model's performance degrade when tested on cases from a future time period, indicating a failure to learn evolving legal standards (the "temporal shift" problem)?

### **3. Data Source & Provenance**
We will utilize the **ECHR-OD (Open Data)** or **LexGLUE** datasets, which provide structured access to ECHR judgments scraped from the HUDOC database.
*   **Dataset Size:** We target approximately 500–2,000 cases to meet the course requirement for a non-textual or textual dataset.
*   **Data Provenance Issue:** We acknowledge a critical limitation identified in the literature: our input data (the "Facts" section of judgments) is generated *after* the decision is made, potentially leaking the outcome. We will address this by strictly removing "Law" and "Operative Provisions" sections during preprocessing.

### **4. Methodology**
We will frame this as a **binary classification task** (Violation vs. No Violation).

**A. Exploratory Data Analysis (EDA):**
*   Analyze the distribution of case outcomes to check for class imbalance.
*   Examine the correlation between specific Articles (e.g., Article 3, 6, 8) and text length or respondent state.

**B. Modeling:**
1.  **Baseline Model:** Replicate the approach of Aletras et al. (2016) using TF-IDF features and a Linear Support Vector Machine (SVM).
2.  **Advanced Model:** Fine-tune a pre-trained **Legal-BERT** model, which has shown superior performance in identifying legal concepts compared to generic models.

**C. Evaluation & Critique:**
*   **Spurious Correlation Check:** We will test if the model over-relies on "distractor" tokens. For example, previous studies found the word "represented" strongly correlated with specific outcomes due to inadmissible cases often listing legal representation differently. We will attempt to replicate this finding or identify new spurious tokens.
*   **Explainability:** We will use techniques like **Integrated Gradients** or **LIME** to visualize which parts of the text drive the prediction. We will analyze if these highlighted regions align with what legal experts would consider relevant facts, or if they point to irrelevant metadata.

### **6. References**
*   **Aletras et al. (2016).** Predicting judicial decisions of the European Court of Human Rights: a Natural Language Processing perspective. *PeerJ Computer Science*.
*   **Chalkidis et al. (2022).** LexGLUE: A Benchmark Dataset for Legal Language Understanding in English. *ACL*.
*   **Medvedeva & McBride (2023).** Legal Judgment Prediction: If You Are Going to Do It, Do It Right. *NLLP @ ACL*.
*   **Santosh et al. (2022).** Deconfounding Legal Judgment Prediction for European Court of Human Rights Cases. *EMNLP*.
*   **Quemy & Wrembel (2021).** ECHR-DB: On building an integrated open repository of legal documents. *Information Systems*.

### **6. Execution Steps**

#### **1. Environment Setup**
Ensure you have the necessary dependencies installed, including `torch`, `transformers`, `echr-extractor`, `pandas`, and `scikit-learn`.

```bash
uv add echr-extractor
```

#### **2. Data Acquisition**
Download cases from the HUDOC database using the `echr-extractor` wrapper script.
```bash
# Download 100 cases (default)
python scripts/download_data.py --count 100
```
This will save metadata to `data/raw/metadata.csv` and full text to `data/raw/full_text.json`.

To download the next country expansion wave and merge it with the current raw dataset:
```bash
python scripts/download_increments_data.py \
  --existing_raw_dir data/raw \
  --output_dir data/incremental/raw
```

Default increment countries are:
- `FRA`
- `ROU`
- `POL`
- `DEU`
- `BEL`

This writes:
- `data/incremental/raw/metadata_increment.csv`
- `data/incremental/raw/full_text_increment.json`
- `data/incremental/raw/metadata_merged.csv`
- `data/incremental/raw/full_text_merged.json`

If you want to replace the base raw dataset with the merged version:
```bash
python scripts/download_increments_data.py \
  --existing_raw_dir data/raw \
  --output_dir data/incremental/raw \
  --write_merged_to_base
```

#### **3. Data Preprocessing**
Parse the downloaded raw data and extract the "FACTS" section.
```bash
python scripts/preprocess_data.py
```
The processed dataset will be saved to `data/processed/processed.csv`.

The preprocessing step now preserves co-respondent information instead of collapsing it to a single country. It keeps:
- `respondent`
- `is_multi_respondent`
- `respondent_list`
- `source_batch`
- `source_type`

#### **4. Reproducible Data Split**
Create fixed-seed train/val/test splits and a metadata-enriched test file.
```bash
python scripts/split_data.py --seed 42
```
This writes:
- `data/processed/train.csv`
- `data/processed/val.csv`
- `data/processed/test.csv`
- `data/processed/metadata_joined_test.csv`

Split behavior is now slightly more robust for expanded country coverage:
- outer split stratifies by `respondent + label` where support is sufficient
- rare respondent-label combinations are collapsed for the outer split
- val/test split uses label-only stratification for stability

#### **5. Classical Baselines**
Train Multinomial Naive Bayes and Linear SVM baselines with word/char n-grams.
```bash
python src/train_classical.py --split_dir data/processed --output_dir results/classical --seed 42
```

#### **6. Model Training**
Train the Legal-BERT classifier on the processed splits.
```bash
# Run training (example with small batch size for verification)
python src/train.py \
  --split_dir data/processed \
  --epochs 3 \
  --batch_size 8 \
  --learning_rate 2e-5 \
  --weight_decay 0.01 \
  --output_dir results/legal_bert
```
To run on CPU (if no GPU available or OOM issues):
```bash
export CUDA_VISIBLE_DEVICES=""
python src/train.py --split_dir data/processed --epochs 1 --batch_size 2
```

Train a long-context Longformer baseline on the same split.
```bash
python src/train.py \
  --split_dir data/processed \
  --model_name allenai/longformer-base-4096 \
  --max_len 4096 \
  --epochs 2 \
  --batch_size 1 \
  --grad_accum_steps 4 \
  --learning_rate 1e-5 \
  --weight_decay 0.01 \
  --use_weighted_loss \
  --gradient_checkpointing \
  --eval_accumulation_steps 8 \
  --output_dir results/longformer_4096
```

Optional imbalance and decision-boundary controls:
```bash
# 1) Weighted loss only
python src/train.py \
  --split_dir data/processed \
  --epochs 3 \
  --batch_size 4 \
  --use_weighted_loss

# 2) Weighted sampler only
python src/train.py \
  --split_dir data/processed \
  --epochs 3 \
  --batch_size 4 \
  --use_weighted_sampler

# 3) Threshold tuning only
python src/train.py \
  --split_dir data/processed \
  --epochs 3 \
  --batch_size 4 \
  --use_threshold_tuning \
  --threshold_metric balanced_accuracy \
  --threshold_grid_size 181

# 4) Combine all three
python src/train.py \
  --split_dir data/processed \
  --epochs 3 \
  --batch_size 4 \
  --use_weighted_loss \
  --use_weighted_sampler \
  --use_threshold_tuning
```

When threshold tuning is enabled, the script writes:
- `test_predictions_default.csv` / `test_metrics_default.json` for the raw argmax output
- `test_predictions_threshold_tuned.csv` / `test_metrics_threshold_tuned.json` for the validation-tuned threshold
- `test_predictions.csv` / `test_metrics.json` as the active output for the current run

Supported threshold-tuning objectives:
- `balanced_accuracy` (default and recommended for avoiding all-positive collapse)
- `macro_f1`
- `f1`

#### **7. Kaggle-Friendly Paths**
All core scripts now support a dataset root and a writable working root. This is the intended pattern on Kaggle:
```bash
python scripts/preprocess_data.py \
  --dataset_dir /kaggle/input/datasets/seaskyjj/echr_data \
  --working_dir /kaggle/working

python scripts/split_data.py \
  --dataset_dir /kaggle/input/datasets/seaskyjj/echr_data \
  --working_dir /kaggle/working \
  --seed 42

python src/train_classical.py \
  --dataset_dir /kaggle/input/datasets/seaskyjj/echr_data \
  --working_dir /kaggle/working \
  --seed 42

python src/train.py \
  --dataset_dir /kaggle/input/datasets/seaskyjj/echr_data \
  --working_dir /kaggle/working \
  --epochs 3 --batch_size 4 --seed 42

python src/train.py \
  --dataset_dir /kaggle/input/datasets/seaskyjj/echr_data \
  --working_dir /kaggle/working \
  --model_name allenai/longformer-base-4096 \
  --max_len 4096 \
  --epochs 2 \
  --batch_size 1 \
  --grad_accum_steps 4 \
  --use_weighted_loss \
  --gradient_checkpointing \
  --eval_accumulation_steps 8 \
  --output_dir /kaggle/working/results/longformer_4096
```

#### **8. Country-Level Bias Analysis Rule**
The current default project rule is:

- keep co-respondent cases in raw and modeling tables
- exclude co-respondent cases from the main country-level bias analysis
- use duplicated country assignment only as a later robustness check

This rule is reflected in:
- `scripts/preprocess_data.py`
- `scripts/analyze_bias.py`
- `Model_Comparison.ipynb`
