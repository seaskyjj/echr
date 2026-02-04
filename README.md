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
pip install echr-extractor
```

#### **2. Data Acquisition**
Download cases from the HUDOC database using the `echr-extractor` wrapper script.
```bash
# Download 100 cases (default)
python scripts/download_data.py --count 100
```
This will save metadata to `data/raw/metadata.csv` and full text to `data/raw/full_text.json`.

#### **3. Data Preprocessing**
Parse the downloaded raw data, extract the "FACTS" section, label the cases, and split into train/val/test sets.
```bash
python scripts/preprocess_data.py
```
processed files will be saved in `data/processed/`.

#### **4. Model Training**
Train the Legal-BERT classifier on the processed data.
```bash
# Run training (example with small batch size for verification)
python src/train.py --epochs 3 --batch_size 8 --output_dir results
```
To run on CPU (if no GPU available or OOM issues):
```bash
export CUDA_VISIBLE_DEVICES=""
python src/train.py --epochs 1 --batch_size 2
```
