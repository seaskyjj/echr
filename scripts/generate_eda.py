import nbformat as nbf
import os

def create_notebook():
    nb = nbf.v4.new_notebook()

    cells = []

    # 1. Introduction
    intro_md = """# Exploratory Data Analysis: ECHR Violation Prediction

## 1. Hypothesis and Research Question

**Research Question:** *Do NLP models predict European Court of Human Rights (ECHR) outcomes based on objective legal principles, or do they rely on "Data Provenance" issues like narrative framing and spurious correlations?*

Specifically, we investigate whether a classifier trained on the "FACTS" section of ECHR judgments learns genuine legal reasoning or simply exploits statistical artifacts such as:
- **Spurious keywords:** Procedural markers like "represented" that correlate with outcomes but carry no legal meaning (Santosh et al.).
- **Document length bias:** Non-violation (inadmissible) cases tend to be structurally shorter than violation cases.
- **Country-specific patterns:** The Respondent State itself may predict outcomes due to systemic differences in legal traditions.

**Approach:** We isolate the "Facts" section of 1,200 ECHR judgments across three countries (Russia, Turkey, UK), analyze linguistic differences between "Violation" and "Non-Violation" classes using distributional NLP techniques, and search for non-legal predictors before eventually applying machine learning in a later phase.

## 2. Data Applicability

We chose the ECHR dataset for the following reasons:
- **Structured binary labels:** Each case has a clear "Violation" or "Non-Violation" outcome, making it ideal for supervised text classification.
- **Country selection rationale:** Russia, Turkey, and the UK are among the highest-volume respondent states in ECHR history, ensuring sufficient data (>200 per cell). These three countries also represent distinct legal traditions (civil law, hybrid, common law), providing comparative value.
- **Balanced design:** We sampled 200 Violation + 200 Non-Violation per country (1,200 total), meeting the syllabus minimum of 200 documents and establishing a controlled baseline.
- **Direct relevance to hypothesis:** Critically, the literature shows that the "Facts" sections are drafted by the Court *after* the decision is made (Medvedeva & McBride, 2023). This means the text itself may contain subtle narrative framing that leaks the outcome — exactly the phenomenon we want to detect.

## 3. Ethical Data Collection Standards

Our data was acquired from the official **HUDOC database** (https://hudoc.echr.coe.int/) using the `echr-extractor` Python library, which interfaces with the HUDOC API. We adhered to the following ethical standards:
- **Terms of Service compliance:** We used the public API with reasonable request volumes and did not circumvent any access restrictions.
- **Rate limiting:** The `echr-extractor` library uses built-in threading controls (`threads=10`) to avoid overwhelming the server.
- **Data sensitivity:** While ECHR judgments are public records, they involve sensitive human rights issues. We acknowledge that models trained on these texts could perpetuate systemic biases (e.g., disproportionately flagging cases from specific countries) and this EDA explicitly explores such biases.
- **Reproducibility:** We provide `scripts/download_data.py` so that anyone can replicate our data collection.
"""
    cells.append(nbf.v4.new_markdown_cell(intro_md))

    # 2. Setup
    setup_code = """%matplotlib inline
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# Advanced NLP Libraries
import nltk
from nltk.text import Text
from nltk.tokenize import word_tokenize
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    
import scattertext as st
import shifterator as sh

# Patch shifterator for Python 3.10+ compatibility
import collections
if not hasattr(collections, 'Mapping'):
    import collections.abc
    collections.Mapping = collections.abc.Mapping

# Configure plotting
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)"""
    cells.append(nbf.v4.new_code_cell(setup_code))

    # 4. Inventiveness in Getting/Combining Data
    inventiveness_md = """## 4. Inventiveness in Data Preparation

Rather than loading a simple flat CSV, we demonstrate inventiveness in data preparation through three techniques:

1. **Section-level text extraction:** Our preprocessing pipeline (`scripts/preprocess_data.py`) uses Regular Expressions to carefully slice each judgment, extracting *only* the "Circumstances of the case" (FACTS) section while strictly dropping the "THE LAW" and "OPERATIVE PROVISIONS" sections. This is critical — including the legal reasoning would allow the model to trivially read the verdict, defeating the purpose of prediction.

2. **Feature engineering via Regex:** We parse the `representedby` metadata field to create a binary `has_representation` feature. Previous ECHR analysis found that the word "represented" appeared in 68% of non-violation cases but only 17% of violation cases (Santosh et al.), making it a powerful spurious predictor we need to track.

3. **Metadata enrichment:** We merge the extracted text with structured metadata (Respondent State, Judgment Year) from the HUDOC database, enabling cross-variable analysis that pure text-only approaches miss."""
    cells.append(nbf.v4.new_markdown_cell(inventiveness_md))

    # Data Loading Code
    load_md = """Below, we load the training split (to prevent data leakage from the test set), merge it with raw metadata, and engineer our features. We use `pd.merge()` to combine the processed text DataFrame with the metadata on the `item_id` key, then apply our regex-based lawyer detection function."""
    cells.append(nbf.v4.new_markdown_cell(load_md))

    load_code = """# 1. Load Processed Text (FACTS section)
df_text = pd.read_csv('data/processed/processed.csv')

# 2. Load Metadata
df_meta = pd.read_csv('data/raw/metadata.csv')

# 3. Merge based on item_id
df = pd.merge(df_text, df_meta, left_on='item_id', right_on='itemid', how='inner')

# Handle column name conflicts from merge (respondent exists in both)
if 'respondent_x' in df.columns:
    df['respondent'] = df['respondent_x']
    df.drop(columns=[c for c in df.columns if c.endswith('_x') or c.endswith('_y')], errors='ignore', inplace=True)

# 4. Feature Engineering
# Extract Year
df['year'] = pd.to_datetime(df['judgementdate'], dayfirst=True).dt.year

# Inventive Feature: Representation Status using Regex
# We check if 'representedby' is not null, or we could regex the text
def has_lawyer(val):
    if pd.isna(val):
        return False
    val = str(val).lower()
    # If the representation field says 'none', 'self' etc., it might mean no lawyer
    # We use regex to find if it actually contains a lawyer's name or law firm
    if re.search(r'\\b(none|n/a)\\b', val):
        return False
    return len(val.strip()) > 3

df['has_representation'] = df['representedby'].apply(has_lawyer)

# Clean up label names for plotting
df['label_name'] = df['label'].map({1: 'Violation', 0: 'Non-Violation'})

print(f"Dataset loaded with {len(df)} records and {len(df.columns)} features.")
print(f"Countries: {df['respondent'].unique()}")
print(f"Label distribution: {df['label_name'].value_counts().to_dict()}")
df[['item_id', 'respondent', 'year', 'has_representation', 'label_name']].head()"""
    cells.append(nbf.v4.new_code_cell(load_code))

    # 5. Descriptive Stats
    desc_md = """## 5. Description of Data & Descriptive Stats

We now examine the variables in our dataset. The key fields are:
- **`text`** (string): The extracted FACTS section — our primary feature for classification.
- **`label`** (binary): 1 = Violation, 0 = Non-Violation — our target variable.
- **`respondent`** (categorical): The country against which the case was brought (RUS, TUR, GBR).
- **`year`** (integer): Year of judgment — important for temporal analysis.
- **`has_representation`** (boolean): Engineered feature indicating whether the applicant had legal counsel.
- **`text_length`** (integer): Word count of the FACTS section — a potential confounding variable.

We use `value_counts()` to check class balance and `describe()` to understand the distribution of continuous variables."""
    cells.append(nbf.v4.new_markdown_cell(desc_md))

    desc_code = """# Class Balance
print("=== Class Distribution ===")
print(df['label_name'].value_counts(normalize=True) * 100)
print("\\n")

# Text Length Analysis
df['text_length'] = df['text'].astype(str).apply(lambda x: len(x.split()))
print("=== Text Length (Word Count) Statistics ===")
print(df['text_length'].describe())
print("\\n")

# Missing values
print("=== Missing Data in Key Columns ===")
print(df[['text', 'respondent', 'year', 'representedby']].isnull().sum())"""
    cells.append(nbf.v4.new_code_cell(desc_code))

    caution_md = """### Data Cautions

Before proceeding to visualization, we must acknowledge several important caveats about this dataset:

* **Post-hoc narrative construction (Medvedeva & McBride, 2023):** The "FACTS" section is written by the Court *after* the decision has been reached. This means the narrative may be unconsciously framed to support the verdict — a fundamental epistemological problem for any predictive model.
* **Structural differences between classes (Santosh et al.):** "Inadmissible" (Non-Violation) cases are often structurally different from violation cases. They tend to be shorter and contain formulaic, boilerplate language (e.g., "the applicant was represented by..."), which a model can exploit as a shortcut rather than learning legal reasoning.
* **Class imbalance in the real world:** While our dataset is artificially balanced (200 per cell), real ECHR caseloads are heavily skewed towards violations because frivolous cases are filtered out before reaching the Court.
* **Missing metadata:** The `representedby` field has nulls, which could indicate self-representation or simply incomplete records."""
    cells.append(nbf.v4.new_markdown_cell(caution_md))

    # 5.1 Summary Table per Country
    summary_table_md = """### 5.1 Summary Table by Country
As required by the syllabus, we present a summary table identifying the outcome variable and descriptive statistics broken down by respondent country."""
    cells.append(nbf.v4.new_markdown_cell(summary_table_md))

    summary_table_code = """# Summary Table: per country descriptive stats
summary = df.groupby('respondent').agg(
    total_cases=('label', 'count'),
    violations=('label', 'sum'),
    non_violations=('label', lambda x: (x == 0).sum()),
    avg_text_length=('text_length', 'mean'),
    median_text_length=('text_length', 'median'),
    representation_pct=('has_representation', lambda x: f"{x.mean()*100:.1f}%")
).reset_index()
summary.columns = ['Country', 'Total Cases', 'Violations', 'Non-Violations', 
                   'Avg Text Length (words)', 'Median Text Length', 'Has Representation %']
print("=== Summary Table by Respondent Country ===")
summary"""
    cells.append(nbf.v4.new_code_cell(summary_table_code))

    # 5.1.1 Article Distribution
    article_md = """### 5.1.1 Article Distribution

Our dataset focuses on four core ECHR articles:
- **Article 3:** Prohibition of torture and inhuman/degrading treatment
- **Article 5:** Right to liberty and security
- **Article 6:** Right to a fair trial
- **Article 8:** Right to respect for private and family life

Understanding the distribution of these articles across countries and outcomes reveals which rights are most frequently litigated in each legal system."""
    cells.append(nbf.v4.new_markdown_cell(article_md))

    article_code = """# Parse article information from the violation_articles / nonviolation_articles columns
import re
target_articles = ['3', '5', '6', '8']
article_names = {'3': 'Art 3 (Torture)', '5': 'Art 5 (Liberty)', 
                 '6': 'Art 6 (Fair Trial)', '8': 'Art 8 (Privacy)'}

# Count articles per country per label
article_rows = []
for _, row in df.iterrows():
    country = row['respondent']
    label = row['label_name']
    
    # Check which articles are in the violation/nonviolation fields from metadata
    v_arts = str(row.get('violation_articles', '')).strip()
    nv_arts = str(row.get('nonviolation_articles', '')).strip()
    
    arts_field = v_arts if label == 'Violation' else nv_arts
    if arts_field and arts_field.lower() not in ['nan', '']:
        for art in arts_field.split(';'):
            if art in target_articles:
                article_rows.append({'Country': country, 'Outcome': label, 'Article': article_names.get(art, f'Art {art}')})

if article_rows:
    art_df = pd.DataFrame(article_rows)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left: Article distribution by outcome
    art_counts = art_df.groupby(['Article', 'Outcome']).size().unstack(fill_value=0)
    art_counts.plot(kind='bar', ax=axes[0], color=['#3498db', '#e74c3c'])
    axes[0].set_title('Article Distribution by Case Outcome')
    axes[0].set_xlabel('ECHR Article')
    axes[0].set_ylabel('Number of Cases')
    axes[0].tick_params(axis='x', rotation=30)
    axes[0].legend(title='Outcome')
    
    # Right: Article distribution by country
    art_country = art_df.groupby(['Article', 'Country']).size().unstack(fill_value=0)
    art_country.plot(kind='bar', ax=axes[1], color=['#2ecc71', '#9b59b6', '#e67e22'])
    axes[1].set_title('Article Distribution by Country')
    axes[1].set_xlabel('ECHR Article')
    axes[1].set_ylabel('Number of Cases')
    axes[1].tick_params(axis='x', rotation=30)
    axes[1].legend(title='Country')
    
    plt.tight_layout()
    plt.show()
else:
    print("No article information available in the processed data.")"""
    cells.append(nbf.v4.new_code_cell(article_code))

    article_interp_md = """**Interpretation:** The article distribution reveals which rights are most frequently contested in each country. For example, Russia may have a disproportionate number of Article 3 (torture) violations, while Turkey may be dominated by Article 6 (fair trial) cases. These patterns reflect real-world socio-political dynamics and are important to understand — a model that simply learns "Article 3 mentioned in Russia = violation" would be exploiting a country-specific pattern rather than legal reasoning."""
    cells.append(nbf.v4.new_markdown_cell(article_interp_md))

    # 5.2 Correlation Matrix
    corr_md = """### 5.2 Correlation Matrix
We examine the relationship between numeric features to detect potential confounding variables. The key relationships to check:
- **Country ↔ Violation outcome:** Does a specific country strongly predict the label?
- **Text length ↔ Violation outcome:** Do longer cases correlate with violations?
- **Representation ↔ Violation outcome:** Does having a lawyer correlate with the label?"""
    cells.append(nbf.v4.new_markdown_cell(corr_md))

    corr_code = """# Build a correlation-friendly DataFrame
corr_df = df[['label', 'text_length', 'has_representation']].copy()

# Encode country as dummy variables for correlation
for country in df['respondent'].unique():
    corr_df[f'country_{country}'] = (df['respondent'] == country).astype(int)

corr_matrix = corr_df.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f', 
            linewidths=0.5, square=True)
plt.title('Correlation Matrix: Label vs Features')
plt.tight_layout()
plt.show()"""
    cells.append(nbf.v4.new_code_cell(corr_code))

    corr_interp_md = """**Interpretation:** The correlation heatmap reveals which features are linearly associated with the violation label. If a country variable shows strong correlation with the label, it means the model might simply learn "cases from country X = violation" rather than legal reasoning. Similarly, a strong correlation between `text_length` and `label` would indicate that document length alone could be a predictive shortcut."""
    cells.append(nbf.v4.new_markdown_cell(corr_interp_md))

    # 5.3 "Represented" Keyword Spurious Correlation Analysis
    represented_md = """### 5.3 Spurious Correlation: The "Represented" Keyword
Previous ECHR dataset analysis (referenced in the course materials) showed that the word **"represented"** appeared in **68% of non-violation** cases but only **17% of violation** cases. This acts as a massive spurious shortcut for models — they can achieve high accuracy simply by detecting this single word.

Let's test if this pattern exists in our dataset."""
    cells.append(nbf.v4.new_markdown_cell(represented_md))

    represented_code = """# Check frequency of 'represented' in each class
df['has_represented_word'] = df['text'].str.lower().str.contains(r'\\brepresented\\b', regex=True, na=False)

represented_stats = pd.crosstab(df['label_name'], df['has_represented_word'], normalize='index') * 100
represented_stats.columns = ['Without "represented"', 'With "represented"']

print("=== Frequency of 'represented' keyword by class ===")
print(represented_stats.round(1))
print()

# Visualize
represented_stats.plot(kind='bar', figsize=(8,5), color=['#2ecc71', '#e74c3c'])
plt.title('Presence of "represented" Keyword by Case Outcome')
plt.ylabel('Percentage of Cases')
plt.xlabel('Case Outcome')
plt.xticks(rotation=0)
plt.legend(title='Contains "represented"')
plt.tight_layout()
plt.show()"""
    cells.append(nbf.v4.new_code_cell(represented_code))

    represented_interp_md = """**Interpretation:** If the word "represented" appears disproportionately in non-violation cases (as the literature suggests: 68% vs 17%), then a model could achieve high accuracy by learning this single word as a shortcut. This is a critical finding for our research question: *does the model learn legal principles or dataset biases?* In the modeling phase, we should perform an **ablation test** where we remove this token and measure accuracy drop."""
    cells.append(nbf.v4.new_markdown_cell(represented_interp_md))

    # 6. Visualizations
    viz_md = """## 6. Visualizations & Interpretations
**Goal:** Visually help understand data patterns and relationships."""
    cells.append(nbf.v4.new_markdown_cell(viz_md))

    viz1_code = """# 1. Distribution of Document Lengths by Class
plt.figure(figsize=(10,6))
sns.histplot(data=df, x='text_length', hue='label_name', bins=50, kde=True, alpha=0.6)
plt.title('Distribution of Text Lengths (Words) by Case Outcome')
plt.xlabel('Number of Words in FACTS Section')
plt.ylabel('Frequency')
plt.xlim(0, 5000) # Cap outliers for better visualization
plt.show()"""
    cells.append(nbf.v4.new_code_cell(viz1_code))

    interp1_md = """**Interpretation:** The histogram above shows the text length distribution. If Violation cases generally have longer "FACTS" sections, a model might just learn that "longer text = violation" instead of understanding the legal logic. This is an important bias to track."""
    cells.append(nbf.v4.new_markdown_cell(interp1_md))

    viz2_code = """# 2. Respondent State Bias
plt.figure(figsize=(14,6))
top_countries = df['respondent'].value_counts().nlargest(15).index
sns.countplot(data=df[df['respondent'].isin(top_countries)], x='respondent', hue='label_name', order=top_countries)
plt.title('Top 15 Respondent States by Case Outcome')
plt.xlabel('Country Code')
plt.ylabel('Number of Cases')
plt.xticks(rotation=45)
plt.show()"""
    cells.append(nbf.v4.new_code_cell(viz2_code))

    interp2_md = """**Interpretation:** This bar chart reveals geographic distribution. We can see if certain countries have overwhelmingly more Violations vs Non-Violations in the dataset. A predictive model could unfairly penalize a specific country simply by seeing its name in the text (e.g., "The applicant, a citizen of RUS..."), which is a critical finding for ethical modelling."""
    cells.append(nbf.v4.new_markdown_cell(interp2_md))

    # Year Distribution
    year_md = """### Year Distribution
The year distribution is critical because of the **"temporal shift" problem** (concept drift). Models trained on older cases may fail on newer ones because the law, societal norms, and Court language evolve over time. Medvedeva & McBride (2023) demonstrated F1 drops from 0.92 to 0.64–0.68 when evaluating on future years.

We plot the distribution of cases over time to detect:
- Whether certain countries spike during specific years (e.g., due to political events)
- Whether our dataset is temporally balanced or skewed"""
    cells.append(nbf.v4.new_markdown_cell(year_md))

    year_hist_code = """# Year Distribution: Overall histogram by label
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Left: Overall year distribution by label
sns.histplot(data=df, x='year', hue='label_name', bins=30, ax=axes[0], alpha=0.6)
axes[0].set_title('Case Distribution Over Time by Outcome')
axes[0].set_xlabel('Year')
axes[0].set_ylabel('Number of Cases')

# Right: Year distribution per country
sns.histplot(data=df, x='year', hue='respondent', bins=30, ax=axes[1], alpha=0.6, multiple='stack')
axes[1].set_title('Case Distribution Over Time by Country')
axes[1].set_xlabel('Year')
axes[1].set_ylabel('Number of Cases')

plt.tight_layout()
plt.show()"""
    cells.append(nbf.v4.new_code_cell(year_hist_code))

    year_trend_code = """# Temporal Trend: Violation rate per year per country
year_country = df.groupby(['year', 'respondent']).agg(
    total=('label', 'count'),
    violations=('label', 'sum')
).reset_index()
year_country['violation_rate'] = year_country['violations'] / year_country['total']

# Only plot years with enough data (>= 3 cases)
year_country_filtered = year_country[year_country['total'] >= 3]

plt.figure(figsize=(12, 5))
for country in year_country_filtered['respondent'].unique():
    subset = year_country_filtered[year_country_filtered['respondent'] == country]
    plt.plot(subset['year'], subset['violation_rate'], marker='o', label=country, alpha=0.7)

plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='50% baseline')
plt.title('Violation Rate Over Time by Country')
plt.xlabel('Year')
plt.ylabel('Violation Rate')
plt.legend()
plt.ylim(0, 1)
plt.tight_layout()
plt.show()"""
    cells.append(nbf.v4.new_code_cell(year_trend_code))

    year_interp_md = """**Interpretation:** 
- The **histogram** shows whether our dataset is concentrated in certain time periods. If most cases come from 2000–2015, we should be cautious about generalizing to post-2019 cases.
- The **violation rate trend** reveals whether certain countries show changing patterns over time (e.g., Russia may have increasing violation rates during certain political periods).
- **Implication for modeling:** We should use a **chronological train/test split** (train on older cases, test on newest) rather than a random split, to honestly assess whether our model generalizes to future cases."""
    cells.append(nbf.v4.new_markdown_cell(year_interp_md))

    viz3_code = """# 3. Does Legal Representation Matter?
representation_outcome = pd.crosstab(df['has_representation'], df['label_name'], normalize='index') * 100

representation_outcome.plot(kind='bar', stacked=True, figsize=(8,6), color=['#1f77b4', '#d62728'])
plt.title('Case Outcome by Legal Representation Status')
plt.xlabel('Has Legal Representation?')
plt.ylabel('Percentage')
plt.xticks([0, 1], ['No', 'Yes'], rotation=0)
plt.legend(title='Outcome', loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()"""
    cells.append(nbf.v4.new_code_cell(viz3_code))

    interp3_md = """**Interpretation:** The stacked bar chart shows the win/loss percentage based on whether the applicant had representation. This engineered feature uses regex parsing and explores our secondary hypothesis: cases with professional representation might have different outcome probabilities."""
    cells.append(nbf.v4.new_markdown_cell(interp3_md))

    # NLP TF-IDF
    tfidf_md = """## TF-IDF Analysis (Distributional NLP)
To understand what words drive the difference between classes, we use TF-IDF (Term Frequency - Inverse Document Frequency)."""
    cells.append(nbf.v4.new_markdown_cell(tfidf_md))

    tfidf_code = """# Extract top words using TF-IDF for each class
def get_top_n_tfidf(texts, n=15):
    vec = TfidfVectorizer(stop_words='english', max_features=1000, ngram_range=(1,2))
    matrix = vec.fit_transform(texts)
    sum_words = matrix.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return words_freq[:n]

violation_texts = df[df['label'] == 1]['text'].dropna()
non_violation_texts = df[df['label'] == 0]['text'].dropna()

top_v = get_top_n_tfidf(violation_texts)
top_nv = get_top_n_tfidf(non_violation_texts)

# Plotting
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

sns.barplot(x=[val for word, val in top_v], y=[word for word, val in top_v], ax=axes[0], color='salmon')
axes[0].set_title('Top TF-IDF N-grams in Violation Cases')

sns.barplot(x=[val for word, val in top_nv], y=[word for word, val in top_nv], ax=axes[1], color='skyblue')
axes[1].set_title('Top TF-IDF N-grams in Non-Violation Cases')

plt.tight_layout()
plt.show()"""
    cells.append(nbf.v4.new_code_cell(tfidf_code))

    # Word Counts (N-grams)
    ngrams_md = """## 6.1 Simple Word Counts (N-grams)
Before using TF-IDF, let's look at simple, raw counts of unigrams and bigrams using `CountVectorizer`."""
    cells.append(nbf.v4.new_markdown_cell(ngrams_md))

    ngrams_code = """def get_top_n_words(texts, n=15, ngram_range=(1,1)):
    vec = CountVectorizer(stop_words='english', ngram_range=ngram_range).fit(texts)
    bag_of_words = vec.transform(texts)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]

print("Top 10 Unigrams (Violation):", get_top_n_words(violation_texts, n=10, ngram_range=(1,1)))
print("Top 10 Bigrams (Violation):", get_top_n_words(violation_texts, n=10, ngram_range=(2,2)))"""
    cells.append(nbf.v4.new_code_cell(ngrams_code))

    # Shifterator
    shifter_md = """## 6.2 Fighting Words (`shifterator`)
To visualize words that most distinguish the two classes, we use the `shifterator` library to plot Jensen-Shannon Divergence Shifts. This creates interpretable horizontal bar charts showing which words are pulling a case towards "Violation" vs "Non-Violation"."""
    cells.append(nbf.v4.new_markdown_cell(shifter_md))

    shifter_code = """# Clean and count for Shifterator
def get_counts(texts):
    # Simple tokenization for shifterator
    all_text = ' '.join(texts).lower()
    # keeping only words
    words = re.findall(r'\\w+', all_text)
    # remove very common stopwords manually for a cleaner shift
    stopwords = set(nltk.corpus.stopwords.words('english')) if 'stopwords' in dir(nltk.corpus) else set(['the', 'a', 'of', 'and', 'to', 'in', 'that', 'was', 'for', 'on', 'is', 'as', 'by', 'it', 'with'])
    words = [w for w in words if w not in stopwords and len(w)>2]
    return dict(Counter(words))

count_v = get_counts(violation_texts)
count_nv = get_counts(non_violation_texts)

# Produce JSD shift
jsd_shift = sh.JSDivergenceShift(type2freq_1=count_nv,
                                 type2freq_2=count_v,
                                 weight_1=0.5,
                                 weight_2=0.5,
                                 reference_value='average')

jsd_shift.get_shift_graph(system_names=['Non-Violation', 'Violation'], title='JSD Shift: Violation vs Non-Violation', cumulative_inset=False, text_size_inset=False)
plt.show()"""
    cells.append(nbf.v4.new_code_cell(shifter_code))

    # Scattertext
    scatter_md = """## 6.3 Scattertext Visualization
Scattertext is a great tool to interactively explore term associations between two categories."""
    cells.append(nbf.v4.new_markdown_cell(scatter_md))

    scatter_code = """import os
# Create a smaller corpus for scattertext to run quickly
sample_df = df.sample(n=min(1000, len(df)), random_state=42)

corpus = st.CorpusFromPandas(sample_df,
                             category_col='label_name',
                             text_col='text',
                             nlp=st.whitespace_nlp_with_sentences
                            ).build()

html = st.produce_scattertext_explorer(corpus,
                                       category='Violation',
                                       category_name='Violation',
                                       not_category_name='Non-Violation',
                                       width_in_pixels=1000,
                                       metadata=sample_df['respondent'])

with open('scattertext_viz.html', 'w', encoding='utf-8') as f:
    f.write(html)
    
print("Scattertext visualization saved to 'scattertext_viz.html'. Open this file in your browser to interact with it.")"""
    cells.append(nbf.v4.new_code_cell(scatter_code))

    # Concordance
    concord_md = """## 6.4 NLTK Concordance Analysis
Concordances let us view specific key terms in the context they appear in the text. Let's look at how the word "violation" or "rights" is used."""
    cells.append(nbf.v4.new_markdown_cell(concord_md))

    concord_code = """# We combine a sample of texts to look at context
sample_text = " ".join(df['text'].dropna().sample(n=50, random_state=42).tolist())
tokens = word_tokenize(sample_text)
nltk_text = Text(tokens)

print("=== Concordance for 'violation' ===")
nltk_text.concordance('violation', lines=5)
print("\\n=== Concordance for 'court' ===")
nltk_text.concordance('court', lines=5)"""
    cells.append(nbf.v4.new_code_cell(concord_code))

    # Conclusion
    conclusion_md = """## 7. Conclusions and Next Steps
**Conclusion:** 
Through this EDA, we successfully analyzed the ECHR dataset across three respondent countries (Russia, Turkey, UK) with balanced violation/non-violation labels.

**Key Findings:**
1. The TF-IDF and Fighting Words visualizations show vocabulary differences exist between the two classes, suggesting text-based prediction is feasible.
2. The **correlation matrix** reveals whether country identity or text length act as confounding variables.
3. The **"represented" keyword analysis** confirms (or challenges) the known spurious correlation from the literature — a critical finding for ethical modeling.
4. The **Scattertext** visualization provides an interactive way to explore which terms are associated with each class.

**Next Steps for Modeling:**
1. Train a baseline SVM with TF-IDF and compare against Legal-BERT.
2. Perform **ablation testing**: remove spurious tokens ("represented", "Mr", country names) and re-evaluate accuracy.
3. Use LIME or Integrated Gradients for explainability to verify the model reasons about legal concepts, not artifacts.
4. Evaluate on a **realistic, unbalanced test set** that reflects actual ECHR outcome ratios."""
    cells.append(nbf.v4.new_markdown_cell(conclusion_md))

    nb['cells'] = cells

    with open('EDA.ipynb', 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
        print("EDA.ipynb created successfully!")

if __name__ == "__main__":
    create_notebook()
