# Experiment Conclusions

## Scope

This note consolidates the current modeling results on the fixed `train / val / test` split built from the latest `705` processed cases.

It is intended to answer one question clearly:

- what has been established already
- what is still uncertain
- what the next stage should focus on

## Fixed Data Split

Current split size:

- train: `493`
- val: `106`
- test: `106`

Current processed dataset:

- total processed cases: `705`

## Final Baselines So Far

### Classical models

Test-set results from [test_metrics.csv](/Users/jiejia/Programs/echr/results/classical/test_metrics.csv):

| model | accuracy | precision | recall | F1 |
|---|---:|---:|---:|---:|
| Linear SVM (word n-gram) | `0.7830` | `0.8378` | `0.8493` | `0.8435` |
| Naive Bayes (word n-gram) | `0.7453` | `0.8194` | `0.8082` | `0.8138` |

Current conclusion:

- `Linear SVM` is the strongest baseline on the present split.
- The classical baselines are stable and still outperform the transformer runs tried so far.

### Legal-BERT (`512` tokens)

The strongest `Legal-BERT` setting found in the current round was:

- `weighted_loss=True`
- `threshold_tuning=False`
- `epochs=3`
- `learning_rate=1e-5`
- `weight_decay=0.01`

Observed test result:

- accuracy: `0.6887`
- precision: `0.8030`
- recall: `0.7260`
- F1: `0.7626`

Current conclusion:

- `Legal-BERT-512` did not beat the classical baselines.
- The main bottleneck is not only data size but also input truncation.

### Longformer-4096

`Longformer-4096` was added to test whether the poor transformer performance was mainly caused by the `512`-token window.

Representative non-collapsed run:

- accuracy: `0.6981`
- precision: `0.8475`
- recall: `0.6849`
- F1: `0.7576`

Threshold tuning was also tested with:

- `balanced_accuracy`
- `macro_f1`

but neither gave a material improvement on the test split.

Current conclusion:

- `Longformer-4096` changed the prediction pattern and reduced the degenerate `recall = 1.0` behavior seen in early Legal-BERT runs.
- However, it still did not surpass the classical baselines.
- The transformer line is therefore no longer blocked by context length alone; dataset size and label structure remain limiting factors.

## What Has Been Established

### 1. The classical baselines are currently the strongest models

This is no longer a tentative observation. It has been reproduced across multiple transformer runs and thresholding variants.

### 2. `512`-token truncation is a real modeling bottleneck

The earlier truncation diagnosis showed that roughly two-thirds of the cases exceed `512` tokens, which makes `Legal-BERT-512` an incomplete full-text model for this task.

### 3. Extending context to `4096` helps, but does not solve the full problem

`Longformer-4096` is a better architectural fit for long judgments, but the gain is still not enough to overtake the classical baselines under the current dataset size and label distribution.

### 4. Threshold tuning is not the main missing piece

Threshold tuning was tried with different objectives. Validation metrics could improve, but the gains did not transfer reliably to the test split.

That means the current bottleneck is not mainly a threshold-selection issue.

## Interpretation

The project now has a defensible intermediate conclusion:

- full-text classical baselines remain the strongest option on the current dataset
- `Legal-BERT-512` is constrained by severe truncation
- `Longformer-4096` addresses truncation but still does not beat the classical baselines
- therefore the next bottleneck is primarily the data, not another small round of hyperparameter tuning

## Recommended Next Step

The next stage should focus on data expansion, not another long tuning round on the same `705` cases.

Priority order:

1. add more countries
2. increase `Article 6` coverage
3. increase `non-violation` coverage
4. keep the current fixed split logic and rerun the same baselines after expansion

## Practical Decision

For the current report or notebook narrative, the cleanest model set is:

- `Naive Bayes`
- `Linear SVM`
- `Legal-BERT-512`
- `Longformer-4096`

And the cleanest research conclusion at this stage is:

- long-context transformers are more appropriate than `512`-token BERT for ECHR judgments
- but with the current sample size they still do not outperform classical full-text baselines
- expanding the dataset is now the highest-value next move
