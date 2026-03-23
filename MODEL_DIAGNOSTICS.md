# Model Diagnostics and Next-Step Plan

## Scope

This note records the main modeling issues observed so far in the ECHR outcome prediction project and explains the reasoning behind the next-step decisions.

It focuses on:

- why `Legal-BERT` underperformed the classical baselines
- why the `512`-token window is a real bottleneck
- what it means when `recall = 1.0`
- why simply changing batch size is not a substantive fix
- what the next model-development priorities should be

## Current Model Snapshot

Using the current fixed split:

- `train`: 493
- `val`: 106
- `test`: 106

Classical baselines on the test split:

- `Naive Bayes`
  - accuracy: `0.7453`
  - precision: `0.8286`
  - recall: `0.7945`
  - F1: `0.8112`
- `Linear SVM`
  - accuracy: `0.7453`
  - precision: `0.8382`
  - recall: `0.7808`
  - F1: `0.8085`

Best `Legal-BERT` configuration found so far:

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

This means that under the current setup, the classical baselines still outperform `Legal-BERT`.

## Why Legal-BERT Underperformed

Two bottlenecks are clearly present.

### 1. Small supervised dataset

The current processed dataset has `705` cases. This is usable for baseline classification, but still small for stable transformer fine-tuning, especially when:

- the documents are long
- labels are imbalanced
- future plans include country-level and time-based bias analysis

### 2. Severe input truncation

`Legal-BERT` is a BERT-family model with a practical maximum input length of `512` tokens.

That was not a minor technical detail. It turned out to be a core modeling bottleneck.

## 512-Token Truncation Diagnosis

Measured with the current split files:

| split | `% over 512` |
|---|---:|
| train | `67.75%` |
| val | `63.21%` |
| test | `69.81%` |

Average tokens lost among truncated cases:

| split | average tokens lost |
|---|---:|
| train | `1803.7` |
| val | `2339.4` |
| test | `2069.6` |

This means that for roughly two-thirds of the dataset, `Legal-BERT` is not seeing the full text. In many cases it is not even close.

### Truncation by respondent state on the test split

| respondent | `% over 512` |
|---|---:|
| GBR | `88.57%` |
| TUR | `84.00%` |
| RUS | `47.83%` |

This matters for both prediction and bias analysis:

- the model does not see comparable amounts of text across countries
- any country-level performance difference may partly reflect truncation severity, not just legal signal

## What Happens at 4096 Tokens

If the context window is increased to `4096`, truncation drops sharply.

| split | `% over 512` | `% over 4096` |
|---|---:|---:|
| train | `67.75%` | `8.72%` |
| val | `63.21%` | `12.26%` |
| test | `69.81%` | `12.26%` |

Test split by respondent state at `4096`:

| respondent | `% over 4096` |
|---|---:|
| GBR | `25.71%` |
| TUR | `12.00%` |
| RUS | `2.17%` |

This is the main reason a `Longformer-4096` baseline was added. Moving from `512` to `4096` is not cosmetic. It changes the amount of legal text the model can actually use.

## Why More Data Alone Is Not Enough

Adding more data is still necessary, but it does not fully solve the transformer bottleneck by itself.

If the model still only sees the first `512` tokens:

- the sample size problem improves
- but the long-document coverage problem remains

So there are really two separate problems:

1. dataset scale and balance
2. long-context coverage

Both need to be addressed.

## What `recall = 1.0` Means

When a binary classifier shows:

- `recall = 1.0`
- while accuracy is only moderate

the first thing to check is whether it is predicting almost everything as the positive class.

In this project, the positive class is `violation = 1`.

If the model predicts nearly all cases as `1`, then:

- it catches all true positives
- recall becomes `1.0`
- but it fails to identify negatives properly

This is what was meant by the model “collapsing to a single-class decision rule”.

### Why this can happen

Common reasons in this project context:

- class imbalance
- limited data
- strong positive-class prior
- default decision rule (`argmax` or implicit threshold `0.5`)
- unstable fine-tuning on a small legal dataset

### Why `weighted_loss=True` may still leave recall at 1

`weighted loss` changes the training objective, but it does not guarantee a balanced decision boundary.

It can help, but if:

- the model’s output probabilities are still strongly shifted toward the positive class
- and the prediction rule is still the default one

then recall may remain very high or even stay at `1.0`.

So `weighted loss` is a useful intervention, but not a complete solution by itself.

## Why Changing Batch Size Is Not a Real Fix

For example, changing from:

- `batch_size = 1`, `grad_accum_steps = 4`

to:

- `batch_size = 2`, `grad_accum_steps = 2`

mainly changes the optimization dynamics:

- gradient estimates may become more stable
- throughput may improve
- training may be slightly smoother

But this does **not** directly change:

- the model architecture
- the decision rule
- the long-context bottleneck
- the thresholding logic

So batch-size changes are useful for training stability, but they are not the main answer to the current prediction-shape problem.

## Threshold Tuning: Why It Did Not Help

Threshold tuning was tested after `weighted loss`, but it made results worse.

The likely reason is that the current threshold search objective is not aligned with the real problem.

If the threshold is chosen to optimize the positive-class F1 only, then it can still favor a decision rule that overpredicts the positive class.

For this project, the better threshold objectives are likely:

- `balanced_accuracy`
- `macro_f1`

rather than only the standard positive-class binary F1.

So the current judgment is:

- threshold tuning is not useless
- but the current threshold-tuning objective is probably not the right one

## Why Longformer-4096 Was Added

`Longformer-4096` was introduced as a stronger long-context baseline because it addresses the main technical limitation of `Legal-BERT-512`.

This is useful for the project for two reasons:

1. it tests whether the poor transformer performance is largely caused by truncation
2. it gives a fairer transformer comparison against the full-text classical models

The default `Longformer` settings were kept conservative:

- `max_len = 4096`
- `batch_size = 1`
- `grad_accum_steps = 4`
- `gradient_checkpointing = True`
- `eval_accumulation_steps = 8`
- `weighted_loss = True`

The notebook and `src/train.py` were also updated to avoid unstable multi-GPU `DataParallel` behavior in Kaggle by forcing single-GPU mode for Longformer runs.

## Recommended Next Steps

### Priority 1: Evaluate Longformer-4096 properly

Do not judge the long-context route from one unstable run.

The main question is:

- does `Longformer-4096` reduce the degenerate positive-class bias relative to `Legal-BERT-512`?

Recommended checks after each run:

- `pd.crosstab(y_true, y_pred)`
- predicted positive rate
- precision / recall tradeoff
- country-level behavior

### Priority 2: Improve threshold tuning objective

If the model still skews heavily toward `violation = 1`, the next high-value change is not more batch-size tweaking.

Instead, threshold tuning should be changed to optimize:

- `balanced_accuracy`
  or
- `macro_f1`

This is a more direct response to the “recall is 1, negatives are missed” problem.

### Priority 3: Expand the dataset

Model-side improvements should continue, but dataset expansion is still necessary for the project’s research goals.

Especially important:

- more countries
- more `Article 6` cases
- more `non-violation` cases
- better year coverage

This is required not only for prediction quality, but also for any serious analysis of:

- country-level bias
- temporal bias
- article-specific bias

### Priority 4: Keep the current strongest classical models as reference baselines

The classical baselines are not just placeholders. They are currently the most stable and strongest reference point.

So future transformer experiments should be evaluated against:

- `Naive Bayes`
- `Linear SVM`

and not only against earlier transformer runs.

## Practical Decision Rule

At this stage, the project should proceed under this logic:

1. Keep the current best `Legal-BERT-512` run as the short-context transformer baseline.
2. Run and evaluate `Longformer-4096` as the long-context transformer baseline.
3. If `Longformer-4096` still has `recall = 1` or near-collapse behavior, fix threshold selection before doing more fine-tuning sweeps.
4. Expand the dataset in parallel, especially for `Article 6` and cross-country analysis.

## Bottom Line

The main modeling lesson so far is not simply that “transformers are worse”.

The stronger conclusion is:

- the current classical models benefit from full-text access
- `Legal-BERT-512` is heavily constrained by truncation
- long-context modeling is therefore methodologically necessary
- dataset expansion is still required for robust bias analysis

That is the current rationale for moving from `Legal-BERT-512` to `Longformer-4096`, while continuing to treat the classical baselines as the main reference point.
