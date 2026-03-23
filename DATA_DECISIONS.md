# Data Decisions

## Scope

This file records the data-side decisions that are now fixed for the next round of downloading, preprocessing, and country-level analysis.

## Decision 1: Next Countries To Add

The next expansion wave is fixed as:

- `FRA`
- `ROU`
- `POL`
- `DEU`
- `BEL`

Status:

- completed on `2026-03-22`
- merged into the main raw dataset

Reason:

- they add meaningful `Article 6` volume
- they add more `non-violation` cases than the current `RUS / TUR` heavy profile
- they improve the country spread needed for later bias analysis

Not selected for the next wave:

- `ITA`
- `UKR`

Reason:

- both add a lot of volume
- but both are strongly skewed toward `violation`
- they are less useful than the five selected countries for balanced modeling and `country x time` analysis

## Decision 2: Co-Respondent Cases

Default rule:

1. keep co-respondent cases unchanged in raw data
2. keep one case per `itemid` in modeling tables
3. mark co-respondent cases explicitly
4. exclude co-respondent cases from the main country-level bias analysis

Examples already present in metadata:

- `DNK;TUR`
- `MDA;RUS`

Reason:

- duplicating one case across multiple countries would artificially increase sample size
- keeping them unchanged preserves fidelity to HUDOC
- excluding them from the main country-level bias analysis gives cleaner interpretation

Optional robustness check:

- create a secondary country-expanded table where co-respondent cases are duplicated across countries
- use that table only for sensitivity analysis, not as the main modeling dataset

## Decision 3: Main Goal of the Next Data Round

The next data round is not only for adding more total rows.

Its main purpose is to improve:

- `Article 6` coverage
- `non-violation` coverage
- country diversity
- later bias-analysis feasibility

## Linked References

- experiment summary: [EXPERIMENT_CONCLUSIONS.md](/Users/jiejia/Programs/echr/EXPERIMENT_CONCLUSIONS.md)
- model diagnostics: [MODEL_DIAGNOSTICS.md](/Users/jiejia/Programs/echr/MODEL_DIAGNOSTICS.md)
- HUDOC expansion analysis: [HUDOC_DATA_EXPANSION.md](/Users/jiejia/Programs/echr/HUDOC_DATA_EXPANSION.md)
- HUDOC counts table: [hudoc_country_counts.csv](/Users/jiejia/Programs/echr/results/hudoc_country_counts.csv)
