# HUDOC Data Expansion Notes

## Scope

This note checks how much more data can be added from HUDOC under the same broad acquisition logic used in [download_data.py](/Users/jiejia/Programs/echr/scripts/download_data.py):

- `contentsitename = ECHR`
- `documentcollectionid2 = JUDGMENTS`
- `languageisocode = ENG`
- labels inferred from `violation` and `nonviolation`
- target articles currently focused on `3 / 5 / 6 / 8`

The counts below were verified against the official HUDOC query endpoint on `2026-03-22`.

## Main Finding

Yes, the dataset can still be expanded substantially.

There are two different expansion paths:

1. add more cases from the current countries
2. add more countries

The second path is more valuable for the next research stage, because the project is moving toward:

- `Article 6`
- country-level analysis
- time-slice analysis
- possible bias analysis

## Current Three-Country Scope vs Available HUDOC Volume

Under the official HUDOC counts for English judgments and the current target-article logic:

### All target articles (`3 / 5 / 6 / 8`)

| respondent | total | violation | non-violation | mixed |
|---|---:|---:|---:|---:|
| RUS | `1181` | `1125` | `16` | `40` |
| TUR | `490` | `426` | `29` | `35` |
| GBR | `87` | `48` | `29` | `10` |

### `Article 6` only

| respondent | total | violation | non-violation | mixed |
|---|---:|---:|---:|---:|
| RUS | `603` | `591` | `8` | `4` |
| TUR | `281` | `259` | `13` | `9` |
| GBR | `37` | `21` | `13` | `3` |

Interpretation:

- `RUS` and `TUR` still have raw headroom if the goal is only to add more cases.
- But both are heavily violation-dominant, especially on `Article 6`.
- `GBR` has much less volume, but comparatively better negative coverage.

So if the goal is balanced modeling and later bias analysis, simply adding more `RUS` and `TUR` is not enough.

## Best Additional Countries for Article 6

The most relevant expansion target is no longer raw volume alone. It is:

- enough `Article 6` cases
- enough `non-violation` cases
- preferably a different country profile from the current three-country set

### Strong first-priority additions

| respondent | article6 total | article6 violation | article6 non-violation | comment |
|---|---:|---:|---:|---|
| FRA | `161` | `123` | `32` | strongest balance among high-volume candidates |
| ROU | `159` | `126` | `23` | good volume and usable negative count |
| POL | `156` | `140` | `13` | strong volume, moderate negatives |
| DEU | `37` | `24` | `11` | smaller total, but relatively balanced |
| BEL | `50` | `37` | `8` | smaller but useful for balance |

These are the most defensible next countries if the project wants better country comparison rather than only more positive examples.

### Good secondary additions

| respondent | article6 total | article6 non-violation | comment |
|---|---:|---:|---|
| LTU | `25` | `12` | good balance, but small |
| ESP | `26` | `7` | moderate support |
| CZE | `28` | `6` | moderate support |
| AUT | `57` | `6` | useful mid-sized add-on |
| PRT | `63` | `6` | useful mid-sized add-on |

These are better for improving balance than for adding large volume.

### High-volume but imbalance-heavy additions

| respondent | article6 total | article6 non-violation | comment |
|---|---:|---:|---|
| ITA | `834` | `13` | very large but strongly violation-heavy |
| UKR | `438` | `7` | large but highly imbalanced |
| GRC | `104` | `4` | limited negative support |
| HUN | `90` | `1` | almost all positive |

These countries are useful if the goal is to increase raw `Article 6` size, but they do not solve the balance problem.

## Recommended Expansion Strategy

### If the goal is better overall model training

Add:

- `FRA`
- `ROU`
- `POL`
- `DEU`
- `BEL`

Reason:

- they add both volume and more `non-violation` support than the current `RUS / TUR` heavy profile

### If the goal is specifically Article 6 bias analysis

Use a two-layer strategy:

1. keep `RUS`, `TUR`, `GBR`
2. add `FRA`, `ROU`, `POL`, `DEU`

This gives:

- more countries
- more institutional variety
- more negatives
- better support for `country x time` comparisons

### If the goal is only to maximize sample size fast

Add:

- `ITA`
- `UKR`

But this is not the best option for balanced classification or bias analysis, because both are strongly skewed toward violation.

## Important Implementation Note

Current raw metadata already contains a few multi-respondent rows such as:

- `DNK;TUR`
- `MDA;RUS`

That means before broadening the country list, the download and preprocessing logic should explicitly define how to treat co-respondent cases:

- keep them as-is
- assign them to multiple country buckets
- or exclude them from country-specific bias analysis

This rule should be fixed before the next full download, otherwise country-level results will be inconsistent.

## Co-Respondent Decision

The project decision is:

1. keep co-respondent cases unchanged in the raw and model-training tables
2. add explicit indicators such as `is_multi_respondent` and a parsed respondent list
3. exclude co-respondent cases from the main country-level bias analysis
4. if needed, run a separate sensitivity check with duplicated country assignment

Why this decision was chosen:

- keeping the raw case unchanged preserves the original legal record
- not duplicating the case in the training table avoids artificial sample inflation
- excluding co-respondent cases from the main country analysis gives the cleanest interpretation for `country x time` bias analysis
- duplicated assignment can still be used later as a robustness check, but it should not be the default analytical table

This means “split to multiple countries” is **not** the default rule for this project.

## Output Files

The country-level HUDOC summary generated for this check is saved in:

- [hudoc_country_counts.csv](/Users/jiejia/Programs/echr/results/hudoc_country_counts.csv)
- [hudoc_summary.json](/Users/jiejia/Programs/echr/results/hudoc_summary.json)

## Practical Conclusion

The next download should not just “add more cases”.

It should add countries in a targeted way:

- first for balance: `FRA`, `ROU`, `POL`, `DEU`, `BEL`
- optionally for raw volume: `ITA`, `UKR`

That is the most defensible path if the project is moving toward `Article 6` country-level bias analysis.

## Next Download Decision

The next expansion wave is fixed as:

- `FRA`
- `ROU`
- `POL`
- `DEU`
- `BEL`

These five countries are selected because they improve the dataset in the most useful way for the next stage:

- more `Article 6` cases
- more `non-violation` support
- broader country coverage
- better support for later `country / time` bias analysis

They should be added before any “raw-volume-only” countries such as `ITA` or `UKR`.
