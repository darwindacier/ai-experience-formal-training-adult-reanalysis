# Replication package

**Exploring the Gap Between AI Experience and Formal Training Among
Pre-service Mathematics Teachers**

Repository: https://github.com/darwindacier/ai-experience-formal-training-adult-reanalysis

Peña-González, D., & Torres-Peña, R. C. — Discover Education

## Contents

```
data/    README.md                controlled-access conditions for row-level data
code/    reanalysis.py            primary pipeline — reproduces every reported value
         reviewer2_addenda.py     CMV, oblique EFA, cluster stability
         make_figures.py          all figures
         verify_manuscript.py     cross-checks the manuscript against the results
         instrument_translation.py  bilingual item mapping
output/  every result table (CSV) plus results.json
```

## Reproducing the analysis

```
pip install -r requirements.txt
python code/reanalysis.py --data /path/to/controlled_adult_dataset.csv
python code/reviewer2_addenda.py --data /path/to/controlled_adult_dataset.csv
python code/make_figures.py
```

The row-level dataset is not included in the public package. Access is controlled
subject to institutional ethics and privacy approval. Random seed is fixed at 42
throughout.

## Key figures

| Statistic | Value |
|---|---|
| N | 84 confirmed adults |
| Predictor-item ratio | 1.53:1 |
| Cronbach's alpha (56 Likert items) | 0.944 |
| Model R² (adjusted) | 0.587 (0.543) |
| Prior AI use / formal AI training | 84.5% / 22.6% |

## Licence

Code and aggregate outputs may be released for verification with attribution.
No row-level human-participant data are licensed for unrestricted public reuse.
