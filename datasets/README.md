# Datasets

This folder keeps only lightweight tracked assets and data-building scripts.

## Included

- `preference/`: example preference data exports;
- `calibration/`: small calibration data used by compression workflows;
- `metadata/`: lightweight references to dataset publication targets;
- `builders/`: scripts that generate or preprocess training data.

## Excluded

Large datasets are intentionally not versioned in git.

If you want to run the heavier training recipes locally, place external corpora under:

```text
datasets/external/
├── gpqa/
├── mathqa/
│   └── train.json
└── tuandunghcmut_coding-mcq-reasoning/
```

You can extend this structure with additional corpora as needed.
