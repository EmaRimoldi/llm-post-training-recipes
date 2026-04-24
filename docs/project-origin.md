# Project Origin

This repository started as three separate GitHub Classroom repositories created for the EPFL `CS-552` Modern NLP course.

The public version was later consolidated into one functional codebase to make the work easier to navigate and more useful to practitioners.

## Why It Was Reorganized

The original milestone-based structure had two drawbacks:

- it emphasized course logistics over engineering workflows;
- it scattered related training scripts across separate repositories.

The current layout fixes that by grouping the code into:

- `datasets/`
- `training/`
- `retrieval/`
- `configs/`
- `reports/`

## What Changed During Consolidation

- hardcoded Hugging Face tokens and API keys were removed from the main Python scripts;
- machine-specific `os.chdir(...)` assumptions were removed from the main quantization scripts;
- duplicated RAG training files were reduced to one canonical tracked location;
- large artifacts, notebooks, caches, and raw corpora were excluded from the public repository.

## Local Backups

Two ignored folders are kept locally for safety:

- `.nested-git-backups/` preserves the original nested git metadata;
- `.legacy-local/` preserves the old directory trees and non-published local artifacts.

They are intentionally excluded from version control.
