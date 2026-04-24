# Repository Notes

This repository groups the tracked code into functional areas:

- `datasets/`
- `training/`
- `retrieval/`
- `configs/`
- `reports/`
- `stemtune/`

## Public Repository Hygiene

The public version intentionally excludes:

- large raw corpora;
- notebooks and cache-heavy artifacts;
- local checkpoints and model binaries;
- nested git metadata from earlier local layouts.

## Local Safety Folders

Two ignored folders are kept locally for safety:

- `.nested-git-backups/` preserves older nested git metadata;
- `.legacy-local/` preserves non-published local trees and artifacts.

They are intentionally excluded from version control.
