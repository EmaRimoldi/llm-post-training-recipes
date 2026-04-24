# Repository Audit And Unification Plan

## Current State

The top-level `MNLP` folder consolidates three former GitHub Classroom repositories:

- `project-m1-2025-veme`
- `project-m2-2025-veme`
- `project-m3-2025-veme`

The original nested repository metadata has been preserved locally during unification, but the public-facing structure is now intended to be a single repository.

## Who Currently Holds The Repositories

The existing remotes point to the course organization, not to a personal GitHub namespace:

- `origin`: `https://github.com/CS-552/project-m1-2025-veme.git`
- `origin`: `https://github.com/CS-552/project-m2-2025-veme.git`
- `origin`: `https://github.com/CS-552/project-m3-2025-veme.git`

There are also `upstream` remotes pointing to GitHub Classroom assignment repositories in the same `CS-552` organization.

In practice, that means:

- the public ownership currently lives under the course organization;
- this local folder is not yet a single portfolio repository under your own GitHub account;
- commit history shows multiple contributors and bot accounts, so authorship should be presented honestly as coursework with collaborative components.

Visible contributors in the commit history include `EmaRimoldi`, `MoroMattia02`, `VinceEPFL`, `Emanuele Nevali`, several `ch-epfl-*` machine/user identities, and `github-classroom[bot]`.

## What The Repository Contains

### Milestone 1

Main theme: data collection and submission validation.

Relevant material:

- preference-data JSON files;
- annotation support notebooks;
- validation scripts;
- literature-review submission structure.

Portfolio value:

- demonstrates data curation and structured annotation work;
- less useful as a standalone engineering showcase than later milestones.

### Milestone 2

Main theme: initial model post-training.

Relevant material:

- supervised fine-tuning scripts;
- quantization experiments;
- Hugging Face configuration files;
- evaluation-support data.

Portfolio value:

- shows model adaptation and practical training scripts;
- useful as the first clear “LLM engineering” step.

### Milestone 3

Main theme: more advanced post-training and retrieval work.

Relevant material:

- DPO training;
- MCQA data pipelines;
- RAFT dataset generation;
- RAG-related data preparation and knowledge-base construction;
- quantization follow-up work.

Portfolio value:

- strongest technical section for recruiters;
- best candidate for README screenshots, diagrams, and highlighted code references.

## Main Issues Found

### 1. Public-repo safety issues

Several Python scripts contained:

- hardcoded Hugging Face access tokens;
- hardcoded `gpt_wrapper` API keys.

Those have now been replaced in the main Python scripts with environment-variable based configuration.

### 2. Portability issues

Some quantization scripts relied on absolute local paths such as `/home/mnlp/...` or `/users/nevali/...`.

Those were replaced in the main scripts with file-relative paths so the code is less tied to one machine.

### 3. Documentation issues

The original milestone READMEs were built for grading, not for readers.

That caused three problems:

- they described submission instructions instead of the technical work;
- they hid the interesting model and data pipelines;
- milestone 3 effectively had no useful README at all.

### 4. Repository-weight issues

Approximate current sizes:

- `project-m1-2025-veme`: `61M`
- `project-m2-2025-veme`: `114M`
- `project-m3-2025-veme`: `1.0G`

The main problem is `project-m3-2025-veme/RAG`, which is roughly `880M`.

That is too heavy and too artifact-driven for a clean public portfolio repository.

## Recommended Unification Strategy

### Option A: Keep milestone directories, unify only the root

Recommended for speed and clarity.

Structure:

- root `README.md`
- root `docs/`
- root `.gitignore`
- keep `project-m1-2025-veme/`
- keep `project-m2-2025-veme/`
- keep `project-m3-2025-veme/`

Why this is the best default:

- preserves the chronological story of the course;
- minimizes refactoring risk;
- gives recruiters a simple progression from data to training to RAG/DPO.

### Option B: Rename folders into recruiter-facing names

Good if you want a more polished presentation later.

Possible mapping:

- `project-m1-2025-veme` -> `01-preference-data-and-literature-review`
- `project-m2-2025-veme` -> `02-post-training-and-quantization`
- `project-m3-2025-veme` -> `03-dpo-and-rag-experiments`

This is more polished, but should happen only after you are sure you no longer need the original classroom folder names.

## How To Turn This Into One GitHub Repository

When you are ready to publish:

1. Create a new empty GitHub repository under your personal account.
2. In this local `MNLP` folder, remove the embedded `.git` directories from the three milestone folders.
3. Initialize Git at the root.
4. Review `git status` carefully before the first commit.
5. Commit only the curated public version, not the full course dump.

Suggested local commands:

```bash
find . -mindepth 2 -name .git -type d -prune -print
find . -mindepth 2 -name .git -type d -prune -exec rm -rf {} +
git init
git add .
git status
```

Important:

- removing embedded `.git` directories is destructive for the local nested repo history;
- do it only after you are sure you no longer need those folders as standalone repositories;
- if you want to preserve them, clone this folder elsewhere first and publish the clone.

## What To Keep Public

Keep:

- the cleaned training scripts;
- model config files;
- curated dataset-generation scripts;
- short READMEs describing method and outcome;
- one or two representative result artifacts if they are lightweight and interpretable.

Exclude or heavily curate:

- raw heavy RAG corpora;
- intermediate Arrow files;
- cached model weights;
- noisy notebooks with local paths or credentials;
- assignment templates copied directly from the course.

## Final Recommendation

Yes, this can become a strong public repository.

The best presentation is not “all coursework dumped in one place”, but:

- a clean umbrella README;
- honest attribution to the course context;
- emphasis on milestone 2 and especially milestone 3;
- careful exclusion of bulky and low-signal artifacts.

If you continue from this version, the next high-value steps are:

1. remove embedded `.git` directories in a publishable copy;
2. add one compact architecture diagram for the RAG/RAFT workflow;
3. optionally add a short “results” section with key numbers or qualitative outcomes.
