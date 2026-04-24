# Milestone 2: Post-Training And Quantization

This milestone moves from data preparation into model adaptation for STEM question answering.

## Scope

The project explores several post-training directions:

- supervised fine-tuning;
- MCQA-oriented training data preparation;
- model quantization;
- Hugging Face model packaging through configuration files.

## Main Components

- `sft/`: supervised fine-tuning scripts and run helpers.
- `quantization/`: quantization experiments, including QLoRA-style workflows.
- `model_configs/`: model metadata used for evaluation and submission.
- `data/`: evaluation-support data and intermediate datasets.
- `pdf/`: report artifacts.

## Technical Signals

This milestone is the first one that clearly demonstrates hands-on LLM engineering work:

- adapting a base model to downstream QA tasks;
- managing training/evaluation data formats;
- trading model quality against deployment cost through quantization.

## Notes

- Some folders still reflect the original course submission format.
- The public-portfolio value is concentrated in the training and quantization scripts rather than in the grading scaffolding.
