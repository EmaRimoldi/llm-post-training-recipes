# Landscape And Positioning

This note summarizes where `STEMTune` sits relative to adjacent open-source projects and a few related research directions.

## Adjacent GitHub Projects

These repositories are close in spirit, but they are not identical to `STEMTune`:

- [huggingface/alignment-handbook](https://github.com/huggingface/alignment-handbook):
  recipes for aligning language models with human and AI preferences.
- [meta-pytorch/torchtune](https://github.com/meta-pytorch/torchtune):
  a PyTorch-native post-training library.
- [Lightning-AI/litgpt](https://github.com/Lightning-AI/litgpt):
  pretraining, finetuning, evaluation, and deployment workflows with recipe configs.
- [OpenRLHF/OpenRLHF](https://github.com/OpenRLHF/OpenRLHF):
  scalable RLHF and agentic RL infrastructure.
- [axolotl-ai-cloud/axolotl](https://github.com/axolotl-ai-cloud/axolotl):
  a broad training stack for open-source LLM finetuning.

## Related Research Directions

These papers are especially relevant because they cover routing, cost-aware model choice, or alignment simplification:

- [RouteLLM](https://arxiv.org/abs/2406.18665):
  routing between strong and weak models to trade off cost and quality.
- [FrugalGPT](https://arxiv.org/abs/2305.05176):
  cost-aware LLM cascades and approximation strategies.
- [Direct Preference Optimization](https://arxiv.org/abs/2305.18290):
  a simpler alignment method than full RLHF.

## Inference From This Landscape

This is an inference from the sources above, not a direct claim made by any one project:

- recipe libraries are strong at training execution;
- routing papers are strong at cost-aware model choice;
- alignment papers are strong at optimization methods;
- very few systems combine project scaffolding, model selection, artifact contracts, evaluation gates, and routing into one lightweight open-source control layer.

## Stronger Direction For STEMTune

`STEMTune` becomes more distinctive if it focuses on:

1. `decision-first` workflows:
   choose task, budget, and deployment target before touching a training script.
2. `config-first` ownership:
   datasets, KB corpora, training settings, evaluation gates, and publishing targets should be generated and edited as manifests.
3. `promotion-aware` model development:
   do not move from SFT to DPO, from training to quantization, or from local artifacts to Hub publication without explicit quality gates.
4. `small-team usability`:
   the framework should stay light enough for single-GPU and prosumer setups.

## What STEMTune Should Avoid

- trying to become a full deep-learning runtime like `torchtune`;
- trying to become a giant general-purpose finetuning platform like `axolotl`;
- trying to replicate full RLHF infrastructure like `OpenRLHF`;
- reducing itself to a static set of scripts without a control plane.
