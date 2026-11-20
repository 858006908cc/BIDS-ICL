# Semantic Homogeneity As Demonstration (BIDS-ICL)

Research code accompanying [Semantic Homogeneity As Demonstration: Batch-Structured Semi-Supervised In-Context Learning for Natural Language Understanding](https://openreview.net/forum?id=y5bJXgBx6a), by Cheng Chen, Yuan Gang Pan, and Ivor W. Tsang (CPAL 2026).

The paper calls its method **Batch-Structured Implicit Demonstration-Free Semi-supervised ICL (BIDS-ICL)**. It uses semantically homogeneous batches and distinguishes **No-Hint** (unlabeled BS-ICL), **With-Hint** (pseudo-labeled BS-ICL), and the **Random Batch** baseline.

This package contains the attention mechanism experiments and a notebook for summarizing recorded aggregate results. Reproducing the paper’s end-to-end accuracy results requires the original dataset splits, student prediction pipeline, and LLM annotation runs. This package starts from supplied examples and predictions.

## Poster

[View the poster (PDF)](poster/bids-icl-poster.pdf) · [Editable PowerPoint](poster/bids-icl-poster.pptx)

## Theory: Plackett–Luce and noisy Plackett–Luce

**Plackett–Luce (PL)** turns utility scores into probabilities over the labels still available at ranking stage $i$. Let $\rho^{(i)},\ldots,\rho^{(L)}$ be those candidates. The probability of latent choice $Y_i=\rho^{(t)}$ is

$
\Delta_{i,t}(\lambda)
= P(Y_i=\rho^{(t)}\mid\lambda)
= \frac{e^{\lambda_{\rho^{(t)}}}}{\sum_{v=i}^{L} e^{\lambda_{\rho^{(v)}}}}.
$

In BIDS-ICL, a coherent batch can widen the utility gap between a plausible label and its competitors. PL remains probabilistic, and the leading utility can favor the wrong label. The candidate label set is known at inference time. The true label for each test input is held out for evaluation.

**Noisy Plackett–Luce** adds a worker response model. On the slide, $Y_i$ is the latent utility-based choice, $\tilde Y_i$ is the worker's reported choice, and $\eta_w$ is the worker's uncertainty profile. For the report $\rho^{(i)}$, the normalized weight $\bar\eta_w^{(t-i+1;i)}$ gives the probability of that report when the latent choice is $\rho^{(t)}$. The report probability sums over every possible latent choice:

$
P(\tilde Y_i=\rho^{(i)}\mid\lambda,\eta_w)
= \sum_{t=i}^{L}
\bar\eta_w^{(t-i+1;i)}\,\Delta_{i,t}(\lambda).
$

Each term multiplies the **PL probability of the latent choice** by the **probability that the worker gives the observed report given that choice**. The slide models a more proficient worker with a profile more concentrated on agreement.

**Worked example.** Suppose PL assigns *reschedule* an 80% chance and *schedule* a 20% chance. If the worker reports *reschedule* with probability 90% when the latent choice is *reschedule* and 10% when it is *schedule*, the overall report probability is $0.8(0.9)+0.2(0.1)=0.74$.

**Noiseless limit.** If the worker always reports the latent choice faithfully, only the $t=i$ term contributes to the probability of reporting $\rho^{(i)}$. The noisy equation reduces to the ordinary PL probability $\Delta_{i,i}(\lambda)$. Correctness still depends on whether that choice matches the held-out true label.

An optional encoder pseudo-label is a **hint supplied in the prompt**. It can influence the utilities through the context. The reported choice $\tilde Y_i$ is the **output** modeled by the noisy PL equation.

## Setup

Use Python 3.10 or later. The default model is Meta Llama 3 8B Instruct. For the full-size checkpoint, obtain access on Hugging Face, authenticate with `hf auth login` or set `HF_TOKEN` in your environment, and use a supported CUDA GPU with sufficient memory. Use `--device cpu` with a small local Llama checkpoint for development.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Inputs

Prepare a UTF-8 JSONL file with an `input` string and a `label` string or integer per row. Create an NPZ with a one-dimensional `pseudo_labels` array containing one encoder-predicted class per JSONL row, in the same order. The legacy `assigned_labels` key is also accepted. Use a string or numeric array. Legacy object arrays require `--allow_pickle` and should only be loaded from a trusted source.

Generate example inputs locally with `python examples/make_bids_icl_demo_inputs.py data/synthetic`. They are intended for smoke tests.

## Attention experiments

```bash
python experiments/bids_icl_homogeneity_diagnostics.py \
  --dataset-name synthetic \
  --examples data/synthetic/examples.jsonl \
  --pseudo-labels data/synthetic/pseudo_labels.npz \
  --batch_size 3 --min_label_frequency 3 --num_batches 4 \
  --layer_idx 0 --head_idx 0 --save_dir results
```

The extended script compares attention entropy, alignment, and residual-stream projections. It writes a JSON report per layer and head. Pass comma-separated zero-based indices to `--test_layers` to generate a summary plot. By default, homogeneous batches are grouped by encoder-predicted pseudo-labels. The original extended script grouped by gold labels. Use `--group-by gold-label` to reproduce that oracle diagnostic. Both scripts use input-only prompts, as in the No-Hint setting. Pseudo-labels can determine the grouping without appearing in the prompt. The notebook summarizes recorded No-Hint and With-Hint results. The simpler `experiments/bids_icl_attention_comparison.py` compares predicted-label batches with randomly drawn batches and prints its results.

Repeatedly drawn batches can overlap. Treat the reported Welch tests as exploratory diagnostics when interpreting these comparisons.

## Results notebook

Run `jupyter lab notebooks/bids_icl_backbone_ablation.ipynb` to regenerate tables and plots from the aggregate values recorded in the notebook.

## Citation

```bibtex
@inproceedings{chen2026semantic,
  title     = {Semantic Homogeneity As Demonstration: Batch-Structured Semi-Supervised In-Context Learning for Natural Language Understanding},
  author    = {Chen, Cheng and Pan, Yuan Gang and Tsang, Ivor W.},
  booktitle = {The Third Conference on Parsimony and Learning},
  year      = {2026},
  url       = {https://openreview.net/forum?id=y5bJXgBx6a}
}
```
