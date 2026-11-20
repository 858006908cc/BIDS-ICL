# Semantic Homogeneity As Demonstration (BIDS-ICL)

Research code accompanying [Semantic Homogeneity As Demonstration: Batch-Structured Semi-Supervised In-Context Learning for Natural Language Understanding](https://openreview.net/forum?id=y5bJXgBx6a), by Cheng Chen, Yuan Gang Pan, and Ivor W. Tsang (CPAL 2026).

The paper calls its method **Batch-Structured Implicit Demonstration-Free Semi-supervised ICL (BIDS-ICL)**. It uses semantically homogeneous batches and distinguishes **No-Hint** (unlabeled BS-ICL), **With-Hint** (pseudo-labeled BS-ICL), and the **Random Batch** baseline.

This package contains the attention mechanism experiments and a notebook for summarizing recorded aggregate results. Reproducing the paper’s end-to-end accuracy results requires the original dataset splits, student prediction pipeline, and LLM annotation runs. This package starts from supplied examples and predictions.

## Poster

[View the poster (PDF)](poster/bids-icl-poster.pdf) · [Editable PowerPoint](poster/bids-icl-poster.pptx)

## Theory: Plackett–Luce and noisy Plackett–Luce

**What is a utility score?** The utility $\lambda_r(x;C)$ is a latent score that represents how strongly the evidence in input $x$ and context $C$ favors candidate label $r$. For example, “Can we do Friday instead?” may fit both *schedule* and *reschedule*. In a batch about moving existing appointments, the model may assign *reschedule* a higher utility. The utility is an unnormalized score. The PL equation converts it to a choice probability.

Only differences between scores matter. If *reschedule* has utility $2$ and *schedule* has utility $1$, the PL odds favor *reschedule* by $e^{2-1}\approx 2.7$ to 1. Utilities $5$ and $4$ give the same odds. These numbers illustrate the model. BIDS-ICL uses the utilities to explain how batch context can change label preferences, while the inference procedure reads the LLM's predictions from the batch prompt.

**Plackett–Luce (PL)** turns utility scores into probabilities over the labels still available at ranking stage $i$. Let $\rho^{(i)},\ldots,\rho^{(L)}$ be those candidates. The probability of latent choice $Y_i=\rho^{(t)}$ is

```math
\Delta_{i,t}(\lambda)= P(Y_i=\rho^{(t)}\mid\lambda)= \frac{e^{\lambda_{\rho^{(t)}}}}{\sum_{v=i}^{L} e^{\lambda_{\rho^{(v)}}}}
```

In BIDS-ICL, a coherent batch can widen the utility gap between a plausible label and its competitors. PL remains probabilistic, and the leading utility can favor the wrong label. The candidate label set is known at inference time. The true label for each test input is held out for evaluation.

**Noisy Plackett–Luce** adds a worker response model. On the slide, $Y_i$ is the latent utility-based choice, $\tilde Y_i$ is the worker's reported choice, and $\eta_w$ is the worker's uncertainty profile. For the report $\rho^{(i)}$, the normalized weight $\bar\eta_w^{(t-i+1;i)}$ gives the probability of that report when the latent choice is $\rho^{(t)}$. The report probability sums over every possible latent choice:

```math
P(\tilde Y_i=\rho^{(i)}\mid\lambda,\eta_w)= \sum_{t=i}^{L}\bar\eta_w^{(t-i+1;i)}\,\Delta_{i,t}(\lambda)
```

Each term multiplies the **PL probability of the latent choice** by the **probability that the worker gives the observed report given that choice**. The slide models a more proficient worker with a profile more concentrated on agreement.

**Worked example.** Suppose PL assigns *reschedule* an 80% chance and *schedule* a 20% chance. If the worker reports *reschedule* with probability 90% when the latent choice is *reschedule* and 10% when it is *schedule*, the overall report probability is $0.8(0.9)+0.2(0.1)=0.74$.

**Noiseless limit.** If the worker always reports the latent choice faithfully, only the $t=i$ term contributes to the probability of reporting $\rho^{(i)}$. The noisy equation reduces to the ordinary PL probability $\Delta_{i,i}(\lambda)$. Correctness still depends on whether that choice matches the held-out true label.

An optional encoder pseudo-label is a **hint supplied in the prompt**. It can influence the utilities through the context. The reported choice $\tilde Y_i$ is the **output** modeled by the noisy PL equation.

### How batch homogeneity and LLM competency enter the theory

In BIDS-ICL, the **worker** $w$ is the LLM making the annotation. **Batch homogeneity** describes the inputs it sees. These affect different parts of the model.

| Part of the paper | Where it enters | What it represents |
| --- | --- | --- |
| **Semantic homogeneity of the batch** | The context $C$ shapes $\lambda_r(x;C)$. In the Bayesian interpretation, within-cluster coherence is a proxy for concentration in the utility prior $P(\lambda\mid\eta_w)$. | Related inputs can support the same interpretation and widen the utility gap between candidate labels. Coherence is estimated by averaging embedding similarity to nearby inputs in the cluster. |
| **LLM competency** | The worker's uncertainty profile $\eta_w$ and the worker prior $P(\eta_w)$. | The theory represents a more proficient LLM with a profile concentrated on agreement between the latent choice and the reported choice. A specialized proxy LLM approximates this regime. |
| **Likelihood of the observed report** | $P(\tilde Y\mid\lambda,\eta_w)$, calculated by the noisy PL equation above. | Given the label utilities and worker profile, how likely is the LLM to report the choice we observed? The likelihood combines the two effects. |

The Bayesian view combines the three terms.

```math
P(\lambda,\eta_w\mid\tilde{Y})
\propto
P(\tilde{Y}\mid\lambda,\eta_w)
\,P(\lambda\mid\eta_w)
\,P(\eta_w)
```

From left to right on the right-hand side, the factors are the **likelihood of the report**, the **prior over utilities**, and the **worker prior**.

For example, a coherent batch about moving existing appointments can make *reschedule* stand out for “Can we do Friday instead?” A proficient LLM may pick up that context in the **No-Hint** setting. In the **With-Hint** setting, an encoder's guess of *reschedule* can give a less proficient LLM additional evidence. A mistaken *schedule* hint can instead pull the prediction toward the wrong label, especially when the surrounding inputs supply weak evidence. The posterior is an explanatory model for this interaction. At inference time, BIDS-ICL supplies the batch prompt and reads the LLM's predictions.


### How noisy Plackett–Luce informs the experimental design

- **Sentence encoder → batch homogeneity.** We vary the sentence encoder used to embed and cluster inputs. Different encoders can produce batches with different levels of semantic homogeneity. For a fixed LLM, comparing these batches probes how the context $C$ changes the evidence for candidate labels and, in the theory, their utility scores $\lambda$. We use cluster coherence to characterize the resulting batches.
- **LLM → worker and observed output.** The LLM is the worker $w$. It receives a prompt containing the clustered batch and produces a label for each input. The generated label is the observed report $\tilde{Y}$. For a fixed encoder and batch, comparing LLMs probes how models of different competency use the same context. The uncertainty profile $\eta_w$ represents these differences in the theory.
- **No-Hint versus With-Hint → additional context.** Using the same batch and LLM, we compare prompts with inputs alone against prompts that also include encoder-generated pseudo-label hints. This tests whether hints help or hurt at different levels of homogeneity and model competency.

The noisy PL likelihood $P(\tilde{Y}\mid\lambda,\eta_w)$ brings these roles together. The encoder shapes the batch context that informs the utilities, and the LLM produces the report from that context. The experiments compare observed prediction accuracy. The utilities and worker uncertainty serve as theoretical explanations.

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
