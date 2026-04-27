---
name: generative-models
description: The two generative paradigms (BiGG and CGT), their mechanisms, output formats, key files, and Conda environments.
---

# SynGraphBench — Generative Models

## The Two Paradigms

A central conceptual distinction in this project is **how** synthetic data is produced. BiGG and CGT operate at fundamentally different levels of abstraction, which determines what they output, how that output is evaluated, and what kind of utility/privacy trade-off they represent.

**BiGG output** is a whole DGL graph → evaluated with standard whole-graph GNNs (`--synthetic_type graph`).
**CGT output** is a `.pt` file of cluster centers and sequence indices → evaluated with computation-graph GNNs (`--synthetic_type comp-graph`).
The two paradigms are **not directly comparable** on a common evaluation framework; the benchmark script handles each path separately.

---

## BiGG — Whole-Graph Generation

BiGG generates a **complete, new graph** — both topology (edges) and node features — from scratch. The output is a full DGL graph that is a drop-in replacement for the original. The original graph is discarded entirely during evaluation; downstream models train and are tested purely on the generated graph.

The generative model captures the joint distribution of graph structure and node attributes, and samples from it to produce a synthetic counterpart.

* **Key Mechanisms:** Decomposes graph generation into a sequence of binary tree decisions, processed efficiently via a custom C++ extension (`tree_clib`). Two modes: a conditional model (features + labels) and a structure-only baseline.
* **Data Augmentation (optional):** Two training-time augmentation strategies, both disabled by default:
  * *Gaussian noise* (`noise_std`): Adds random noise to hidden state during training to improve generalization. Default `0.0` (disabled).
  * *Scheduled self-sampling* (`ss_max_prob`, `ss_start_epoch`): Gradually replaces teacher-forced inputs with the model's own predictions during training, reducing exposure bias. `ss_max_prob` controls the max probability (default `0.0` = disabled); `ss_start_epoch` controls when ramp-up begins.
* **Label Masking (optional):** `--mask_test_labels` excludes test node labels (split 0) from the label loss during training, preventing data leakage in anomaly detection benchmarks. The model still sees test node features and structure (and uses test labels for teacher forcing / state updates) — only the label *loss* is masked. Appends `_masked` to the save name.
* **Binary Feature Head (optional):** `--binary_feat` auto-detects binary {0,1} feature columns and uses a separate BCE loss + Bernoulli sampling head for them, instead of the Gaussian head. Binary columns also skip normalization. Appends `_binfeat` to the save name.
* **CVAE Feature Latent (optional):** `--vae_feat` adds a per-node label-agnostic VAE latent `z` shared across the continuous and binary feature decoders. Encoder `q(z|h,x)` produces `μ, log σ²`; reparameterized `z` is concatenated into both feature-head inputs (label head stays on `h`). Training uses the posterior; generation samples `z ∼ N(0, I)`. Intent: recover joint cross-feature covariance that the deterministic per-head decoders cannot represent. Controlled by `-vae_dim` (latent size, default 16) and `-kl_weight` (KL coefficient, default 1.0; not subject to dynamic calibration). Appends `_vae{dim}_kl{weight}` to the save name.
* **MDN Feature Head (optional):** `--mdn_feat` replaces the continuous feature head with a **Mixture Density Network** — per continuous column the model emits K components `(π_k, μ_k, σ_k)` conditioned on `[h, label_embed, (z)]`. Training loss is the NLL of the true value under the mixture; generation samples a component index then a per-component sample. No bins, no AR chain. Handles embeddings, mass points, and near-binary columns uniformly (components collapse onto modes as needed). Under k-means privacy the centroids become a discrete mixture and MDN components naturally collapse onto them, preserving architectural parity across privacy levels. Binary columns keep the BCE/Bernoulli head. Composes with `--vae_feat` (z is concatenated into the head's conditioning input — keeps cross-feature correlation while components handle marginal multi-modality). Mutually exclusive with `--hetero_feat` and `--cat_feat`. Controlled by `-mdn_components` (default 8), `-mdn_logsigma_floor` (default -4.0), and `-mdn_base` (default `gaussian`). Save-name prefix is `_mdn{K}_lsf{floor}` for `gaussian` and `_lnmdn{K}_lsf{floor}` for `logit_normal`.
* **MDN base distribution (`-mdn_base`):** Selects the per-component base. `gaussian` (default) is unbounded — appropriate for raw / z-score / quantile-normalised targets. `logit_normal` places components on the logit of `u ∈ [0,1]`: training target is `t = logit(u)` and at sample time the output is `sigmoid(μ_k + σ_k·ε)`. Pairs with `--normalize cdf` for [0,1]-encoded targets — components can specialise on mass points (μ_k near `±logit(eps)`, σ_k tiny) without smearing. The Jacobian of `logit` is constant w.r.t. parameters and is dropped from the loss. Recommended composition for CDF-encoded targets with multi-modal features: `--mdn_feat --mdn_base logit_normal --vae_feat --normalize cdf`.
* **AR Categorical Feature Predictor (optional):** `--cat_feat` replaces the continuous feature head with an autoregressive categorical predictor implemented in `BiggWithARCatFeats`. Continuous columns are binned per-feature using fixed quantile edges (fit on the full training matrix via `fit_feature_bins` in `preprocessing.py`). At each node the label is predicted first, then features are generated autoregressively across feature positions, with a shared MLP conditioned on `[h, label_embed, feat_pos_embed, prev_bin_embed]`. Training batches all AR steps in parallel via teacher forcing on previous bins; generation loops sequentially. Loss is value-space Gaussian soft-label cross-entropy — smoothing kernel sits in feature-value units with a **per-feature σ** (0.5 × median positive spacing of that feature's bin centers). Per-feature σ adapts across regimes in a single mechanism: tight σ for dense continuous, wide σ for binary-like (spans both values), automatic scaling under k-means privacy (k centers, gaps set by k). This removes the need for a separate binary head — every column goes through cat_feat uniformly, which keeps the architecture identical across privacy levels. Binary columns keep the BCE/Bernoulli head. Mutually exclusive with `--hetero_feat` and `--vae_feat`. Controlled by `-n_bins` (default 32) and `-bin_sigma` (override auto). Persists `bin_edges`, `bin_centers`, `bin_sigma`, `n_bins` to `cat_bins.pt` alongside the graph artifact; appends `_cat{n_bins}_s{sigma}` to the save name.
* **Output format:** A full DGL graph stored as a file under `datasets/synthetic/bigg/<dataset>/<task>/`.
* **Important Files:**
  * `bigg/extension/pipeline.py`: Conditional model training (features + labels).
  * `bigg/extension/pipeline_structure_only.py`: Structure-only baseline.
  * `bigg/data_process/`: Dataset preparation scripts.
* **Environment:** `bigg` Conda environment (Python 3.9, PyTorch 2.4.1).
  * **Agent Alert:** Requires compiling the `tree_clib` C++ extension via `make` before running. This is handled in `scripts/env_setups/bigg_setup.sh`. Modern CUDA architectures are patched via `sed` in that script.

---

## CGT — Computation-Graph Generation

CGT operates at a different level. Rather than generating a new graph, it **generates synthetic node feature distributions** while leaving the original graph topology unchanged. The key idea is that GNNs aggregate information via local computation trees (computation graphs) rooted at each node. CGT learns to generate sequences of feature vectors that match the distribution of these computation trees.

Concretely, CGT uses **DP-k-means** to cluster the real node features into `k` cluster centers. These centers — not the raw features — are what is "synthesized" and shared. During evaluation, the original graph's training and validation node features are replaced by the nearest cluster center, and a GNN is trained on this feature-masked graph and tested on the unmasked original test nodes.

* **Key Mechanisms:** Operates on minibatches of computation graph sequences (not the full graph). DP-k-means provides differential privacy. A Transformer learns to generate realistic sequences of cluster assignments.
* **Output format:** A `.pt` file containing cluster centers, generated sequence indices, and train/val/test node ID mappings. Stored under `datasets/synthetic/cgt/<dataset>/<task>/`.
* **Important Files:**
  * `CGT/train.py`: Training script.
  * `CGT/test.py`: Generation and evaluation script.
  * `CGT/args.py`: Hyperparameter configurations.
* **Environment:** `CGT` Conda environment (Python 3.11). Setup via `scripts/env_setups/cgt_setup.sh`.
