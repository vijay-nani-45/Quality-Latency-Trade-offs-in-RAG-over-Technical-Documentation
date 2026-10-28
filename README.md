# Quality-Latency Trade-offs in RAG over Technical Documentation


The study compares retrieval depth (k) and generator size (Qwen2.5 1.5B / 3B / 7B) on RAGBench TechQA and DelucionQA, then tests whether a router that costs under a millisecond can cut average latency without losing answer quality.

## 1. Setup

```bash
# 1. environment (use your cluster's conda/module system; Python 3.10+)
conda create -n ragql python=3.11 -y && conda activate ragql

# 2. a CUDA build of torch that matches the node driver, for example:
pip install torch --index-url https://download.pytorch.org/whl/cu121

# 3. the rest
pip install -r requirements.txt

# 4. keep every download inside the project (shared home directories fill up fast)
export HF_HOME=$PWD/hf_cache        # config.py also does this if HF_HOME is unset
```

If `HF_HOME` is already exported in your shell profile and points at your home directory, override it with the line above. `config.py` only sets a default; it does not overwrite an existing value.

Download everything once, from a node that has internet access:

```bash
python download_assets.py                # 2 RAGBench subsets, 3 Qwen2.5 models (~25 GB), bge-small
python download_assets.py --verify-only  # offline check that the cache is complete
```



## 2. Commands

```bash
# Plumbing check on a tiny slice (6 validation + 10 test queries per dataset, 64 new tokens)
python run_experiments.py --smoke-test --gpu 0

# The whole study, one command (prep -> retriever -> fixed -> adaptive -> report)
python run_experiments.py --stage all --gpu 0

# Or stage by stage
python run_experiments.py --stage prep      --gpu 0   # pool corpora, build BM25 + FAISS indices
python run_experiments.py --stage retriever --gpu 0   # BM25 vs Dense vs Hybrid-RRF
python run_experiments.py --stage fixed     --gpu 0   # 12 fixed configs, validation + test
python run_experiments.py --stage adaptive  --gpu 0   # validation-Pareto calibration, then live on test
python run_experiments.py --stage report              # statistics, tables, figures (CPU only)
```



## 3. What gets written

```
results/                              (results_smoke/ for --smoke-test)
  hardware_env.json                   GPU name, VRAM, torch/CUDA versions, bf16 support
  prep/<dataset>/                     pooled corpus, queries per split, summary.json (dedup counts)
  prep/router_calibration_<ds>.json   frozen router: chosen Pareto tiers, weights, thresholds, validation diagnostics
  indices/<dataset>/                  BM25 pickle, dense embeddings
  raw_logs/
    retriever_<ds>_<retriever>_test.jsonl
    fixed_<ds>_<split>_k<k>_<model>.jsonl      one row per query
    adaptive_<ds>_test.jsonl
  tables/
    hypothesis_verdicts.md            H1 and H2 verdicts with every test statistic
    deployment_guidelines.md          guidelines filled in with this run's numbers
    retriever_comparison.{csv,md}
    latency_table.{csv,md}            mean / P50 / P95 / IQR of each latency component
    fixed_config_summary.csv, fixed_all_queries.csv, adaptive_all_queries.csv
    h1_friedman_<ds>.csv, h1_steps_<ds>.csv, h2_summary.csv
    router_calibration_<ds>.json      copy of the frozen router calibration (the original is in results/prep/)
  figures/                            fig1 ... fig5, each as 300-DPI PNG and PDF
```

Each row in `fixed_*.jsonl` and `adaptive_*.jsonl` has the retrieved document ids, the answer, `prompt_tokens`, `output_tokens`, `retrieval_latency_ms`, `router_latency_ms`, `prompt_build_ms`, `ttft_ms`, `decode_latency_ms`, `ms_per_token`, `generation_latency_ms`, `total_latency_ms`, `recall_at_k`, `mrr_at_k`, all quality sub-metrics and `composite_quality`.

# Live Demo

An interactive demo of the project **"Quality–Latency Trade-offs in Retrieval-Augmented Question Answering over Technical Documentation"**. It runs on an ordinary Windows or Mac laptop (8–16 GB RAM, no graphics card needed).

You can ask a question, watch the system retrieve passages, route the question to a cheap or expensive configuration, generate an answer, and see latency and quality numbers for that exact run. Next to it sit the stored results of the full study on the complete test split (314 TechQA + 184 DelucionQA questions).

## Start in three commands

```bash
pip install -r requirements.txt
python download_laptop_models.py
streamlit run app.py
```

Use a virtual environment if you like:

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # Mac / Linux
```



### A 6-minute demo script

1. **Tab ①, Fixed Configuration.** Pick a TechQA question, k = 3, Small. Run. Point at the answer, the retrieved passages (✅ marks passages annotated as relevant), the five latency cards and the five quality cards.
2. **Tab ①, Adaptive.** Same question. Show the router card: top-1 score, margin, complexity, difficulty, the tier it picked, and the overhead (a fraction of a millisecond).
3. **Tab ①, Side-by-Side Live Compare.** Show the milliseconds saved and the quality change. Say this is one question, an anecdote.
4. **Tab ②.** Run the 5-question mini-benchmark while you explain the design. Show the summary table and the speed-up chart.
5. **Tab ③.** Switch dataset in the sidebar. Walk through the H1 card (k matters, generator size does not consistently help) and the H2 card (latency down, quality non-inferior). Mention the honest note about simple fixed configurations.
6. **Tab ④.** Drag the budget slider and read out the recommendation.

## The metrics in one place

* **Total latency** = retrieval + router + prompt building + **TTFT** (time to first token, the prompt-reading step) + **decode** (writing the rest of the answer). Model loading and warm-up are never counted.
* **Composite quality** = 0.35 × evidence token F1 + 0.25 × ROUGE-L + 0.25 × semantic similarity + 0.15 × context faithfulness. For your own custom questions there is no reference answer, so only context faithfulness is computed.
* **Recall@k / MRR@k** measure whether the annotated relevant passages were retrieved, and how high.
* **Non-inferiority (δ = 0.02)**: the adaptive pipeline counts as not worse if its average quality loss is below 0.02 with statistical confidence.







