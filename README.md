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




