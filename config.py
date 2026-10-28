
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
CACHE_DIR = PROJECT_ROOT / "hf_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(CACHE_DIR))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

PREP_DIR = PROJECT_ROOT / "results" / "prep"
INDEX_DIR = PROJECT_ROOT / "results" / "indices"


@dataclass(frozen=True)
class RunPaths:
    """Output locations of one run. Smoke runs write to results_smoke/ so they
    never contaminate the real results/ folder."""

    root: Path

    @property
    def raw_logs(self) -> Path:
        return self.root / "raw_logs"

    @property
    def tables(self) -> Path:
        return self.root / "tables"

    @property
    def figures(self) -> Path:
        return self.root / "figures"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    @property
    def hardware_json(self) -> Path:
        return self.root / "hardware_env.json"

    def ensure(self) -> "RunPaths":
        for p in (self.raw_logs, self.tables, self.figures, self.logs, PREP_DIR, INDEX_DIR):
            p.mkdir(parents=True, exist_ok=True)
        return self


def make_paths(smoke: bool = False) -> RunPaths:
    return RunPaths(PROJECT_ROOT / ("results_smoke" if smoke else "results")).ensure()


# Datasets (Hugging Face RAGBench)
HF_DATASET_IDS = ("rungalileo/ragbench", "galileo-ai/ragbench")  # tried in order
SPLITS = ("train", "validation", "test")
DATASETS = {
    "techqa": {"pretty": "TechQA", "expected_docs": 4066, "expected_test": 314},
    "delucionqa": {"pretty": "DelucionQA", "expected_docs": 930, "expected_test": 184},
}
DATASET_ORDER = ("techqa", "delucionqa")

# Models 
def _env_model(key: str, default: str) -> str:
    return os.environ.get("RAG_MODEL_" + key.replace(".", "_"), default)


MODEL_ORDER = ("1.5B", "3B", "7B")
MODEL_IDS = {
    "1.5B": _env_model("1.5B", "Qwen/Qwen2.5-1.5B-Instruct"),
    "3B": _env_model("3B", "Qwen/Qwen2.5-3B-Instruct"),
    "7B": _env_model("7B", "Qwen/Qwen2.5-7B-Instruct"),
}
EMBED_MODEL = os.environ.get("RAG_EMBED_MODEL", "BAAI/bge-small-en-v1.5")

# Experimental grid
RETRIEVER_K_VALUES = (1, 3, 5, 8, 10)  # retriever comparison table
FIXED_K_VALUES = (1, 3, 5, 8)          # RQ1 grid: 4 k x 3 models = 12 fixed configs
MAX_RETRIEVE = 10
ROUTER_TOP_N = 8                       # dense scores fetched once for routing


def config_id(k: int, model: str) -> str:
    return f"k{k}_{model}"


CONFIG_ORDER = tuple(config_id(k, m) for m in MODEL_ORDER for k in FIXED_K_VALUES)

# Adaptive tiers (Pareto-aligned)
TIERS = (
    {"name": "fast", "label": "Fast", "k": 1, "model": "1.5B"},
    {"name": "balanced", "label": "Balanced", "k": 3, "model": "3B"},
    {"name": "heavy", "label": "Heavy", "k": 5, "model": "7B"},
)
TIER_CONFIG_IDS = tuple(config_id(t["k"], t["model"]) for t in TIERS)

# Generation 
MAX_NEW_TOKENS = 256
SMOKE_MAX_NEW_TOKENS = 64
MAX_PROMPT_TOKENS = 24000   # safety cap; documents are truncated evenly beyond this
WARMUP_RUNS = 3
RESIDENT_MIN_FREE_GIB = 30  # >= this free VRAM -> keep 1.5B+3B+7B resident together
SEED = 42

SMOKE_N_VAL = 6
SMOKE_N_TEST = 10

SYSTEM_PROMPT = (
    "You are a technical support assistant. Answer the question using ONLY the "
    "documentation context provided. Be concise and precise. If the context does "
    "not contain the answer, say so briefly."
)

# Metrics
QUALITY_WEIGHTS = {
    "evidence_token_f1": 0.35,
    "rouge_l_f1": 0.25,
    "semantic_similarity": 0.25,
    "context_faithfulness": 0.15,
}
# Retrieval gold: "relevant" = documents holding annotated relevant sentences
RETRIEVAL_GOLD_MODE = "relevant"
RRF_K = 60
RRF_POOL = 50

# Router calibration 
ROUTER_QUALITY_TOLERANCE = 0.015
ROUTER_GRID = {
    "fast_top1_min": (0.30, 0.45, 0.60, 0.75, 0.90),
    "fast_margin_min": (0.0, 0.30, 0.50, 0.70),
    "fast_complexity_max": (0.50, 0.75, 1.0),
    "heavy_top1_max": (0.0, 0.10, 0.20, 0.30, 0.45, 0.60),
    "heavy_conc_max": (0.0, 0.10, 0.25, 0.40),
    "heavy_complexity_min": (1.0, 0.90, 0.75, 0.50, 0.30),
}
DISABLED_LOW, DISABLED_HIGH = -1e9, 1e9

# Statistics
ALPHA = 0.05
NONINFERIORITY_DELTA = 0.02
BOOTSTRAP_RESAMPLES = 10_000
GUIDELINE_QUALITY_SLACK = 0.02
