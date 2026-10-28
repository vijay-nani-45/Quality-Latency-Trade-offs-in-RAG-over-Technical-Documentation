
import config  # noqa: F401  (MUST be first: sets HF cache env vars)

import argparse
import json
import logging
import random
import shutil
import sys
import time

import numpy as np
from tqdm import tqdm

from src.adaptive_router import (TIER_NAMES, AdaptiveRouter, calibrate_from_validation, calibration_path,
                                 features_for_queries, load_calibration, save_calibration, unique_models)
from src.data_corpus import JsonlLog, load_corpus, load_queries, prepare_dataset, read_jsonl
from src.metrics import retrieval_metrics

log = logging.getLogger("run")
STAGE_ORDER = ["prep", "retriever", "fixed", "adaptive", "report"]


# Runtime container
class DatasetContext:
    def __init__(self, rt, name):
        self.rt, self.name = rt, name
        prepare_dataset(name, rt.args.local_data_dir)  # no-op when already pooled
        self.corpus = load_corpus(name)
        self._dense = None
        self._all = None
        self._evaluator = None

    def dense(self):
        if self._dense is None:
            from src.retrievers import DenseRetriever

            self._dense = DenseRetriever(self.corpus, self.rt.embedder(), self.name)
        return self._dense

    def all_retrievers(self):
        if self._all is None:
            from src.retrievers import BM25Retriever, HybridRRFRetriever

            bm25 = BM25Retriever(self.corpus, self.name)
            self._all = {"bm25": bm25, "dense": self.dense(), "hybrid_rrf": HybridRRFRetriever(bm25, self.dense())}
        return self._all

    @property
    def evaluator(self):
        if self._evaluator is None:
            from src.metrics import QualityEvaluator

            self._evaluator = QualityEvaluator(self.rt.embedder(), self.corpus)
        return self._evaluator


class Runtime:
    def __init__(self, args, paths):
        self.args, self.paths = args, paths
        self.device = None
        self._embedder = None
        self._ctx: dict[str, DatasetContext] = {}
        self._mgr = None
        self.datasets = list(args.datasets)
        smoke = args.smoke_test
        self.n_val = args.max_val or (config.SMOKE_N_VAL if smoke else None)
        self.n_test = args.max_test or (config.SMOKE_N_TEST if smoke else None)
        self.max_new = config.SMOKE_MAX_NEW_TOKENS if smoke else config.MAX_NEW_TOKENS

    def init_device(self):
        if self.device is not None:
            return
        import torch

        from src.generators import collect_hardware_env

        if torch.cuda.is_available():
            if self.args.gpu >= torch.cuda.device_count():
                raise SystemExit(f"--gpu {self.args.gpu} invalid: only {torch.cuda.device_count()} visible GPU(s).")
            self.device = torch.device(f"cuda:{self.args.gpu}")
            torch.cuda.set_device(self.device)
        elif self.args.allow_cpu:
            log.warning("CUDA not available: running on CPU (timings are NOT hardware-valid).")
            self.device = torch.device("cpu")
        else:
            raise SystemExit("No CUDA GPU visible. Request a GPU node (or pass --allow-cpu for plumbing tests only).")
        env = collect_hardware_env(self.args.gpu, self.device)
        env["smoke_test"] = self.args.smoke_test
        self.paths.hardware_json.write_text(json.dumps(env, indent=2))
        log.info("Hardware: %s", {k: env[k] for k in ("gpu_name", "total_vram_gb", "free_vram_gb", "bf16_supported")})

    def embedder(self):
        if self._embedder is None:
            from src.retrievers import Embedder

            self.init_device()
            self._embedder = Embedder(config.EMBED_MODEL, str(self.device))
        return self._embedder

    def ctx(self, name) -> DatasetContext:
        if name not in self._ctx:
            self._ctx[name] = DatasetContext(self, name)
        return self._ctx[name]

    def queries(self, name, split):
        limit = self.n_val if split == "validation" else self.n_test if split == "test" else None
        return load_queries(name, split, limit)

    def manager(self):
        if self._mgr is None:
            from src.generators import ModelManager

            self.init_device()
            self._mgr = ModelManager(self.device, self.max_new)
        return self._mgr


# Shared per-query worker
def process_query(ctx, engine, q, indices, retrieval_ms, router_ms, k, model_key, split, extra=None) -> dict:
    corpus = ctx.corpus
    top = [int(i) for i in indices[:k]]
    doc_ids = [corpus.doc_ids[i] for i in top]
    pb = engine.build_prompt(q.question, [corpus.texts[i] for i in top])
    gen = engine.generate(pb.input_ids)
    quality = ctx.evaluator.score(gen.text, q.reference_response, q.evidence_sentences, doc_ids)
    rmet = retrieval_metrics(doc_ids, q.retrieval_gold(), k)
    row = {
        "qid": q.qid, "dataset": q.dataset, "split": split, "config_id": config.config_id(k, model_key),
        "k": k, "model": model_key, "retrieved_doc_ids": doc_ids, "answer": gen.text,
        "prompt_tokens": pb.prompt_tokens, "output_tokens": gen.output_tokens, "prompt_truncated": pb.truncated,
        "hit_max_tokens": gen.hit_max_tokens,
        "retrieval_latency_ms": retrieval_ms, "router_latency_ms": router_ms, "prompt_build_ms": pb.build_ms,
        "ttft_ms": gen.ttft_ms, "decode_latency_ms": gen.decode_latency_ms, "ms_per_token": gen.ms_per_token,
        "generation_latency_ms": gen.generation_latency_ms,
        "total_latency_ms": retrieval_ms + router_ms + pb.build_ms + gen.generation_latency_ms,
        **rmet, **quality,
    }
    if extra:
        row.update(extra)
    return row


# Stages
def stage_prep(rt: Runtime):
    log.info("== STAGE prep: pooling corpora and building/verifying indices ==")
    for ds in rt.datasets:
        s = prepare_dataset(ds, rt.args.local_data_dir, force=rt.args.force_prep)
        log.info("[%s] unique docs=%d (brief expects ~%d) | questions=%s | duplicate question texts=%d | "
                 "evidence fallbacks=%d", ds, s["n_unique_documents"], s["expected_unique_documents"],
                 s["n_questions"], s["n_duplicate_question_texts"], s["rows_with_evidence_fallback_to_response"])
    rt.init_device()
    for ds in rt.datasets:
        c = rt.ctx(ds)
        retrs = c.all_retrievers()
        assert retrs["dense"].index.ntotal == len(c.corpus), "FAISS index size mismatch"
        res = retrs["dense"].retrieve("verification query", 3)
        log.info("[%s] indices OK (docs=%d, dense probe %.1f ms)", ds, len(c.corpus), res.latency_ms)


def stage_retriever(rt: Runtime):
    log.info("== STAGE retriever: BM25 vs Dense vs Hybrid-RRF ==")
    rt.init_device()
    for ds in rt.datasets:
        c = rt.ctx(ds)
        qs = rt.queries(ds, "test")
        for name, retr in c.all_retrievers().items():
            logf = JsonlLog(rt.paths.raw_logs / f"retriever_{ds}_{name}_test.jsonl")
            pending = [q for q in qs if q.qid not in logf]
            if not pending:
                log.info("[%s/%s] already complete", ds, name)
                continue
            for q in qs[:3]:  # untimed warmup
                retr.retrieve(q.question, config.MAX_RETRIEVE)
            for q in tqdm(pending, desc=f"retriever {ds}/{name}"):
                res = retr.retrieve(q.question, config.MAX_RETRIEVE)
                ids = [c.corpus.doc_ids[int(i)] for i in res.indices]
                row = {"qid": q.qid, "dataset": ds, "retriever": name, "split": "test",
                       "retrieval_latency_ms": res.latency_ms}
                for k in config.RETRIEVER_K_VALUES:
                    m = retrieval_metrics(ids, q.retrieval_gold(), k)
                    row[f"recall_at_{k}"], row[f"mrr_at_{k}"] = m["recall_at_k"], m["mrr_at_k"]
                logf.append(row)
            logf.close()


def stage_fixed(rt: Runtime):
    log.info("== STAGE fixed: 12-configuration grid (k x model) on validation + test ==")
    rt.init_device()
    mgr = rt.manager()
    for model_key in config.MODEL_ORDER:
        jobs = []
        for ds in rt.datasets:
            for split in ("validation", "test"):
                qs = rt.queries(ds, split)
                for k in config.FIXED_K_VALUES:
                    logf = JsonlLog(rt.paths.raw_logs / f"fixed_{ds}_{split}_{config.config_id(k, model_key)}.jsonl")
                    pending = [q for q in qs if q.qid not in logf]
                    if pending:
                        jobs.append((ds, split, k, pending, logf))
        if not jobs:
            log.info("[%s] all fixed configs already complete - skipping model load", model_key)
            continue
        engine = mgr.acquire(model_key)
        for ds, split, k, pending, logf in jobs:
            c = rt.ctx(ds)
            dense = c.dense()
            for q in tqdm(pending, desc=f"{config.config_id(k, model_key)} {ds}/{split}"):
                res = dense.retrieve(q.question, k)
                logf.append(process_query(c, engine, q, res.indices, res.latency_ms, 0.0, k, model_key, split))
            logf.close()
        mgr.release(model_key)


def _fixed_val_table(rt: Runtime, ds: str) -> dict:
    """{config_id: {qid: (quality, latency)}} from the validation logs."""
    out = {}
    for cid in config.CONFIG_ORDER:
        rows = read_jsonl(rt.paths.raw_logs / f"fixed_{ds}_validation_{cid}.jsonl")
        out[cid] = {r["qid"]: (r["composite_quality"], r["total_latency_ms"]) for r in rows}
    return out


def calibrate_dataset(rt: Runtime, ds: str) -> dict:
    """Frozen validation-only calibration: Pareto tier selection + difficulty thresholds.
    Uses ONLY the validation logs of the fixed grid; never reads a test row."""
    cal_path = calibration_path(ds, rt.paths)
    if cal_path.exists() and not rt.args.recalibrate:
        log.info("[%s] using FROZEN validation calibration %s", ds, cal_path)
        return load_calibration(cal_path)
    table = _fixed_val_table(rt, ds)
    empty = [c for c, d in table.items() if not d]
    if empty:
        raise SystemExit(f"[{ds}] validation logs missing for {empty}. Run: python run_experiments.py --stage fixed")
    qids = set.intersection(*[set(d) for d in table.values()])
    val_qs = [q for q in rt.queries(ds, "validation") if q.qid in qids]
    if not val_qs:
        raise SystemExit(f"[{ds}] no validation queries common to all 12 fixed configs.")
    feats = features_for_queries(val_qs, rt.ctx(ds).dense())
    cal = calibrate_from_validation(ds, [q.qid for q in val_qs], feats, table)
    save_calibration(cal_path, cal, rt.paths.tables)
    log.info("[%s] calibration saved to %s (id %s)", ds, cal_path, cal["calibration_id"])
    return cal


def archive_stale_adaptive(rt: Runtime, ds: str, cal: dict) -> None:
    """Adaptive rows written under a different calibration must not be mixed with new ones.
    They are moved (not deleted) to raw_logs/archive/. Fixed/retriever logs are never touched."""
    f = rt.paths.raw_logs / f"adaptive_{ds}_test.jsonl"
    rows = read_jsonl(f)
    if rows and any(r.get("calibration_id") != cal["calibration_id"] for r in rows):
        dst_dir = rt.paths.raw_logs / "archive"
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / f"adaptive_{ds}_test.{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
        shutil.move(str(f), str(dst))
        log.warning("[%s] existing adaptive log was produced under a different calibration -> archived to %s", ds, dst)


def stage_adaptive(rt: Runtime):
    log.info("== STAGE adaptive: validation-Pareto calibration + live adaptive pipeline on test ==")
    rt.init_device()
    mgr = rt.manager()
    cals, routers, plans = {}, {}, {}
    for ds in rt.datasets:  # Phase A: live retrieval + routing (timed per query)
        c = rt.ctx(ds)
        cal = calibrate_dataset(rt, ds)
        cals[ds], routers[ds] = cal, AdaptiveRouter(cal)
        archive_stale_adaptive(rt, ds, cal)
        depth = max([config.ROUTER_TOP_N] + [t["k"] for t in cal["tiers"]])
        log.info("[%s] tiers: fast=%s balanced=%s heavy=%s | thresholds t1=%.3f t2=%.3f | retrieval depth %d", ds,
                 cal["fast_cfg"], cal["balanced_cfg"], cal["heavy_cfg"], cal["t1"], cal["t2"], depth)
        dense = c.dense()
        logf = JsonlLog(rt.paths.raw_logs / f"adaptive_{ds}_test.jsonl")
        qs = rt.queries(ds, "test")
        for q in qs[:3]:  # untimed warmup of the retrieval path
            dense.retrieve(q.question, depth)
        planned = []
        for q in [q for q in qs if q.qid not in logf]:
            res = dense.retrieve(q.question, depth)
            tier, feats, router_ms = routers[ds].route(q.question, res.scores)
            planned.append({"q": q, "res": res, "tier": tier, "feats": feats, "router_ms": router_ms})
        plans[ds] = (planned, logf)
        log.info("[%s] %d queries routed, %d already done", ds, len(planned), len(qs) - len(planned))

    pending_models = unique_models([{"tiers": [routers[ds].tiers[it["tier"]]]} for ds, (pl, _) in plans.items() for it in pl])
    if not pending_models:
        log.info("adaptive stage already complete for all datasets - nothing to run")
        for _, logf in plans.values():
            logf.close()
        return
    resident = mgr.can_keep_resident(pending_models)  # Phase B: generation

    def run_item(ds, item, engine):
        cid, k_tier, model = routers[ds].tier_config(item["tier"])
        extra = {"tier": TIER_NAMES[item["tier"]], "tier_index": item["tier"], "resident_mode": resident,
                 "calibration_id": cals[ds]["calibration_id"], **{f"feat_{n}": v for n, v in item["feats"].items()}}
        row = process_query(rt.ctx(ds), engine, item["q"], item["res"].indices, item["res"].latency_ms, item["router_ms"],
                            k_tier, model, "test", extra)
        assert row["config_id"] == cid
        plans[ds][1].append(row)

    if resident:
        for m in pending_models:
            mgr.acquire(m)
        for ds, (planned, _) in plans.items():
            for item in tqdm(planned, desc=f"adaptive {ds} (resident)"):
                run_item(ds, item, mgr.engines[routers[ds].tier_config(item["tier"])[2]])
    else:
        for m in pending_models:
            work = [(ds, it) for ds, (pl, _) in plans.items() for it in pl if routers[ds].tier_config(it["tier"])[2] == m]
            engine = mgr.acquire(m)
            for ds, item in tqdm(work, desc=f"adaptive model {m} (grouped)"):
                run_item(ds, item, engine)
            mgr.release(m)
    mgr.release_all()
    for _, logf in plans.values():
        logf.close()


def stage_report(rt: Runtime):
    log.info("== STAGE report: hypothesis tests, tables, figures ==")
    from src.stats_and_plots import generate_all

    generate_all(rt.paths, rt.datasets)


STAGES = {"prep": stage_prep, "retriever": stage_retriever, "fixed": stage_fixed,
          "adaptive": stage_adaptive, "report": stage_report}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", default="all", choices=STAGE_ORDER + ["all"])
    p.add_argument("--gpu", type=int, default=0, help="CUDA device index (models are pinned to cuda:<gpu>)")
    p.add_argument("--smoke-test", action="store_true",
                   help=f"{config.SMOKE_N_VAL} val + {config.SMOKE_N_TEST} test queries per dataset, "
                        f"{config.SMOKE_MAX_NEW_TOKENS} new tokens; writes to results_smoke/")
    p.add_argument("--datasets", nargs="+", default=list(config.DATASET_ORDER), choices=list(config.DATASETS))
    p.add_argument("--max-val", type=int, default=None, help="limit validation queries per dataset")
    p.add_argument("--max-test", type=int, default=None, help="limit test queries per dataset")
    p.add_argument("--recalibrate", action="store_true", help="recompute the frozen router calibration from validation logs (stale adaptive logs are archived)")
    p.add_argument("--fresh", action="store_true", help="delete existing raw logs of the selected stage(s) first")
    p.add_argument("--force-prep", action="store_true", help="rebuild pooled corpora from the HF dataset")
    p.add_argument("--local-data-dir", default=None, help="read <dir>/<dataset>/<split>.jsonl instead of the HF hub")
    p.add_argument("--allow-cpu", action="store_true", help=argparse.SUPPRESS)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    paths = config.make_paths(args.smoke_test)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(paths.logs / "run.log")])
    logging.getLogger("faiss.loader").setLevel(logging.WARNING)  # its AVX-probing INFO lines look like errors
    random.seed(config.SEED)
    np.random.seed(config.SEED)
    stages = STAGE_ORDER if args.stage == "all" else [args.stage]
    if args.fresh:
        for st in stages:
            for f in paths.raw_logs.glob({"retriever": "retriever_*", "fixed": "fixed_*", "adaptive": "adaptive_*"}.get(st, "__none__")):
                f.unlink()
    rt = Runtime(args, paths)
    t0 = time.time()
    for st in stages:
        t1 = time.time()
        STAGES[st](rt)
        log.info("stage '%s' finished in %.1f s", st, time.time() - t1)
    log.info("All requested stages done in %.1f s. Outputs: %s", time.time() - t0, paths.root)
    shutil.rmtree(config.PROJECT_ROOT / "__pycache__", ignore_errors=True)


if __name__ == "__main__":
    main()
