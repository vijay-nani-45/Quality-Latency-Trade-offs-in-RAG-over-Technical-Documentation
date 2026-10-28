
import config  # noqa: F401  (sets HF_HOME to ./hf_cache first)

import argparse
import os
import sys
import time
from pathlib import Path

from src.tier_sets import APPROX_DOWNLOAD_MB, LAPTOP_TIER_MODELS, TIER_ORDER

IGNORE = ["*.bin", "*.pt", "*.pth", "*.h5", "*.msgpack", "*.onnx", "onnx/*", "openvino/*", "*.ot", "*.gguf", "original/*"]


def gb(p: Path) -> float:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1024 ** 3


def hardware_line() -> str:
    try:
        import psutil

        return f"{psutil.virtual_memory().total / 1024 ** 3:.0f} GB RAM, {psutil.cpu_count(logical=False) or psutil.cpu_count()} CPU cores"
    except Exception:  # noqa: BLE001
        return "hardware details unavailable"


def fetch(repo: str, verify_only: bool) -> bool:
    from huggingface_hub import snapshot_download

    if os.path.isdir(repo):
        print(f"  [local folder] {repo}")
        return True
    try:
        path = Path(snapshot_download(repo_id=repo, ignore_patterns=IGNORE, local_files_only=verify_only))
        if not list(path.rglob("*.safetensors")):
            raise RuntimeError("no .safetensors weights found")
        print(f"  OK   {repo:<42} {gb(path):5.2f} GB")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"  FAIL {repo}: {exc}")
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-indices", action="store_true")
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()
    print(f"Machine: {hardware_line()}\nDownloads go to: {os.environ['HF_HOME']}\n")
    ok = True

    print("[1/3] Models")
    for tier in TIER_ORDER:
        print(f"  {tier:<7} (~{APPROX_DOWNLOAD_MB[tier]} MB)")
        ok &= fetch(LAPTOP_TIER_MODELS[tier], args.verify_only)
    ok &= fetch(config.EMBED_MODEL, args.verify_only)

    print("\n[2/3] Prepared data (results/prep/)")
    from src.demo_plots import index_status, prep_ready

    for ds in config.DATASET_ORDER:
        if prep_ready(ds):
            print(f"  OK   {ds}")
            continue
        if args.verify_only:
            print(f"  MISSING {ds}: copy results/prep/ from the project folder")
            ok = False
            continue
        print(f"  {ds}: not found, building from the dataset (needs internet, one time)...")
        try:
            from src.data_corpus import prepare_dataset

            s = prepare_dataset(ds)
            print(f"  OK   {ds}: {s['n_unique_documents']} documents, test questions {s['n_questions']['test']}")
        except Exception as exc:  # noqa: BLE001
            print(f"  FAIL {ds}: {exc}\n       Copy the results/prep/ folder from the project instead.")
            ok = False

    print("\n[3/3] Retrieval indices (results/indices/)")
    missing = []
    for ds in config.DATASET_ORDER:
        st = index_status(ds) if prep_ready(ds) else {"dense": False, "bm25": False}
        print(f"  {ds:<11} dense index: {'OK' if st['dense'] else 'missing'} | BM25 index: {'OK' if st['bm25'] else 'missing'}")
        if not (st["dense"] and st["bm25"]):
            missing.append(ds)
    if missing and not args.skip_indices and not args.verify_only:
        print("  Building the missing indices now. The dense index embeds every document once; on a laptop CPU this takes a few minutes.")
        import torch

        from src.data_corpus import load_corpus
        from src.retrievers import BM25Retriever, DenseRetriever, Embedder

        emb = Embedder(config.EMBED_MODEL, "cuda:0" if torch.cuda.is_available() else "cpu")
        for ds in missing:
            t0 = time.perf_counter()
            corpus = load_corpus(ds)
            BM25Retriever(corpus, ds)
            DenseRetriever(corpus, emb, ds)
            print(f"  built {ds} in {time.perf_counter() - t0:.0f}s")
    elif missing:
        print("  (indices will be built on the first run, or rerun this script without --skip-indices)")

    print("\n" + ("All set. Start the app with:  streamlit run app.py" if ok else "Some items need attention (see above)."))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
