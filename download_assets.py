
import config  # noqa: F401  (sets HF_HOME before any HF import)

import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

IGNORE = ["*.bin", "*.pt", "*.pth", "*.h5", "*.msgpack", "*.onnx", "onnx/*", "openvino/*", "*.ot", "*.gguf", "original/*"]


def dir_gb(p: Path) -> float:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1024 ** 3


def fetch_model(repo_id: str, verify_only: bool) -> str:
    if os.path.isdir(repo_id):
        print(f"  [local] {repo_id}")
        return repo_id
    path = snapshot_download(repo_id=repo_id, ignore_patterns=IGNORE, local_files_only=verify_only)
    st = list(Path(path).rglob("*.safetensors"))
    if not st:
        raise RuntimeError(f"{repo_id}: no .safetensors weights found in {path}")
    print(f"  OK  {repo_id:<34} {dir_gb(Path(path)):6.2f} GB  ({len(st)} safetensors shard(s))")
    return path


def verify_tokenizer(repo_id: str) -> None:
    from transformers import AutoConfig, AutoTokenizer

    AutoConfig.from_pretrained(repo_id, local_files_only=True)
    AutoTokenizer.from_pretrained(repo_id, local_files_only=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-models", action="store_true")
    ap.add_argument("--skip-datasets", action="store_true")
    ap.add_argument("--verify-only", action="store_true", help="do not hit the network")
    args = ap.parse_args()
    print(f"HF_HOME = {os.environ['HF_HOME']}")
    failed = []

    if not args.skip_datasets:
        from src.data_corpus import inspect_schema, load_ragbench

        print("\n[datasets]")
        for ds in config.DATASETS:
            try:
                if args.verify_only:
                    os.environ["HF_DATASETS_OFFLINE"] = "1"
                splits = load_ragbench(ds)
                inspect_schema(ds, splits)
                print(f"  OK  {ds:<11} " + ", ".join(f"{s}={len(splits[s])}" for s in config.SPLITS)
                      + f"  columns={list(splits['test'].column_names)[:6]}...")
            except Exception as exc:  # noqa: BLE001
                failed.append(ds)
                print(f"  FAIL {ds}: {exc!r}")

    if not args.skip_models:
        print("\n[models]")
        for repo in [*(config.MODEL_IDS[k] for k in config.MODEL_ORDER), config.EMBED_MODEL]:
            try:
                fetch_model(repo, args.verify_only)
                if repo != config.EMBED_MODEL:
                    verify_tokenizer(repo)
            except Exception as exc:  # noqa: BLE001
                failed.append(repo)
                print(f"  FAIL {repo}: {exc!r}")

    print(f"\nCache size: {dir_gb(config.CACHE_DIR):.2f} GB at {config.CACHE_DIR}")
    if failed:
        print("FAILED:", failed)
        return 1
    print("All assets present.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
