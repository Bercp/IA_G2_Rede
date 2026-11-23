import argparse
from pathlib import Path

from .common import ROOT, save_json, sha256
from .data import CachedDataset
from .predict import load_model
from .training import evaluate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Avaliação final do melhor modelo no teste")
    parser.add_argument("--run", type=Path, default=ROOT / "runs/aronian_v1")
    parser.add_argument("--data", type=Path, default=ROOT / "data/cache/aronian_v1")
    args = parser.parse_args()
    model, bundle = load_model(args.run / "best_weights.pt")
    if bundle["cache_id"] != sha256(args.data / "manifest.json"):
        raise ValueError("Cache differs from training")
    metrics = evaluate(model, CachedDataset(args.data, "test"), bundle["training_config"]["batch_size"], "cpu")
    result = {"split": "test", "weights_sha256": sha256(args.run / "best_weights.pt"),
              "diagnostic_only": bundle["diagnostic_only"], "metrics": metrics}
    save_json(args.run / "test_metrics.json", result)
    print(result)
