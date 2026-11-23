"""Training, validation, interruption-safe checkpoints and resumption."""
import argparse
import json
import math
import platform
import signal
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .common import ROOT, configure, save_json, save_torch, sha256
from .data import CachedDataset
from .encoding import VERSION
from .model import PolicyNet


class Metrics:
    def __init__(self, totals=None):
        self.totals = totals or {}

    def add(self, losses, logits, targets, mask, inputs, plies):
        best = logits.topk(3, dim=1).indices
        top1 = (best[:, 0] == targets).detach().cpu().numpy()
        top3 = (best == targets[:, None]).any(1).detach().cpu().numpy()
        loss = losses.detach().cpu().numpy()
        uniform = (1.0 / mask.sum(1)).cpu().numpy()
        white = inputs[:, 12, 0, 0].cpu().numpy() == 1
        groups = {"all": np.ones(len(loss), dtype=bool), "white": white, "black": ~white, "after_ply20": plies > 20}
        for group, selected in groups.items():
            total = self.totals.setdefault(group, {"n": 0, "loss": 0.0, "top1": 0.0, "top3": 0.0, "uniform_top1": 0.0})
            total["n"] += int(selected.sum())
            for key, values in (("loss", loss), ("top1", top1), ("top3", top3), ("uniform_top1", uniform)):
                total[key] += float(values[selected].sum())

    def result(self):
        return {group: {"n": t["n"], **{k: v / t["n"] if t["n"] else None for k, v in t.items() if k != "n"}}
                for group, t in self.totals.items()}


@torch.inference_mode()
def evaluate(model, data, batch_size, device):
    model.eval()
    metrics = Metrics()
    for start in range(0, len(data), batch_size):
        x, target, mask, plies = data.batch(np.arange(start, min(start + batch_size, len(data))), device)
        logits = model(x).masked_fill(~mask, -1e9)
        losses = F.cross_entropy(logits, target, reduction="none")
        if not torch.isfinite(losses).all():
            raise ValueError("Non-finite evaluation loss")
        metrics.add(losses, logits, target, mask, x, plies)
    return metrics.result()


def weight_bundle(model, config, cache_id, epoch, steps, diagnostic):
    return {"encoding": VERSION, "architecture": model.config, "state_dict": model.state_dict(),
            "cache_id": cache_id, "epoch": epoch, "steps": steps,
            "diagnostic_only": diagnostic, "training_config": config}


def plot_history(run, history):
    if not history:
        return
    points = [(h["epoch"], h["train"]["all"]["loss"], h["validation"]["all"]["loss"]) for h in history]
    ymax = max(max(p[1:]) for p in points) * 1.1 or 1
    xmax = max(len(points) - 1, 1)
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="720" height="320" viewBox="0 0 720 320">',
             '<rect width="720" height="320" fill="white"/><text x="40" y="25">Perda por época: treino (azul), validação (laranja)</text>',
             '<path d="M40 45 V280 H690" stroke="black" fill="none"/>']
    for column, color in ((1, "#2563eb"), (2, "#ea580c")):
        coords = " ".join(f"{40+i*650/xmax:.1f},{280-p[column]*230/ymax:.1f}" for i, p in enumerate(points))
        parts.append(f'<polyline points="{coords}" stroke="{color}" fill="none" stroke-width="2"/>')
    parts.append(f'<text x="40" y="305">Épocas 1–{points[-1][0]}; escala vertical 0–{ymax:.2f}</text></svg>')
    (run / "learning_curve.svg").write_text("\n".join(parts), encoding="utf-8")


def train(args):
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.epochs is not None:
        config["epochs"] = args.epochs
    for key in ("channels", "blocks", "batch_size", "epochs", "threads", "patience"):
        if config[key] < 1:
            raise ValueError(f"{key} must be positive")
    device = configure(config)
    torch.manual_seed(config["seed"])
    train_data, validation = CachedDataset(args.data, "train"), CachedDataset(args.data, "validation")
    cache_id = sha256(args.data / "manifest.json")
    diagnostic = train_data.manifest["diagnostic_only"]
    run = args.run
    run.mkdir(parents=True, exist_ok=True)
    if not args.resume and any(run.iterdir()):
        raise ValueError("Pasta da execução já contém arquivos. Use --resume ou outra --run.")
    model = PolicyNet(config["channels"], config["blocks"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    epoch, steps, next_index, bad_epochs = 1, 0, 0, 0
    best_loss = math.inf
    order = None
    history = []
    metrics = Metrics()
    if args.resume:
        ckpt = torch.load(run / "last_checkpoint.pt", map_location="cpu", weights_only=True)
        if ckpt["cache_id"] != cache_id or ckpt["encoding"] != VERSION:
            raise ValueError("Checkpoint/cache incompatíveis")
        if {k: v for k, v in config.items() if k != "epochs"} != {k: v for k, v in ckpt["config"].items() if k != "epochs"}:
            raise ValueError("Na retomada, somente o total de épocas pode mudar.")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        epoch, steps, next_index, bad_epochs = (ckpt[k] for k in ("epoch", "steps", "next_index", "bad_epochs"))
        order, history, best_loss = ckpt["order"], ckpt["history"], ckpt["best_loss"]
        metrics = Metrics(ckpt["train_totals"])
        torch.set_rng_state(ckpt["rng"])
        if device.type == "cuda" and ckpt["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(ckpt["cuda_rng"])
    save_json(run / "config.json", config)
    save_json(run / "environment.json", {"python": platform.python_version(), "torch": str(torch.__version__),
                                         "device": str(device), "encoding": VERSION, "cache_id": cache_id,
                                         "diagnostic_only": diagnostic, "parameters": sum(p.numel() for p in model.parameters())})
    save_json(run / "history.json", history)
    stop = False

    def request_stop(signum, frame):
        nonlocal stop
        stop = True
        print("\nInterrupção solicitada: salvando após o lote atual...", flush=True)

    previous_handler = signal.signal(signal.SIGINT, request_stop)

    def checkpoint():
        save_torch(run / "last_checkpoint.pt", {
            "encoding": VERSION, "cache_id": cache_id, "config": config,
            "model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch,
            "steps": steps, "next_index": next_index, "bad_epochs": bad_epochs, "best_loss": best_loss,
            "order": order, "history": history, "train_totals": metrics.totals,
            "rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
        })

    start_steps = steps
    start_time = last_save = last_log = time.monotonic()
    try:
        print(f"Device={device}; parâmetros={sum(p.numel() for p in model.parameters()):,}; diagnóstico={diagnostic}", flush=True)
        while epoch <= config["epochs"] and bad_epochs < config["patience"]:
            if order is None:
                order = torch.randperm(len(train_data))
                next_index = 0
                metrics = Metrics()
            model.train()
            while next_index < len(order):
                indices = order[next_index:next_index + config["batch_size"]].numpy()
                x, targets, mask, plies = train_data.batch(indices, device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(x).masked_fill(~mask, -1e9)
                losses = F.cross_entropy(logits, targets, reduction="none")
                loss = losses.mean()
                if not torch.isfinite(loss):
                    raise ValueError("Perda não finita. Retome o último checkpoint válido após investigar.")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                metrics.add(losses, logits, targets, mask, x, plies)
                next_index += len(indices)
                steps += 1
                now = time.monotonic()
                if now - last_log >= 20:
                    print(f"Época {epoch}/{config['epochs']} | {next_index}/{len(order)} | perda={metrics.result()['all']['loss']:.4f}", flush=True)
                    last_log = now
                if stop or (args.max_steps and steps - start_steps >= args.max_steps):
                    checkpoint()
                    print(f"Pausado e salvo em {run / 'last_checkpoint.pt'}", flush=True)
                    return
                if now - last_save >= config["checkpoint_seconds"]:
                    checkpoint()
                    last_save = now
            checkpoint()  # Training is complete even if validation is interrupted externally.
            val = evaluate(model, validation, config["batch_size"], device)
            val_loss = val["all"]["loss"]
            improved = val_loss < best_loss - config["min_delta"]
            if improved:
                best_loss = val_loss
                bad_epochs = 0
                save_torch(run / "best_weights.pt", weight_bundle(model, config, cache_id, epoch, steps, diagnostic))
            else:
                bad_epochs += 1
            history.append({"epoch": epoch, "steps": steps, "train": metrics.result(), "validation": val})
            print(f"Época {epoch}: treino={metrics.result()['all']['loss']:.4f}, validação={val_loss:.4f}, top1={val['all']['top1']:.2%}", flush=True)
            epoch += 1
            order, next_index, metrics = None, 0, Metrics()
            checkpoint()
            save_json(run / "history.json", history)
            plot_history(run, history)
            if stop:
                return
        save_json(run / "training_summary.json", {"epochs_completed": epoch - 1, "steps": steps,
                  "best_validation_loss": best_loss if math.isfinite(best_loss) else None,
                  "early_stopped": bad_epochs >= config["patience"], "diagnostic_only": diagnostic,
                  "this_session_seconds": time.monotonic() - start_time})
        print("Treino concluído. Melhor modelo: " + str(run / "best_weights.pt"), flush=True)
    finally:
        signal.signal(signal.SIGINT, previous_handler)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/local_cpu.json")
    parser.add_argument("--data", type=Path, default=ROOT / "data/cache/aronian_v1")
    parser.add_argument("--run", type=Path, default=ROOT / "runs/aronian_v1")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-steps", type=int, help="Pausar após N lotes nesta chamada")
    train(parser.parse_args())
