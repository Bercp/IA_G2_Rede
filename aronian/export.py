"""Export a portable bundle and check PyTorch/ONNX numerical parity."""
import argparse
import json
import shutil
import zipfile
from pathlib import Path

import chess
import numpy as np
import torch

from .common import ROOT, save_json, sha256
from .encoding import SPEC, encode_board, legal_ids
from .predict import load_model, reconstruct


def parity_boards():
    return [chess.Board(), reconstruct(moves=["e2e4"]),
            reconstruct(moves=["e2e4", "a7a6", "e4e5", "d7d5"]),
            chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"),
            chess.Board("7k/P7/8/8/8/8/8/7K w - - 0 1"),
            chess.Board("7k/8/8/8/8/8/p7/7K b - - 0 1")]


def export(run, output, with_onnx=True):
    output.mkdir(parents=True, exist_ok=True)
    model, bundle = load_model(run / "best_weights.pt")
    # Reusing an export folder could mix artifacts from different models.
    if any(output.iterdir()):
        raise ValueError("Escolha uma pasta de exportação vazia para não misturar modelos.")
    shutil.copy2(run / "best_weights.pt", output / "best_weights.pt")
    save_json(output / "encoding.json", SPEC)
    save_json(output / "config.json", {"architecture": bundle["architecture"], "training": bundle["training_config"],
                                      "diagnostic_only": bundle["diagnostic_only"]})
    pkg = output / "aronian"
    pkg.mkdir()
    for filename in ("__init__.py", "model.py", "encoding.py", "predict.py"):
        shutil.copy2(ROOT / "aronian" / filename, pkg / filename)
    for filename in ("history.json", "test_metrics.json", "environment.json", "learning_curve.svg"):
        if (run / filename).exists():
            shutil.copy2(run / filename, output / filename)
    (output / "requirements-inference.txt").write_text(
        f"chess=={chess.__version__}\nnumpy=={np.__version__}\ntorch=={torch.__version__.split('+')[0]}\n", encoding="utf-8")
    parity = {"onnx_exported": False, "diagnostic_only": bundle["diagnostic_only"]}
    if with_onnx:
        import onnx
        import onnxruntime as ort
        boards = parity_boards()
        inputs = np.stack([encode_board(b) for b in boards])
        torch.onnx.export(model, (torch.from_numpy(inputs[:2]),), str(output / "model.onnx"),
                          input_names=["position"], output_names=["logits"],
                          dynamic_shapes=({0: torch.export.Dim("batch", min=1, max=512)},),
                          opset_version=18, dynamo=True, external_data=False, verbose=False)
        onnx.checker.check_model(str(output / "model.onnx"))
        session = ort.InferenceSession(str(output / "model.onnx"), providers=["CPUExecutionProvider"])
        with torch.inference_mode():
            expected = model(torch.from_numpy(inputs)).numpy()
        actual = session.run(None, {"position": inputs})[0]
        np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-4)
        single = session.run(None, {"position": inputs[:1]})[0]
        np.testing.assert_allclose(single, expected[:1], atol=1e-5, rtol=1e-4)
        cases = []
        for i, board in enumerate(boards):
            legal = legal_ids(board)
            best_torch = int(legal[expected[i, legal].argmax()])
            best_onnx = int(legal[actual[i, legal].argmax()])
            if best_torch != best_onnx:
                raise ValueError("ONNX legal argmax differs from PyTorch; inspect before delivery")
            cases.append({"fen": board.fen(en_passant="fen"), "legal_action_ids": legal.tolist(),
                          "expected_legal_logits": expected[i, legal].tolist(), "best_action_id": best_torch})
        save_json(output / "parity_cases.json", cases)
        parity = {**parity, "onnx_exported": True, "cases_verified": len(boards),
                  "max_absolute_error": float(np.max(np.abs(actual - expected))), "atol": 1e-5, "rtol": 1e-4,
                  "onnxruntime": ort.__version__, "opset": 18, "browser_verified": False}
    save_json(output / "export_report.json", parity)
    warning = "ATENÇÃO: este pacote contém pesos de diagnóstico, sem treinamento completo.\n\n" if bundle["diagnostic_only"] else ""
    (output / "README.md").write_text(
        "# Modelo de imitação de Aronian\n\n" + warning +
        "Instale PyTorch CPU pelo índice oficial e as dependências de requirements-inference.txt.\n\n"
        "Na pasta extraída: `python -m aronian.predict --weights best_weights.pt --moves e2e4`.\n\n"
        "Use `--play` para jogar de brancas no terminal. Informe lances UCI como e2e4 ou a7a8q.\n\n"
        "O modelo retorna pontuações. A aplicação deve gerar os lances legais e aplicar a máscara. "
        "Consulte encoding.json para o contrato completo. O controlador reivindica empates disponíveis automaticamente. "
        "Para repetições, forneça o histórico completo, não apenas a FEN.\n\n"
        "ONNX, quando presente, foi verificado em CPU. O runtime no navegador ainda precisa de validação própria.\n",
        encoding="utf-8")
    save_json(output / "checksums.json", {str(p.relative_to(output)).replace("\\", "/"): sha256(p)
              for p in output.rglob("*") if p.is_file() and "__pycache__" not in p.parts})
    zip_path = output.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for file in output.rglob("*"):
            if file.is_file() and "__pycache__" not in file.parts:
                z.write(file, file.relative_to(output))
    print(f"Pacote criado: {zip_path}")
    print(json.dumps(parity, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "runs/aronian_v1")
    parser.add_argument("--output", type=Path, default=ROOT / "exports/aronian_v1")
    parser.add_argument("--skip-onnx", action="store_true", help="Exportar apenas pacote PyTorch")
    args = parser.parse_args()
    export(args.run, args.output, not args.skip_onnx)
