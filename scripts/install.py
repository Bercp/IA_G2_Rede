"""Install the local game in its own virtual environment using CPU packages."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def run(*args):
    subprocess.run([str(arg) for arg in args], cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", action="store_true", help="Install the original Python 3.13 training environment")
    args = parser.parse_args()
    if not (3, 10) <= sys.version_info[:2] <= (3, 13) or sys.maxsize <= 2**32:
        parser.error("Use Python 3.10 a 3.13 de 64 bits.")

    env_dir = ROOT / ".venv"
    python = env_dir / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    if not python.is_file():
        run(sys.executable, "-m", "venv", env_dir)
    info = json.loads(subprocess.check_output(
        [str(python), "-c", "import json,sys; print(json.dumps([*sys.version_info[:2],sys.maxsize>2**32]))"],
        cwd=ROOT, text=True))
    if not (3, 10) <= tuple(info[:2]) <= (3, 13) or not info[2]:
        parser.error("A .venv existente precisa usar Python 3.10 a 3.13 de 64 bits.")
    if args.training and tuple(info[:2]) != (3, 13):
        parser.error("A reprodução do treino original exige uma .venv com Python 3.13. Para jogar, omita --training.")

    version = "2.15.0" if args.training else "2.14.1"
    requirements = "requirements-training.txt" if args.training else "requirements-play.txt"
    print(f"Instalando em .venv: Python {info[0]}.{info[1]}, CPU", flush=True)
    run(python, "-m", "pip", "install", "--upgrade", "pip")
    run(python, "-m", "pip", "install", f"torch=={version}", "--index-url", "https://download.pytorch.org/whl/cpu")
    run(python, "-m", "pip", "install", "-r", requirements)
    run(python, "-m", "pip", "check")
    run(python, "-c", "from aronian.predict import load_model,choose_move; import chess; "
        "model,_=load_model('runs/aronian_v1/best_weights.pt'); "
        "board=chess.Board(); result=choose_move(model,board); "
        "assert chess.Move.from_uci(result['move_uci']) in board.legal_moves; "
        "print('Modelo carregado e jogada legal conferida.')")
    print("Instalação concluída. Execute .\\jogar.cmd", flush=True)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        print("Instalação interrompida. Corrija o erro acima antes de abrir o jogo.", file=sys.stderr)
        raise SystemExit(error.returncode or 1)
