import argparse
from pathlib import Path
import sys

scripts_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(scripts_dir))

from run_bss import run_bss  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/bss.yaml")
parser.add_argument("overrides", nargs="*")
args = parser.parse_args()

run_bss(args.config, args.overrides)
