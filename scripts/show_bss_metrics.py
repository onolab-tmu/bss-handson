import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--pattern", required=True)
    args = parser.parse_args()

    paths = sorted(Path(args.results_dir).glob(args.pattern))
    if not paths:
        raise RuntimeError(f"{args.results_dir}/{args.pattern} が見つかりません")

    for path in paths:
        metrics = json.loads(path.read_text())
        print(path.parent.name, metrics)


if __name__ == "__main__":
    main()
