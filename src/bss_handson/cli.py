from pathlib import Path
import runpy


def main() -> None:
    script_path = Path.cwd() / "scripts" / "run_bss.py"
    if not script_path.exists():
        raise RuntimeError(
            "scripts/run_bss.py が見つかりません．"
            "pyproject.toml のある bss-handson/ で実行してください．"
        )
    runpy.run_path(str(script_path), run_name="__main__")
