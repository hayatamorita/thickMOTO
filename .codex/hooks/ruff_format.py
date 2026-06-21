import json
import re
import subprocess
import sys
from pathlib import Path


def main() -> int:
    payload = json.load(sys.stdin)
    if payload.get("tool_name") != "apply_patch":
        return 0

    command = payload.get("tool_input", {}).get("command", "")
    paths = re.findall(r"^\*\*\* (?:Add|Update) File: (.+\.py)$", command, re.MULTILINE)
    if not paths:
        return 0

    root = Path(payload["cwd"]).resolve()
    targets = []

    for path_text in paths:
        path = Path(path_text)
        path = path.resolve() if path.is_absolute() else (root / path).resolve()

        if path.is_relative_to(root) and path.is_file():
            targets.append(str(path))

    targets = sorted(set(targets))

    if not targets:
        return 0

    ruff = root / ".venv/bin/ruff"
    if not ruff.is_file():
        print("Ruff is not installed in .venv/bin/ruff", file=sys.stderr)
        return 2

    check_result = subprocess.run(
        [str(ruff), "check", "--fix", *targets],
        check=False,
    )

    format_result = subprocess.run(
        [str(ruff), "format", *targets],
        check=False,
    )

    return check_result.returncode or format_result.returncode


if __name__ == "__main__":
    raise SystemExit(main())