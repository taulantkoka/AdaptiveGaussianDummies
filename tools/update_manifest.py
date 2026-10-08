#!/usr/bin/env python3
"""Hash tracked and untracked source files, respecting the repository ignore rules."""

import hashlib
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main():
    files = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=ROOT,
    ).decode().split("\0")
    lines = []
    for name in sorted(set(files)):
        path = ROOT / name
        if not name or name == "MANIFEST.sha256" or not path.is_file():
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  ./{name}\n")
    (ROOT / "MANIFEST.sha256").write_text("".join(lines))
    print(f"Hashed {len(lines)} source/reference files")


if __name__ == "__main__":
    main()
