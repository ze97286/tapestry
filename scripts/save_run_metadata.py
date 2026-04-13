#!/usr/bin/env python3
"""Save version metadata to a run directory.

Creates a run_metadata.json with git commit, tag, timestamp, and config.

Usage:
    python scripts/save_run_metadata.py --run-dir runs/run_v0.2 --tag v0.2
"""

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path


def get_git_info():
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"]
        ).strip().decode()
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"]
        ).strip().decode()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"]
        ).strip())
        return {"commit": commit, "branch": branch, "dirty": dirty}
    except Exception:
        return {"commit": "unknown", "branch": "unknown", "dirty": True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--tag", default="")
    parser.add_argument("--notes", default="")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "tag": args.tag,
        "notes": args.notes,
        "timestamp": datetime.now().isoformat(),
        "git": get_git_info(),
        "run_dir": str(run_dir.resolve()),
    }

    out_path = run_dir / "run_metadata.json"
    with open(out_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {out_path}")


if __name__ == "__main__":
    main()
