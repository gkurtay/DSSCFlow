from __future__ import annotations

from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def write_manifest(
    output_path: Path,
    files: list[Path],
    parameters: dict,
    software_version: str,
) -> None:
    payload = {
        "software_version": software_version,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "parameters": parameters,
        "files": [
            {
                "path": str(p),
                "sha256": sha256_file(p),
            }
            for p in files
        ],
    }

    output_path.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8"
    )
