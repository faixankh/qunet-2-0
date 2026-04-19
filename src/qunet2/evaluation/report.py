from __future__ import annotations
from pathlib import Path
import json

def write_markdown_report(metrics: dict, path: str | Path):
    path = Path(path)
    lines = ["# QUNet 2.0 Evaluation Report", ""]
    for k, v in metrics.items():
        lines.append(f"- **{k}**: {v}")
    path.write_text("\n".join(lines))

def write_json_report(metrics: dict, path: str | Path):
    Path(path).write_text(json.dumps(metrics, indent=2))
