"""Append-only action audit log."""

import json
from datetime import datetime
from pathlib import Path


class AuditLog:
    def __init__(self, log_dir: str = "logs/audit"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def record(
        self,
        plan: dict,
        approved: bool,
        operator: str,
        result: dict | None = None,
    ) -> None:
        entry = {
            "timestamp": datetime.now().isoformat(),
            "operator": operator,
            "plan": plan,
            "approved": approved,
            "result": result,
        }
        log_file = self.log_dir / f"{datetime.now():%Y-%m-%d}.jsonl"
        with open(log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")
