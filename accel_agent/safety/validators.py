"""Parameter range validation from safety_limits.toml."""

import tomllib
from pathlib import Path


class SafetyError(Exception):
    pass


class ParameterValidator:
    """Range validation loaded from safety_limits.toml."""

    def __init__(self, limits_path: str = "configs/safety_limits.toml"):
        path = Path(limits_path)
        self.limits = {}
        if path.exists():
            with open(path, "rb") as f:
                self.limits = tomllib.load(f)

    def check(self, device_name: str, value) -> None:
        limits = self._find_limits(device_name)
        if limits is None:
            raise SafetyError(
                f"No safety limits defined for {device_name}. "
                "Cannot write without defined limits."
            )
        if not (limits["min"] <= value <= limits["max"]):
            raise SafetyError(
                f"Value {value} for {device_name} outside safe range "
                f"[{limits['min']}, {limits['max']}] {limits.get('units', '')}"
            )

    def _find_limits(self, device_name: str) -> dict | None:
        if device_name in self.limits:
            return self.limits[device_name]
        import fnmatch
        for pattern, limits in self.limits.items():
            if "*" in pattern and fnmatch.fnmatch(device_name, pattern):
                return limits
        return None
