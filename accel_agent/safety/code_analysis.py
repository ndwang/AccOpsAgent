"""Static analysis for Python sandbox (blocked imports/calls)."""

import ast
from dataclasses import dataclass


@dataclass
class AnalysisResult:
    safe: bool
    violations: list[str]
    warnings: list[str]


BLOCKED_IMPORTS = {
    "epics", "pcaspy", "pyepics", "cothread",
    "tango", "PyTango", "taurus", "pydoocs",
    "subprocess", "shutil", "ctypes",
    "socket", "http", "urllib", "requests",
    "httpx", "aiohttp", "ftplib", "smtplib",
    "xmlrpc", "paramiko", "fabric", "code", "codeop",
}

ALLOWED_IMPORTS = {
    "numpy", "scipy", "pandas", "matplotlib", "matplotlib.pyplot",
    "json", "csv", "math", "statistics", "datetime", "time",
    "collections", "itertools", "functools", "dataclasses",
    "typing", "re", "io", "os.path", "glob",
    "h5py", "tables", "sqlite3", "struct", "pickle",
    "logging", "warnings", "sklearn", "lmfit", "PIL", "skimage",
}

BLOCKED_CALLS = {
    "exec", "eval", "compile", "__import__",
    "os.system", "os.popen",
    "os.remove", "os.unlink", "os.rmdir", "os.rename",
    "os.chmod", "os.chown",
}


class CodeAnalyzer:
    def analyze(self, code: str) -> AnalysisResult:
        violations, warnings = [], []
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return AnalysisResult(False, [f"Syntax error: {e}"], [])
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    self._check_import(alias.name, violations, warnings)
            elif isinstance(node, ast.ImportFrom) and node.module:
                self._check_import(node.module, violations, warnings)
            elif isinstance(node, ast.Call):
                name = self._get_call_name(node)
                if name in BLOCKED_CALLS:
                    violations.append(f"Blocked call: {name}() at line {node.lineno}")
        return AnalysisResult(len(violations) == 0, violations, warnings)

    def _check_import(self, module: str, violations: list, warnings: list) -> None:
        root = module.split(".")[0]
        if root in BLOCKED_IMPORTS or module in BLOCKED_IMPORTS:
            violations.append(f"Blocked import: {module}")
        elif root not in ALLOWED_IMPORTS and module not in ALLOWED_IMPORTS:
            warnings.append(f"Unknown import: {module}")

    def _get_call_name(self, node: ast.Call) -> str:
        if isinstance(node.func, ast.Name):
            return node.func.id
        if isinstance(node.func, ast.Attribute):
            parts = []
            n = node.func
            while isinstance(n, ast.Attribute):
                parts.append(n.attr)
                n = n.value
            if isinstance(n, ast.Name):
                parts.append(n.id)
            return ".".join(reversed(parts))
        return ""
