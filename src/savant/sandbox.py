"""Run model-written Python in a separate process with time, memory and file limits.

The child process starts with isolated mode, an empty working directory, no proxy
settings and no inherited secrets. This is defense in depth for a personal tool. A public
deployment must also run inside a container with networking disabled.
"""

from __future__ import annotations

import json
import os
import resource
import signal
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

MARKER = "__SAVANT_RESULT__"
PRELUDE = f'''
import json as _json, math as _math
import polars as pl
import numpy as np

def _clean(v):
    if isinstance(v, (np.generic,)):
        v = v.item()
    if isinstance(v, float) and not _math.isfinite(v):
        return None
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v][:50]
    if isinstance(v, dict):
        return {{str(k): _clean(x) for k, x in list(v.items())[:50]}}
    if isinstance(v, (int, float, str, bool)) or v is None:
        return v
    return str(v)[:200]

def emit(name, value):
    """Record a result the report may cite as {{{{step.name}}}}."""
    print("{MARKER}" + _json.dumps({{"name": str(name), "value": _clean(value)}}))

df = pl.read_parquet(DATA_PATH)
'''


@dataclass
class Execution:
    ok: bool
    results: dict[str, object] = field(default_factory=dict)
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False


def _limits(mem_mb: int, cpu_s: int):
    def apply() -> None:
        resource.setrlimit(resource.RLIMIT_AS, (mem_mb * 2**20, mem_mb * 2**20))
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s))
        resource.setrlimit(resource.RLIMIT_FSIZE, (10 * 2**20, 10 * 2**20))
        os.setsid()

    return apply


def run(code: str, data_path: Path, timeout: int = 30, mem_mb: int = 2048) -> Execution:
    with tempfile.TemporaryDirectory(prefix="savant-") as work:
        script = Path(work) / "step.py"
        script.write_text(f"DATA_PATH = {str(data_path)!r}\n{PRELUDE}\n{code}\n")
        env = {"PATH": "/usr/bin:/bin", "HOME": work, "PYTHONHASHSEED": "0"}
        try:
            proc = subprocess.run(
                [sys.executable, "-I", str(script)],
                cwd=work,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout,
                preexec_fn=_limits(mem_mb, timeout),
            )
        except subprocess.TimeoutExpired as e:
            return Execution(False, stdout=str(e.stdout or "")[-2000:], timed_out=True)
    results: dict[str, object] = {}
    lines = []
    for line in proc.stdout.splitlines():
        if line.startswith(MARKER):
            item = json.loads(line[len(MARKER) :])
            results[item["name"]] = item["value"]
        else:
            lines.append(line)
    cpu_killed = proc.returncode in (-signal.SIGXCPU, -signal.SIGKILL)
    return Execution(
        ok=proc.returncode == 0,
        timed_out=cpu_killed,
        results=results,
        stdout="\n".join(lines)[-2000:],
        stderr=proc.stderr[-2000:],
    )
