"""Keep the breaker's RED stubs out of the builders' default `pytest tests/`
run, while leaving them runnable on demand.

These files are intentionally RED (they encode contracts the slices haven't
implemented yet — F1/F2/F3/F5/N1/N2/N3). If they were collected by the default
suite they'd add noise to every builder's green-check. So this directory is
skipped UNLESS explicitly targeted (`pytest tests/stubs/...`) or opted in via
RUN_BREAKER_STUBS=1.

The integrator / breaker run them deliberately:
    pytest tests/stubs/                 # path given explicitly → collected
    RUN_BREAKER_STUBS=1 pytest tests/   # opt-in across the whole tree
"""

import os


def _explicitly_targeted() -> bool:
    """True if the user pointed pytest at this dir/file directly (not via a
    bare recursive `tests/` collection)."""
    import sys
    return any("stubs" in arg for arg in sys.argv[1:])


if os.environ.get("RUN_BREAKER_STUBS") != "1" and not _explicitly_targeted():
    collect_ignore_glob = ["*.py"]
