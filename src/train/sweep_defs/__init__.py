"""Sweep dict definitions, grouped by environment.

Each env module exposes uppercase constants (no leading underscore). The
`from .X import *` re-exports here are how `sweeps_active.py` and other
callers reach the named sweep dicts.
"""
from .ant import *           # noqa: F401,F403
from .walker2d import *      # noqa: F401,F403
from .half_cheetah import *  # noqa: F401,F403
from .swimmer import *       # noqa: F401,F403
from .hopper import *        # noqa: F401,F403
from .mo_pointmaze import *  # noqa: F401,F403
