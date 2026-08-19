"""Stable official-client Windows eyes-and-hands surface."""
from experiments.dti.windows_base import *
from experiments.dti.windows_guard import WindowsTargetGuard

# Compatibility repair for the current stacked branch: three private guard
# methods call the historical misspelling ``is_windos``.  Keep the correction
# at the import seam so every WindowsGameDriver path fails closed rather than
# crashing before the target-window refusal receipt can be written.
if not hasattr(WindowsTargetGuard, "is_windos"):
    WindowsTargetGuard.is_windos = staticmethod(WindowsTargetGuard.is_windows)

from experiments.dti.windows_io import WindowsGameDriver
