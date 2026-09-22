"""Load the existing Exp35 package with an experiment-local output root.

Each entry point is a fresh process. Execute the original initializer with the
local __file__, then resolve all submodules from the unchanged Exp35 sources.
This keeps its cache setup, adapters and training loops in one place.
"""
import sys
from types import ModuleType
from pathlib import Path


def configure(root):
    root = Path(root).resolve()
    repo = root.parent
    package = ModuleType('exp35')
    package.__file__ = str(root / '__init__.py')
    package.__path__ = [str(repo / 'exp35')]
    package.__package__ = 'exp35'
    sys.modules['exp35'] = package
    source = repo / 'exp35/__init__.py'
    exec(compile(source.read_text(), str(source), 'exec'), package.__dict__)
    from exp35 import config
    assert config.ROOT == root and config.DATA_ROOT == repo / 'data'
    return config
