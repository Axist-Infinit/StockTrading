"""Compatibility wrapper for loading axist-technical.py as a module named
`axist_technical`. This allows unit tests to import the original script
without renaming the file."""
from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path

_spec = spec_from_file_location("axist_technical_orig", Path(__file__).with_name("axist-technical.py"))
_module = module_from_spec(_spec)
_spec.loader.exec_module(_module)  # type: ignore[arg-type]

# Re-export all public attributes
for _name in dir(_module):
    if not _name.startswith("_"):
        globals()[_name] = getattr(_module, _name)

del _name, _module, _spec
