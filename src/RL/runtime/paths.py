from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_ROOT.parents[1]
DEFAULT_CONFIG = PACKAGE_ROOT / "config" / "default.json"
