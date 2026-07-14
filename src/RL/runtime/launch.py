"""Human-facing architecture launcher and preflight inspector."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from RL.config.schema import load_config
from RL.runtime.paths import DEFAULT_CONFIG
from RL.runtime.validation import describe_paths, validate_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="quake-rl", description="Quake/Xonotic RL architecture launcher")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--check", action="store_true", help="Validate paths and configuration without launching anything")
    parser.add_argument("--show-config", action="store_true", help="Print the resolved configuration")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = load_config(args.config)

    print(f"Quake RL architecture: {config.project_name}")
    print(f"Engine: {config.engine.kind}")
    print(f"Match: {config.match.mode} on {config.match.map}")

    if args.show_config:
        print(json.dumps({k: str(v) for k, v in describe_paths(config).items()}, indent=2))

    issues = validate_config(config)
    if issues:
        print("Preflight status: NOT READY")
        for issue in issues:
            print(f" - {issue}")
        return 2

    print("Preflight status: READY")
    print("Architecture and local engine paths are valid.")
    print("No game process or training job was started.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
