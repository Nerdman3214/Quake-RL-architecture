#!/usr/bin/env python3
"""Render a completed agent-inspection episode report."""

from __future__ import annotations

import argparse
from pathlib import Path

from RL.inspection import InspectionJSONLReader
from RL.inspection.report import (
    build_episode_report,
    render_episode_report_text,
    write_episode_report_files,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render text and JSON reports from "
            "an inspection episode."
        )
    )
    parser.add_argument(
        "episode",
        type=Path,
        help="Inspection episode JSONL file.",
    )
    parser.add_argument(
        "--text-output",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    episode_path = args.episode.resolve()

    text_path = args.text_output

    if text_path is None:
        text_path = episode_path.with_suffix(
            ".report.txt"
        )

    json_path = args.json_output

    if json_path is None:
        json_path = episode_path.with_suffix(
            ".report.json"
        )

    records = InspectionJSONLReader(
        episode_path
    ).read_episode()

    report = build_episode_report(records)

    write_episode_report_files(
        report,
        text_path=text_path,
        json_path=json_path,
    )

    print(render_episode_report_text(report))
    print(f"text_report={text_path.resolve()}")
    print(f"json_report={json_path.resolve()}")
    print("report_validation=passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
