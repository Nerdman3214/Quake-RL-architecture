"""Typed configuration contracts for the Quake RL runtime."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json

@dataclass(frozen=True)
class EngineConfig:
    kind: str
    install_dir: Path
    client_executable: str
    server_executable: str
    base_game_dir: str
    host: str
    port: int

@dataclass(frozen=True)
class MatchConfig:
    mode: str
    map: str
    max_players: int
    bot_count: int
    score_limit: int
    time_limit_minutes: int
    agent_name: str

@dataclass(frozen=True)
class EnvironmentConfig:
    observation_version: str
    action_version: str
    reward_version: str
    frame_width: int
    frame_height: int
    frame_stack: int
    action_repeat: int

@dataclass(frozen=True)
class RuntimeConfig:
    headless_server: bool
    launch_client: bool
    startup_timeout_seconds: int
    shutdown_timeout_seconds: int
    log_dir: Path

@dataclass(frozen=True)
class ProjectConfig:
    project_name: str
    engine: EngineConfig
    match: MatchConfig
    environment: EnvironmentConfig
    runtime: RuntimeConfig


def load_config(path: Path) -> ProjectConfig:
    raw: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    engine = dict(raw["engine"])
    engine["install_dir"] = Path(engine["install_dir"])
    runtime = dict(raw["runtime"])
    runtime["log_dir"] = Path(runtime["log_dir"])
    return ProjectConfig(
        project_name=raw["project_name"],
        engine=EngineConfig(**engine),
        match=MatchConfig(**raw["match"]),
        environment=EnvironmentConfig(**raw["environment"]),
        runtime=RuntimeConfig(**runtime),
    )
