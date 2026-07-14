"""Preflight checks that do not launch or modify the game."""
from pathlib import Path
from RL.config.schema import ProjectConfig


def validate_config(config: ProjectConfig) -> list[str]:
    issues: list[str] = []
    install_dir = config.engine.install_dir
    if not install_dir.exists():
        issues.append(f"Engine install directory does not exist: {install_dir}")
    client = install_dir / config.engine.client_executable
    server = install_dir / config.engine.server_executable
    if not client.exists():
        issues.append(f"Client executable not found: {client}")
    if not server.exists():
        issues.append(f"Dedicated-server executable not found: {server}")
    if not (1 <= config.engine.port <= 65535):
        issues.append(f"Port is outside the valid range: {config.engine.port}")
    if config.match.bot_count >= config.match.max_players:
        issues.append("bot_count must be lower than max_players")
    return issues


def describe_paths(config: ProjectConfig) -> dict[str, Path]:
    base = config.engine.install_dir
    return {
        "install_dir": base,
        "client": base / config.engine.client_executable,
        "server": base / config.engine.server_executable,
        "game_data": base / config.engine.base_game_dir,
    }
