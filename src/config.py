# Author: Bradley R. Kinnard
"""Pydantic models for loading agent and application configs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field, field_validator


class AgentConfig(BaseModel):
    """single agent role definition, loaded from yaml."""

    role: str
    model: str
    prompt: str
    tools: list[str] = Field(default_factory=list)
    max_tokens: int = 256
    temperature: float = 0.7
    score_threshold: float = 0.8

    @field_validator("role")
    @classmethod
    def role_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("agent role cannot be empty")
        return v.strip()

    @field_validator("temperature")
    @classmethod
    def temp_in_range(cls, v: float) -> float:
        if not 0.0 <= v <= 2.0:
            raise ValueError(f"temperature must be 0.0-2.0, got {v}")
        return v


class AppConfig(BaseModel):
    """top-level application configuration from data/config.json."""

    theme: str = "dark"
    agent_count: int = 5
    score_threshold: float = 0.8
    max_cycles: int = 10
    max_tokens: int = 256
    temperature: float = 0.7
    default_model: str = "phi3:mini"
    log_level: str = "INFO"
    context_window: int = 5
    cross_thread_memory: bool = False
    auto_save_interval: int = 60


def load_agent_config(path: Path) -> AgentConfig:
    """load a single agent config from a yaml file."""
    if not path.exists():
        raise FileNotFoundError(f"agent config not found: {path}")
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return AgentConfig(**raw)


def load_all_agents(directory: Path) -> list[AgentConfig]:
    """load every .yaml config from the agents directory."""
    if not directory.is_dir():
        raise FileNotFoundError(f"agents directory not found: {directory}")
    configs = []
    for p in sorted(directory.glob("*.yaml")):
        configs.append(load_agent_config(p))
    if not configs:
        raise ValueError(f"no agent configs found in {directory}")
    return configs


def load_app_config(path: Path) -> AppConfig:
    """load the main app config from json."""
    if not path.exists():
        raise FileNotFoundError(f"app config not found: {path}")
    with open(path, "r") as f:
        raw = json.load(f)
    return AppConfig(**raw)


def save_app_config(config: AppConfig, path: Path) -> None:
    """persist app config back to json."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(config.model_dump(), f, indent=2)


def save_agent_config(config: AgentConfig, directory: Path) -> Path:
    """write an agent config to a yaml file. returns the file path."""
    directory.mkdir(parents=True, exist_ok=True)
    filename = config.role.lower().replace(" ", "-") + ".yaml"
    path = directory / filename
    data = config.model_dump()
    # prepend a comment for humans
    header = f"# Author: Bradley R. Kinnard\n# {config.role.lower()} agent\n\n"
    with open(path, "w") as f:
        f.write(header)
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    return path
