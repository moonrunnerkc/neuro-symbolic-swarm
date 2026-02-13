# Author: Bradley R. Kinnard
"""Tests for config loading and validation."""

import json
import tempfile
from pathlib import Path

import pytest
import yaml

from src.config import (
    AgentConfig,
    AppConfig,
    load_agent_config,
    load_all_agents,
    load_app_config,
    save_app_config,
)


# -- AgentConfig validation --

class TestAgentConfig:
    def test_valid_config(self):
        cfg = AgentConfig(
            role="Parser",
            model="phi3:mini",
            prompt="test prompt",
        )
        assert cfg.role == "Parser"
        assert cfg.max_tokens == 256
        assert cfg.temperature == 0.7

    def test_empty_role_rejected(self):
        with pytest.raises(ValueError, match="empty"):
            AgentConfig(role="  ", model="m", prompt="p")

    def test_temperature_out_of_range(self):
        with pytest.raises(ValueError, match="temperature"):
            AgentConfig(role="X", model="m", prompt="p", temperature=3.0)

    def test_temperature_lower_bound(self):
        cfg = AgentConfig(role="X", model="m", prompt="p", temperature=0.0)
        assert cfg.temperature == 0.0

    def test_defaults(self):
        cfg = AgentConfig(role="X", model="m", prompt="p")
        assert cfg.tools == []
        assert cfg.score_threshold == 0.8


# -- AppConfig --

class TestAppConfig:
    def test_defaults(self):
        cfg = AppConfig()
        assert cfg.theme == "dark"
        assert cfg.agent_count == 5

    def test_custom_values(self):
        cfg = AppConfig(agent_count=8, log_level="DEBUG")
        assert cfg.agent_count == 8
        assert cfg.log_level == "DEBUG"


# -- file loading --

class TestLoadAgentConfig:
    def test_load_valid_yaml(self, tmp_path):
        data = {
            "role": "Critic",
            "model": "phi3:mini",
            "prompt": "evaluate things",
            "tools": ["embedding_scoring"],
        }
        f = tmp_path / "critic.yaml"
        f.write_text(yaml.dump(data))
        cfg = load_agent_config(f)
        assert cfg.role == "Critic"
        assert "embedding_scoring" in cfg.tools

    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_agent_config(tmp_path / "nope.yaml")

    def test_invalid_yaml_content(self, tmp_path):
        f = tmp_path / "bad.yaml"
        f.write_text(yaml.dump({"model": "m", "prompt": "p"}))
        with pytest.raises(Exception):
            load_agent_config(f)


class TestLoadAllAgents:
    def test_loads_multiple(self, tmp_path):
        for name in ["a", "b"]:
            data = {"role": name.upper(), "model": "m", "prompt": "p"}
            (tmp_path / f"{name}.yaml").write_text(yaml.dump(data))
        configs = load_all_agents(tmp_path)
        assert len(configs) == 2

    def test_empty_directory(self, tmp_path):
        with pytest.raises(ValueError, match="no agent configs"):
            load_all_agents(tmp_path)

    def test_missing_directory(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_all_agents(tmp_path / "fake")


class TestAppConfigIO:
    def test_save_and_load(self, tmp_path):
        cfg = AppConfig(agent_count=3, theme="light")
        path = tmp_path / "cfg.json"
        save_app_config(cfg, path)
        loaded = load_app_config(path)
        assert loaded.agent_count == 3
        assert loaded.theme == "light"

    def test_load_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_app_config(tmp_path / "nope.json")
