# Author: Bradley R. Kinnard
"""Tests for the project file structure and agent YAML definitions."""

from pathlib import Path

import pytest
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestFileStructure:
    """verify the project tree matches the spec in copilot-instructions.md."""

    def test_src_package_exists(self):
        assert (PROJECT_ROOT / "src" / "__init__.py").exists()

    def test_core_modules_exist(self):
        modules = ["main.py", "swarm.py", "agent.py", "memory.py", "embedder.py", "config.py", "state_manager.py"]
        for mod in modules:
            assert (PROJECT_ROOT / "src" / mod).exists(), f"missing src/{mod}"

    def test_world_state_schema(self):
        """world_state.json is created at runtime -- if present, validate schema."""
        import json
        path = PROJECT_ROOT / "data" / "world_state.json"
        if not path.exists():
            # file is gitignored and regenerated on first run; skip on clean checkout
            return
        raw = json.loads(path.read_text())
        assert raw.get("version") == 1
        assert "threads" in raw

    def test_ui_package_exists(self):
        assert (PROJECT_ROOT / "src" / "ui" / "__init__.py").exists()

    def test_ui_modules_exist(self):
        ui_files = ["controller.py", "theme.py"]
        for f in ui_files:
            assert (PROJECT_ROOT / "src" / "ui" / f).exists(), f"missing src/ui/{f}"

    def test_widget_modules_exist(self):
        widgets = ["chat_area.py", "sidebar_left.py", "sidebar_right.py", "agent_card.py"]
        for w in widgets:
            assert (PROJECT_ROOT / "src" / "ui" / "widgets" / w).exists(), f"missing {w}"

    def test_agent_yamls_exist(self):
        expected = ["parser.yaml", "retriever.yaml", "synthesizer.yaml", "critic.yaml", "innovator.yaml", "reasoner.yaml", "fact_extractor.yaml"]
        for f in expected:
            assert (PROJECT_ROOT / "agents" / f).exists(), f"missing agents/{f}"

    def test_data_directory(self):
        assert (PROJECT_ROOT / "data" / "config.json").exists()
        assert (PROJECT_ROOT / "data" / "threads").is_dir()

    def test_run_script(self):
        assert (PROJECT_ROOT / "run.py").exists()

    def test_requirements_txt(self):
        assert (PROJECT_ROOT / "requirements.txt").exists()

    def test_readme(self):
        assert (PROJECT_ROOT / "README.md").exists()


class TestAgentYAMLIntegrity:
    """load each agent yaml and verify required fields are present."""

    @pytest.fixture(params=["parser", "retriever", "synthesizer", "critic", "innovator", "reasoner", "fact_extractor"])
    def agent_data(self, request):
        path = PROJECT_ROOT / "agents" / f"{request.param}.yaml"
        with open(path) as f:
            return yaml.safe_load(f)

    def test_has_role(self, agent_data):
        assert "role" in agent_data
        assert len(agent_data["role"].strip()) > 0

    def test_has_model(self, agent_data):
        assert "model" in agent_data

    def test_has_prompt(self, agent_data):
        assert "prompt" in agent_data
        assert len(agent_data["prompt"].strip()) > 10

    def test_has_max_tokens(self, agent_data):
        assert "max_tokens" in agent_data
        assert agent_data["max_tokens"] <= 4096

    def test_has_score_threshold(self, agent_data):
        assert "score_threshold" in agent_data
        assert 0.0 <= agent_data["score_threshold"] <= 1.0


class TestFactExtractorYAML:
    """verify the fact-extractor agent has extraction-specific config."""

    def test_low_temperature(self):
        path = PROJECT_ROOT / "agents" / "fact_extractor.yaml"
        with open(path) as f:
            data = yaml.safe_load(f)
        # extraction needs deterministic output
        assert data["temperature"] <= 0.2

    def test_zero_score_threshold(self):
        path = PROJECT_ROOT / "agents" / "fact_extractor.yaml"
        with open(path) as f:
            data = yaml.safe_load(f)
        # extractor output isn't scored against query semantics
        assert data["score_threshold"] == 0.0

    def test_role_name(self):
        path = PROJECT_ROOT / "agents" / "fact_extractor.yaml"
        with open(path) as f:
            data = yaml.safe_load(f)
        assert data["role"] == "Fact-Extractor"
