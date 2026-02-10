# Author: Bradley R. Kinnard
"""Tests for the Agent class and SwarmMessage."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.agent import Agent, AgentError, AgentStatus, SwarmMessage
from src.config import AgentConfig


@pytest.fixture
def agent_config():
    return AgentConfig(
        role="TestAgent",
        model="phi3:mini",
        prompt="you are a test agent. be concise.",
        max_tokens=64,
        temperature=0.5,
        score_threshold=0.5,
    )


class TestSwarmMessage:
    def test_default_fields(self):
        msg = SwarmMessage(content="hello")
        assert msg.source == ""
        assert msg.target == "all"
        assert msg.cycle == 0
        assert msg.score == 0.0
        assert len(msg.id) == 12

    def test_to_dict(self):
        msg = SwarmMessage(source="Critic", content="looks good", score=0.9)
        d = msg.to_dict()
        assert d["source"] == "Critic"
        assert d["score"] == 0.9
        assert "content" in d

    def test_to_dict_truncates_long_content(self):
        msg = SwarmMessage(content="x" * 300)
        d = msg.to_dict()
        assert len(d["content"]) == 200

    def test_context_field(self):
        msg = SwarmMessage(content="q", context="past data")
        assert msg.context == "past data"


class TestAgent:
    def test_init(self, agent_config):
        agent = Agent(config=agent_config)
        assert agent.role == "TestAgent"
        assert agent.status == AgentStatus.IDLE

    def test_get_stats(self, agent_config):
        agent = Agent(config=agent_config)
        stats = agent.get_stats()
        assert stats["role"] == "TestAgent"
        assert stats["status"] == "idle"
        assert stats["messages_processed"] == 0

    @patch("src.agent.requests.post")
    @patch("src.agent.embed_text")
    def test_process_message_high_score(
        self, mock_embed, mock_post, agent_config,
    ):
        """agent should return result when score exceeds threshold."""
        agent = Agent(config=agent_config)

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"response": "concise answer"}
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        # mock embedding that yields high similarity
        query_vec = np.ones(384, dtype=np.float32) / np.sqrt(384)
        mock_embed.return_value = query_vec

        msg = SwarmMessage(
            source="user",
            content="test query",
            query_embedding=query_vec,
        )
        result = agent.process_message(msg)
        assert result is not None
        assert result.source == "TestAgent"
        assert result.content == "concise answer"

    @patch("src.agent.requests.post")
    @patch("src.agent.embed_text")
    def test_process_message_low_score(
        self, mock_embed, mock_post, agent_config,
    ):
        """agent marks low-score messages as gated."""
        agent = Agent(config=agent_config)

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"response": "bad answer"}
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        # embeddings that yield low similarity
        query_vec = np.zeros(384, dtype=np.float32)
        query_vec[0] = 1.0
        resp_vec = np.zeros(384, dtype=np.float32)
        resp_vec[1] = 1.0
        mock_embed.return_value = resp_vec

        msg = SwarmMessage(
            source="user",
            content="test query",
            query_embedding=query_vec,
        )
        result = agent.process_message(msg)
        # agent returns the message but flags it as gated
        assert result is not None
        assert result.metadata.get("gated") is True

    @patch("src.agent.requests.post")
    def test_process_message_empty_response(
        self, mock_post, agent_config,
    ):
        """empty LLM output should return None."""
        agent = Agent(config=agent_config)

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"response": ""}
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        msg = SwarmMessage(source="user", content="test")
        result = agent.process_message(msg)
        assert result is None

    @patch("src.agent.requests.post", side_effect=Exception("connection refused"))
    def test_ollama_failure_returns_none(self, mock_post, agent_config):
        """network failure should return None and set error status."""
        agent = Agent(config=agent_config)

        msg = SwarmMessage(source="user", content="test")
        result = agent.process_message(msg)
        assert result is None
        assert agent.status == AgentStatus.ERROR

    def test_stop(self, agent_config):
        agent = Agent(config=agent_config)
        agent.stop()
        assert agent.status == AgentStatus.STOPPED

    @patch("src.agent.requests.post")
    @patch("src.agent.embed_text")
    def test_context_passed_to_llm(self, mock_embed, mock_post, agent_config):
        """when context is set, agent should prepend it to the LLM input."""
        agent = Agent(config=agent_config)

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"response": "answer"}
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        query_vec = np.ones(384, dtype=np.float32) / np.sqrt(384)
        mock_embed.return_value = query_vec

        msg = SwarmMessage(
            source="user", content="question",
            context="RELEVANT PAST CONTEXT: some old data",
            query_embedding=query_vec,
        )
        agent.process_message(msg)

        # verify the prompt sent to ollama includes the context
        call_args = mock_post.call_args
        payload = call_args.kwargs.get("json") or call_args[1].get("json")
        assert "RELEVANT PAST CONTEXT" in payload["prompt"]
