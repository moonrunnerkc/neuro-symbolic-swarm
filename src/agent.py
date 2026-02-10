# Author: Bradley R. Kinnard
"""Base agent class. Runs inference via ollama, scores responses
against the query embedding, gates propagation by threshold."""

from __future__ import annotations

import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
import requests

from src.config import AgentConfig
from src.embedder import cosine_similarity, embed_text

logger = logging.getLogger(__name__)


class AgentStatus(str, Enum):
    IDLE = "idle"
    ACTIVE = "active"
    ERROR = "error"
    STOPPED = "stopped"


@dataclass
class SwarmMessage:
    """a message passed between agents."""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    source: str = ""
    target: str = "all"
    content: str = ""
    context: str = ""
    query_embedding: Optional[np.ndarray] = None
    score: float = 0.0
    cycle: int = 0
    timestamp: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """serialize for logging (skip numpy)."""
        return {
            "id": self.id,
            "source": self.source,
            "target": self.target,
            "content": self.content[:200],
            "score": round(self.score, 4),
            "cycle": self.cycle,
            "timestamp": self.timestamp,
        }


class AgentError(Exception):
    """raised when an agent fails inference or processing."""
    pass


# strip chain-of-thought blocks so the synthesizer only sees the answer
_REASONING_RE = re.compile(
    r"<reasoning>.*?</reasoning>",
    re.DOTALL | re.IGNORECASE,
)


def _strip_reasoning(text: str) -> str:
    """remove <reasoning>...</reasoning> blocks, return the clean answer."""
    cleaned = _REASONING_RE.sub("", text).strip()
    # if the model put everything inside reasoning and nothing after, keep it all
    return cleaned if cleaned else text.strip()


class Agent:
    """single swarm agent wrapping an ollama model.

    call process_message() directly -- the swarm orchestrator
    handles parallelism via ThreadPoolExecutor.
    """

    def __init__(
        self,
        config: AgentConfig,
        ollama_url: str = "http://localhost:11434",
    ):
        self.config = config
        self.role = config.role
        self.model = config.model
        self.prompt = config.prompt
        self.max_tokens = config.max_tokens
        self.temperature = config.temperature
        self.threshold = config.score_threshold
        self._ollama_url = ollama_url.rstrip("/")
        self._status = AgentStatus.IDLE
        self._message_count = 0
        self._last_error: str = ""

    @property
    def status(self) -> AgentStatus:
        return self._status

    def process_message(self, msg: SwarmMessage) -> Optional[SwarmMessage]:
        """handle a single message: generate, embed, score, gate."""
        self._status = AgentStatus.ACTIVE
        try:
            # build prompt: prepend memory context if available
            llm_input = msg.content
            if msg.context:
                llm_input = f"{msg.context}\n\n---\nUSER QUERY: {msg.content}"

            llm_response = self._call_ollama(llm_input)
            if not llm_response.strip():
                self._status = AgentStatus.IDLE
                return None

            # strip chain-of-thought reasoning, keep only the final answer
            raw_response = llm_response
            clean_response = _strip_reasoning(llm_response)

            if raw_response != clean_response:
                logger.debug(
                    "agent %s: stripped %d chars of reasoning",
                    self.role, len(raw_response) - len(clean_response),
                )

            # embed and score the clean answer against query
            response_vec = embed_text(clean_response)
            score = 0.0
            if msg.query_embedding is not None:
                score = cosine_similarity(response_vec, msg.query_embedding)

            self._message_count += 1
            self._status = AgentStatus.IDLE

            logger.info(
                "agent %s | score=%.3f | threshold=%.3f | len=%d",
                self.role, score, self.threshold, len(llm_response),
            )

            # always return the message, mark whether it passed the gate
            return SwarmMessage(
                source=self.role,
                target="all",
                content=clean_response,
                query_embedding=msg.query_embedding,
                score=score,
                cycle=msg.cycle + 1,
                metadata={"parent_id": msg.id, "gated": score < self.threshold},
            )

        except Exception as exc:
            self._status = AgentStatus.ERROR
            self._last_error = str(exc)
            logger.error("agent %s failed: %s", self.role, exc)
            return None

    def _call_ollama(self, user_input: str) -> str:
        """call the local ollama api for generation."""
        payload = {
            "model": self.model,
            "prompt": user_input,
            "system": self.prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }
        try:
            resp = requests.post(
                f"{self._ollama_url}/api/generate",
                json=payload,
                timeout=120,
            )
            resp.raise_for_status()
            return resp.json().get("response", "").strip()
        except requests.RequestException as exc:
            raise AgentError(f"ollama call failed for {self.role}: {exc}") from exc

    def stop(self) -> None:
        """mark agent as stopped."""
        self._status = AgentStatus.STOPPED

    def get_stats(self) -> dict:
        """current agent stats for status reporting."""
        return {
            "role": self.role,
            "model": self.model,
            "status": self._status.value,
            "messages_processed": self._message_count,
            "threshold": self.threshold,
            "last_error": self._last_error,
        }
