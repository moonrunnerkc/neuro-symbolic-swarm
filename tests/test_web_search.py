# Author: Bradley R. Kinnard
"""Tests for web search grounding module."""

import pytest

from src.web_search import (
    build_grounding_block,
    extract_search_query,
    needs_grounding,
)


# -- needs_grounding tests --

class TestNeedsGrounding:
    """query classification: factual vs fiction."""

    def test_factual_explain(self):
        assert needs_grounding("Explain how kilonova produces gold") is True

    def test_factual_what_is(self):
        assert needs_grounding("What is r-process nucleosynthesis?") is True

    def test_factual_who_was(self):
        assert needs_grounding("Who was the first person on the moon?") is True

    def test_factual_actually_prefix(self):
        assert needs_grounding("Actually, LIGO detected GW170817, not Hubble") is True

    def test_factual_percentage(self):
        assert needs_grounding("What percentage of gold comes from neutron stars?") is True

    def test_factual_compare(self):
        assert needs_grounding("Compare TCP and UDP protocols") is True

    def test_fiction_scene(self):
        assert needs_grounding("Write the scene where the wizard enters the castle") is False

    def test_fiction_genre_override(self):
        """factual pattern but fiction genre = skip search."""
        assert needs_grounding("Explain the magic system", genre="fantasy") is False

    def test_fiction_continue_story(self):
        assert needs_grounding("Continue the story from where we left off") is False

    def test_generic_short(self):
        """short messages without factual triggers skip."""
        assert needs_grounding("hello there") is False

    def test_verify_keyword(self):
        assert needs_grounding("Can you verify this claim about vaccines?") is True

    def test_research_keyword(self):
        assert needs_grounding("What does the research say about sleep?") is True


# -- extract_search_query tests --

class TestExtractSearchQuery:

    def test_strips_prefix(self):
        result = extract_search_query("Can you explain how photosynthesis works?")
        assert "photosynthesis" in result.lower()

    def test_truncates_long_query(self):
        long_q = "Explain " + "word " * 100
        result = extract_search_query(long_q)
        assert len(result) <= 200

    def test_preserves_core(self):
        result = extract_search_query("What is the speed of light in a vacuum?")
        assert "speed of light" in result.lower()


# -- build_grounding_block tests --

class TestBuildGroundingBlock:

    def test_empty_hits(self):
        assert build_grounding_block([]) == ""

    def test_formats_results(self):
        hits = [
            {"title": "Neutron Stars", "url": "https://example.com", "content": "facts about stars"},
        ]
        block = build_grounding_block(hits)
        assert "WEB SEARCH RESULTS" in block
        assert "Neutron Stars" in block
        assert "https://example.com" in block
        assert "facts about stars" in block

    def test_multiple_results_numbered(self):
        hits = [
            {"title": "Source A", "url": "https://a.com", "content": "content a"},
            {"title": "Source B", "url": "https://b.com", "content": "content b"},
        ]
        block = build_grounding_block(hits)
        assert "[1]" in block
        assert "[2]" in block

    def test_content_truncated(self):
        hits = [
            {"title": "Long", "url": "https://x.com", "content": "x" * 600},
        ]
        block = build_grounding_block(hits)
        # content should be truncated to 300 chars in the block
        assert len(block) < 600
