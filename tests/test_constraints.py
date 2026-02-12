# Author: Bradley R. Kinnard
"""Tests for the generalized symbolic constraints module."""

import pytest

from src.constraints import (
    AGE_PATTERN,
    ERA_BLOCKLISTS,
    GENRE_ERA_MAP,
    ROLE_PATTERN,
    ROMANTIC_TERMS,
    WORLD_ANCHOR_KEYS,
    extract_words,
    find_anachronisms,
    find_fact_contradictions,
    get_blocklist,
    validate_extracted_facts,
)


# -- extract_words --

class TestExtractWords:
    def test_basic(self):
        words = extract_words("Hello world")
        assert "hello" in words
        assert "world" in words

    def test_punctuation_stripped(self):
        words = extract_words("Hello, world! (yes)")
        assert "hello" in words
        assert "world" in words
        assert "yes" in words

    def test_hyphenated_expansion(self):
        words = extract_words("laser-precise weapon")
        assert "laser" in words
        assert "precise" in words
        assert "laser-precise" in words


# -- get_blocklist --

class TestGetBlocklist:
    def test_medieval_era(self):
        bl = get_blocklist({"era": "medieval"})
        assert "smartphone" in bl
        assert "truck" in bl
        assert "gun" in bl

    def test_futuristic_era(self):
        bl = get_blocklist({"era": "futuristic"})
        assert "parchment" in bl
        assert "catapult" in bl

    def test_genre_fallback_fantasy(self):
        bl = get_blocklist({"genre": "fantasy"})
        assert "smartphone" in bl  # fantasy -> medieval

    def test_genre_fallback_scifi(self):
        bl = get_blocklist({"genre": "sci-fi"})
        assert "catapult" in bl  # sci-fi -> futuristic

    def test_modern_era(self):
        bl = get_blocklist({"era": "modern"})
        assert "trebuchet" in bl
        assert "smartphone" not in bl

    def test_unknown_era(self):
        bl = get_blocklist({"era": "unknown"})
        assert len(bl) == 0

    def test_thriller_genre(self):
        bl = get_blocklist({"genre": "thriller"})
        assert "trebuchet" in bl  # thriller -> modern


# -- find_anachronisms --

class TestFindAnachronisms:
    def test_medieval_smartphone(self):
        bl = ERA_BLOCKLISTS["medieval"]
        found = find_anachronisms("He pulled out his smartphone", bl)
        assert "smartphone" in found

    def test_medieval_gun(self):
        bl = ERA_BLOCKLISTS["medieval"]
        found = find_anachronisms("She fired her gun at the dragon", bl)
        assert "gun" in found

    def test_medieval_clean(self):
        bl = ERA_BLOCKLISTS["medieval"]
        found = find_anachronisms("He drew his sword and charged", bl)
        assert len(found) == 0

    def test_futuristic_castle(self):
        bl = ERA_BLOCKLISTS["futuristic"]
        found = find_anachronisms("They stormed the castle gates", bl)
        assert "castle" in found

    def test_empty_blocklist(self):
        found = find_anachronisms("anything goes", set())
        assert len(found) == 0

    def test_multiple_hits(self):
        bl = ERA_BLOCKLISTS["medieval"]
        found = find_anachronisms("He drove his truck to buy a laptop and a gun", bl)
        assert "truck" in found
        assert "laptop" in found
        assert "gun" in found


# -- find_fact_contradictions --

class TestFindFactContradictions:

    # age checks
    def test_age_mismatch_years_old(self):
        facts = {"protagonist_age": "42"}
        c = find_fact_contradictions("Aldric, who is 25 years old", facts)
        assert any("42" in x and "25" in x for x in c)

    def test_age_mismatch_year_old(self):
        facts = {"companion_age": "30"}
        c = find_fact_contradictions("the 19-year-old girl", facts)
        assert any("30" in x and "19" in x for x in c)

    def test_age_match_no_contradiction(self):
        facts = {"protagonist_age": "42"}
        c = find_fact_contradictions("the 42 year old monk", facts)
        assert len(c) == 0

    # role checks
    def test_role_contradiction_blacksmith(self):
        facts = {"protagonist_role": "monk and herbalist"}
        c = find_fact_contradictions("he was a blacksmith by trade", facts)
        assert any("blacksmith" in x for x in c)

    def test_role_match_no_contradiction(self):
        facts = {"protagonist_role": "monk and herbalist"}
        c = find_fact_contradictions("he was a monk in the order", facts)
        assert len(c) == 0

    def test_role_became_pattern(self):
        facts = {"protagonist_role": "engineer"}
        c = find_fact_contradictions("she became a lawyer in her twenties", facts)
        assert any("lawyer" in x for x in c)

    def test_role_works_as_pattern(self):
        facts = {"companion_role": "visiting nun"}
        c = find_fact_contradictions("she works as a mercenary for hire", facts)
        assert any("mercenary" in x for x in c)

    # relationship checks
    def test_romantic_contradiction(self):
        facts = {"companion_relationship": "friend"}
        c = find_fact_contradictions("he kissed his wife Margaux", facts)
        assert any("wife" in x for x in c)

    def test_romantic_no_contradiction(self):
        facts = {"companion_relationship": "wife"}
        c = find_fact_contradictions("he spoke to his wife", facts)
        assert len(c) == 0

    # timeline checks
    def test_timeline_contradiction(self):
        facts = {"timeline": "1347"}
        c = find_fact_contradictions("it was the year 2024", facts)
        assert any("1347" in x and "2024" in x for x in c)

    def test_timeline_match(self):
        facts = {"timeline": "1347"}
        c = find_fact_contradictions("in the year 1347", facts)
        assert len(c) == 0

    # identity/project name checks
    def test_project_name_contradiction(self):
        facts = {"project_name": "Meridian"}
        c = find_fact_contradictions("the project is called Horizon", facts)
        assert any("Meridian" in x and "horizon" in x.lower() for x in c)

    def test_project_name_match(self):
        facts = {"project_name": "Meridian"}
        c = find_fact_contradictions("the project is called Meridian", facts)
        assert len(c) == 0

    def test_language_contradiction(self):
        facts = {"programming_language": "Rust"}
        c = find_fact_contradictions("it's written in Java", facts)
        assert any("Rust" in x and "java" in x.lower() for x in c)

    def test_database_contradiction(self):
        facts = {"database": "CockroachDB"}
        c = find_fact_contradictions("we're using MySQL as the database now", facts)
        assert any("CockroachDB" in x and "mysql" in x.lower() for x in c)

    def test_database_no_false_positive_on_broker(self):
        """'using Redis as a message broker' should NOT trigger database contradiction."""
        facts = {"database": "CockroachDB"}
        c = find_fact_contradictions("We're using Redis 7.2 as a message broker between services", facts)
        assert len(c) == 0

    # empty inputs
    def test_empty_query(self):
        c = find_fact_contradictions("", {"protagonist_age": "42"})
        assert len(c) == 0

    def test_empty_facts(self):
        c = find_fact_contradictions("some query", {})
        assert len(c) == 0

    def test_no_matching_patterns(self):
        facts = {"setting": "Normandy"}
        c = find_fact_contradictions("hello there", facts)
        assert len(c) == 0


# -- coverage checks --

class TestBlocklistCoverage:
    """ensure important words are in the blocklists."""

    def test_medieval_has_firearms(self):
        bl = ERA_BLOCKLISTS["medieval"]
        firearms = {"gun", "pistol", "rifle", "revolver", "shotgun", "bullet"}
        assert firearms.issubset(bl)

    def test_medieval_has_electronics(self):
        bl = ERA_BLOCKLISTS["medieval"]
        electronics = {"computer", "laptop", "smartphone", "wifi", "internet"}
        assert electronics.issubset(bl)

    def test_medieval_has_vehicles(self):
        bl = ERA_BLOCKLISTS["medieval"]
        vehicles = {"car", "truck", "airplane", "motorcycle", "helicopter"}
        assert vehicles.issubset(bl)

    def test_medieval_has_media(self):
        bl = ERA_BLOCKLISTS["medieval"]
        media = {"television", "radio", "camera"}
        assert media.issubset(bl)

    def test_medieval_has_medicine(self):
        bl = ERA_BLOCKLISTS["medieval"]
        medicine = {"microscope", "antibiotic", "vaccine", "x-ray"}
        assert medicine.issubset(bl)

    def test_futuristic_has_medieval_items(self):
        bl = ERA_BLOCKLISTS["futuristic"]
        medieval = {"parchment", "quill", "castle", "drawbridge", "sword"}
        assert medieval.issubset(bl)

    def test_world_anchor_keys_include_region(self):
        assert "region" in WORLD_ANCHOR_KEYS

    def test_genre_era_map_has_thriller(self):
        assert "thriller" in GENRE_ERA_MAP


# -- validate_extracted_facts --

class TestValidateExtractedFacts:

    def test_direct_substring_match(self):
        """values that appear verbatim in user input should pass."""
        user = "The genre is sci-fi. The setting is Mars."
        extracted = {"genre": "sci-fi", "setting": "Mars"}
        result = validate_extracted_facts(extracted, user)
        assert result == {"genre": "sci-fi", "setting": "Mars"}

    def test_rejects_hallucinated_era(self):
        """extractor says medieval but user said sci-fi; reject."""
        user = "The year is 2847. The genre is sci-fi. The setting is Europa."
        extracted = {"era": "medieval", "setting": "castle near a river valley"}
        result = validate_extracted_facts(extracted, user)
        assert "era" not in result
        assert "setting" not in result

    def test_accepts_valid_era_inference(self):
        """extractor says futuristic, user said sci-fi; accept via inference."""
        user = "The genre is sci-fi and the year is 2847."
        extracted = {"era": "futuristic", "genre": "sci-fi", "timeline": "2847"}
        result = validate_extracted_facts(extracted, user)
        assert result["era"] == "futuristic"
        assert result["genre"] == "sci-fi"
        assert result["timeline"] == "2847"

    def test_accepts_medieval_era_from_fantasy(self):
        """extractor says medieval, user said fantasy; accept via inference."""
        user = "A fantasy story set in the Shattered Reach."
        extracted = {"era": "medieval", "genre": "fantasy", "setting": "Shattered Reach"}
        result = validate_extracted_facts(extracted, user)
        assert result["era"] == "medieval"

    def test_word_overlap_match(self):
        """values with shared significant words should pass."""
        user = "The protagonist is a 38-year-old xenobiologist named Reva Chen."
        extracted = {"protagonist_name": "Reva Chen", "protagonist_age": "38"}
        result = validate_extracted_facts(extracted, user)
        assert "protagonist_name" in result
        assert "protagonist_age" in result

    def test_rejects_completely_ungrounded(self):
        """values with zero overlap to user input should be rejected."""
        user = "Set in a spaceship orbiting Jupiter in 3001."
        extracted = {"setting": "medieval castle", "era": "medieval"}
        result = validate_extracted_facts(extracted, user)
        assert len(result) == 0

    def test_preserves_valid_rejects_invalid(self):
        """mixed bag: some grounded, some not."""
        user = "The year is 2847. The setting is a research station. The genre is sci-fi."
        extracted = {
            "timeline": "2847",
            "genre": "sci-fi",
            "setting": "castle near a river valley in medieval Europe",
            "era": "medieval",
        }
        result = validate_extracted_facts(extracted, user)
        assert result["timeline"] == "2847"
        assert result["genre"] == "sci-fi"
        assert "setting" not in result  # hallucinated
        assert "era" not in result  # no medieval basis in input

    def test_empty_input(self):
        result = validate_extracted_facts({}, "some text")
        assert result == {}

    def test_empty_user(self):
        result = validate_extracted_facts({"key": "val"}, "")
        assert result == {"key": "val"}
