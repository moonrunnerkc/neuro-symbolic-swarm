# Author: Bradley R. Kinnard
"""Symbolic constraints for the neuro-symbolic validation pipeline.

All era blocklists, genre mappings, and fact-checking patterns live here.
Extend these data structures to add coverage for new eras or domains
without touching the orchestrator."""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# -- era blocklists --
# words that don't belong in a given technology era.
# key = era name (matched via substring), value = set of forbidden terms.
# these are checked against both user input AND agent drafts.

ERA_BLOCKLISTS: dict[str, set[str]] = {
    "medieval": {
        # transportation
        "truck", "pickup", "highway", "car", "automobile", "van", "bus",
        "airplane", "aeroplane", "motorcycle", "bicycle", "subway", "taxi",
        "uber", "train", "railroad", "freeway", "asphalt", "helicopter",
        "jet", "rocket",
        # electronics / computing
        "phone", "telephone", "smartphone", "computer", "laptop", "tablet",
        "internet", "wifi", "bluetooth", "email", "website", "app",
        "software", "hardware", "microchip", "processor", "server",
        "digital", "android", "download", "upload", "streaming",
        # weapons / modern tech
        "gun", "pistol", "rifle", "revolver", "shotgun", "bullet",
        "grenade", "bomb", "missile", "tank", "machine gun",
        "laser", "radar", "sonar", "drone", "satellite",
        # energy / industrial
        "electric", "electricity", "battery", "generator", "engine",
        "gasoline", "diesel", "petroleum", "nuclear", "reactor",
        "turbine", "motor", "factory",
        # brands / modern concepts
        "walmart", "amazon", "tesla", "google", "uber",
        "supermarket", "mall", "skyscraper",
        # media / entertainment
        "television", "tv", "radio", "camera", "photograph", "video",
        "movie", "film", "podcast", "social media",
        # science / medicine (post-medieval)
        "microscope", "telescope", "x-ray", "antibiotic", "vaccine",
        "surgery", "anesthesia", "thermometer",
        # lighting / materials
        "neon", "plastic", "concrete", "steel",
        "hologram", "plasma", "robot",
    },
    "futuristic": {
        # pre-industrial transport
        "horse-drawn", "horse drawn", "carriage", "wagon", "oxcart",
        "stagecoach", "chariot",
        # medieval specifics
        "candle", "candlelight", "torch", "lantern",
        "parchment", "quill", "inkwell", "scroll",
        "feudal", "peasant", "serf", "vassal", "lord",
        "castle", "drawbridge", "moat", "catapult", "trebuchet",
        "sword", "crossbow", "longbow", "shield", "armor", "armour",
        "blacksmith", "farrier", "thatched",
    },
    "modern": {
        # purely medieval/ancient terms that feel wrong in modern context
        "trebuchet", "catapult", "drawbridge", "moat",
        "feudal", "serf", "vassal",
    },
}


# -- genre -> era inference --
# when the extractor captures genre but misses era, infer technology level.
GENRE_ERA_MAP: dict[str, str] = {
    "fantasy": "medieval",
    "medieval": "medieval",
    "historical": "medieval",
    "sci-fi": "futuristic",
    "science fiction": "futuristic",
    "cyberpunk": "futuristic",
    "space opera": "futuristic",
    "steampunk": "medieval",   # close enough — no modern electronics
    "modern": "modern",
    "contemporary": "modern",
    "thriller": "modern",
    "mystery": "modern",
}


# -- world anchor keys --
# predicates that define the thread's "world". changes to these are
# high-severity; they're used for blocklist selection, setting enforcement.
WORLD_ANCHOR_KEYS: set[str] = {
    "setting", "genre", "era", "timeline",
    "planet", "city", "country", "region",
}


# -- fact contradiction patterns --
# generic patterns to detect when user input contradicts locked facts.
# these work across any predicate, not just character-specific ones.

# pattern: "<number> years old" or "<number>-year-old"
AGE_PATTERN = re.compile(
    r"\b(\d{1,3})\s*[-–]?\s*(?:years?\s*old|year[\s-]*old)\b",
    re.IGNORECASE,
)

# pattern: "is a <word>", "was a <word>", "always been a <word>",
#          "works as a <word>", "trained as a <word>", "became a <word>"
# captures the role noun(s) up to a natural boundary (preposition, conjunction,
# punctuation, or end of string)
ROLE_PATTERN = re.compile(
    r"(?:always been|has been|is|was|works as|trained as|became|"
    r"served as|acts as|employed as|known as)"
    r"\s+(?:a |an )?(\w[\w\s]{0,30}?)"
    r"(?:[.,;!?]|\s+(?:who|and|but|in|at|for|not|by|from|to|with|since|during)\b|$)",
    re.IGNORECASE,
)

# pattern: "his/her wife/husband/lover/etc" — romantic relationship claims
ROMANTIC_TERMS: set[str] = {
    "wife", "husband", "spouse", "lover", "girlfriend", "boyfriend",
    "fiancee", "fiance", "fiancé", "fiancée", "partner", "betrothed",
}


def extract_words(text: str) -> set[str]:
    """split text into a clean word set, expanding hyphenated compounds."""
    raw = {w.strip(".,!?;:\"'()[]{}") for w in text.lower().split()}
    expanded: set[str] = set()
    for w in raw:
        expanded.add(w)
        if "-" in w:
            expanded.update(w.split("-"))
    return expanded


def get_blocklist(anchors: dict[str, str]) -> set[str]:
    """resolve the era blocklist from world anchors (era or genre)."""
    era = anchors.get("era", "")
    for era_key, words in ERA_BLOCKLISTS.items():
        if era_key in era:
            return words

    # fallback: infer from genre
    genre = anchors.get("genre", "")
    for genre_key, mapped_era in GENRE_ERA_MAP.items():
        if genre_key in genre:
            return ERA_BLOCKLISTS.get(mapped_era, set())

    return set()


def find_anachronisms(text: str, blocklist: set[str]) -> set[str]:
    """find blocklist words present in the given text."""
    if not blocklist:
        return set()
    words = extract_words(text)
    return blocklist & words


def find_fact_contradictions(
    query: str,
    facts: dict[str, str],
) -> list[str]:
    """detect when user input contradicts ANY established fact.

    Returns a list of human-readable contradiction descriptions.
    Works generically across all predicate types.
    """
    if not query or not facts:
        return []

    query_lower = query.lower()
    contradictions: list[str] = []

    for pred, val in facts.items():
        val_lower = val.lower()

        # -- numeric facts (ages, counts, years) --
        if val_lower.isdigit():
            # find all ages/numbers in the query
            if "age" in pred:
                for match in AGE_PATTERN.finditer(query_lower):
                    found = match.group(1)
                    if found != val_lower:
                        contradictions.append(
                            f"{pred} is established as {val}, not {found}"
                        )
            # also check for bare number near the predicate subject
            # e.g. "timeline" = "1347" vs user says "in 2024"
            if "timeline" in pred:
                year_matches = re.findall(r"\b(\d{4})\b", query_lower)
                for found_year in year_matches:
                    if found_year != val_lower:
                        contradictions.append(
                            f"{pred} is established as {val}, not {found_year}"
                        )

        # -- role/occupation facts --
        if any(k in pred for k in ("role", "occupation", "job", "class")):
            for match in ROLE_PATTERN.finditer(query_lower):
                found_role = match.group(1).strip()
                # only flag if the claimed role is truly different
                if (
                    found_role not in val_lower
                    and val_lower not in found_role
                    and len(found_role) > 2
                ):
                    contradictions.append(
                        f"{pred} is '{val}', not '{found_role}'"
                    )

        # -- relationship facts --
        if any(k in pred for k in ("relationship", "companion")):
            for term in ROMANTIC_TERMS:
                if term in query_lower and term not in val_lower:
                    contradictions.append(
                        f"{pred} is '{val}', user claims '{term}'"
                    )

    return contradictions


# -- extraction validation --
# synonyms and inference rules for world-building predicates.
# if the user says "sci-fi" and the extractor infers "futuristic" for era,
# that's a valid inference even though "futuristic" doesn't appear verbatim.

VALID_ERA_INFERENCES: dict[str, set[str]] = {
    # genre keywords that validly imply an era
    "medieval": {"fantasy", "medieval", "historical", "middle ages", "dark ages"},
    "futuristic": {"sci-fi", "science fiction", "cyberpunk", "space opera",
                   "futuristic", "space", "deep-space", "starship", "2847"},
    "modern": {"modern", "contemporary", "thriller", "mystery", "present day",
               "current day", "detective", "noir"},
}


def validate_extracted_facts(
    extracted: dict[str, str],
    user_input: str,
) -> dict[str, str]:
    """filter extraction output to only facts grounded in user input.

    Small models hallucinate facts the user never stated. This checks
    that each extracted value has SOME basis in the actual user text.
    Returns only the validated subset.
    """
    if not extracted or not user_input:
        return extracted or {}

    user_lower = user_input.lower()
    user_words = extract_words(user_input)
    validated: dict[str, str] = {}

    for key, value in extracted.items():
        str_val = str(value).strip()
        val_lower = str_val.lower()

        # skip empty
        if not str_val:
            continue

        # RULE 1: direct substring match -- value appears in user text
        if val_lower in user_lower:
            validated[key] = str_val
            continue

        # RULE 2: word overlap -- at least one significant word from the
        # value appears in the user input
        val_words = {w for w in extract_words(str_val) if len(w) > 2}
        if val_words & user_words:
            validated[key] = str_val
            continue

        # RULE 3: era inference -- if user said "sci-fi" and extractor
        # produced era="futuristic", that's a valid inference
        if key == "era":
            era_key = val_lower
            valid_sources = VALID_ERA_INFERENCES.get(era_key, set())
            if any(src in user_lower for src in valid_sources):
                validated[key] = str_val
                continue

        # RULE 4: numeric values -- if value is a number and it appears
        # in the user text, accept it
        if str_val.isdigit() and str_val in user_lower:
            validated[key] = str_val
            continue

        # fact not grounded -- reject
        logger.debug(
            "extraction rejected: %s='%s' not grounded in user input",
            key, str_val,
        )

    return validated
