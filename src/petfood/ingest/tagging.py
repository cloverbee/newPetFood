"""Rule-based tagging: species/topics for papers, ingredients/allergens for products.

Deliberately simple and transparent (keyword rules). These tags power hard *filters*
(e.g. "exclude chicken" must be exact), while semantic search handles the fuzzy part.
Tagging is conservative: "chicken fat" and "fish oil" count as chicken / fish, because a
false "safe" label is worse than an over-cautious one for an allergic pet.
"""
from __future__ import annotations

import re

# ───────────── species & topics (papers) ─────────────
_SPECIES = {
    "cat": r"\b(cats?|felines?|felis|kittens?)\b",
    "dog": r"\b(dogs?|canines?|canis|pupp(y|ies))\b",
}

TOPIC_PATTERNS: dict[str, str] = {
    "obesity": r"\b(obes\w*|overweight|weight (loss|management|gain)|body condition)\b",
    "renal": r"\b(renal|kidney|ckd|nephr\w*)\b",
    "urinary": r"\b(urinary|struvite|oxalate|urolith\w*|flutd|lower urinary)\b",
    "diabetes": r"\b(diabet\w*|insulin|glyc(a)?emic)\b",
    "food_allergy": r"\b(food allerg\w*|adverse food reaction\w*|hypersensitiv\w*|elimination diet|hydrolys\w* (protein|diet))\b",
    "gastrointestinal": r"\b(gastrointestinal|diarrh\w*|enteropath\w*|colitis|microbio\w*|digestib\w*|f(a)?eces|fecal)\b",
    "cardiac": r"\b(cardiac|cardiomyopath\w*|dcm|taurine|heart (disease|failure)|congestive heart)\b",
    "hepatic": r"\b(hepatic|hepatopath\w*|lipidosis|liver (disease|failure|function|enzymes?))\b",
    # (plain "liver"/"heart" removed: they are common pet food *ingredients*)
    "dental": r"\b(dental|periodont\w*|plaque|calculus|oral health)\b",
    "senior": r"\b(senior|geriatric|aged|ageing|aging|older (cats|dogs))\b",
    "growth": r"\b(growth|growing (kittens|puppies|cats|dogs)|gestation|lactation|weaning|reproduct\w*)\b",
    "hydration": r"\b(hydration|water intake|moisture|wet food|canned)\b",
    "raw_diet": r"\b(raw (meat|food|diet)\w*|barf|unprocessed)\b",
    "grain_free": r"\b(grain[- ]free|legume\w*|pulses?)\b",
    "plant_based": r"\b(vegan|vegetarian|plant[- ]based)\b",
    "macronutrients": r"\b(protein|amino acid\w*|fat|fatty acid\w*|omega[- ]?3|carbohydrate\w*|fib(er|re))\b",
}


def detect_species(text: str) -> list[str]:
    t = text.lower()
    return sorted(sp for sp, pat in _SPECIES.items() if re.search(pat, t))


def detect_topics(text: str) -> list[str]:
    t = text.lower()
    return sorted(topic for topic, pat in TOPIC_PATTERNS.items() if re.search(pat, t))


# ───────────── ingredients & allergens (products) ─────────────
PROTEIN_SOURCES: dict[str, list[str]] = {
    "chicken": ["chicken"],
    "turkey": ["turkey"],
    "duck": ["duck"],
    "poultry": ["poultry", "fowl", "quail", "goose"],
    "beef": ["beef", "bovine", "veal"],
    "lamb": ["lamb", "mutton", "sheep"],
    "pork": ["pork", "pig", "ham", "porcine"],
    "fish": ["fish", "fishmeal", "salmon", "tuna", "trout", "whitefish", "herring", "cod", "sardine",
             "mackerel", "pollock", "anchovy", "anchovies", "menhaden", "tilapia", "pilchard", "ocean fish",
             "krill", "seafood", "plaice", "haddock", "hake"],
    "shellfish": ["shrimp", "prawn", "crab", "mussel", "clam"],
    "venison": ["venison", "deer"],
    "rabbit": ["rabbit"],
    "egg": ["egg"],
    "insect": ["insect", "black soldier fly", "cricket", "mealworm"],
}
OTHER_ALLERGENS: dict[str, list[str]] = {
    "grain": ["wheat", "corn", "cornmeal", "maize", "rice", "barley", "oat", "sorghum", "millet", "rye",
              "cereal", "grain", "spelt"],
    "gluten": ["wheat", "gluten", "barley", "rye", "spelt"],
    "dairy": ["milk", "cheese", "whey", "lactose", "yogurt", "yoghurt", "casein", "dairy", "butter"],
    "soy": ["soy", "soya", "soybean"],
}


def _kw_regex(words: list[str]) -> re.Pattern:
    # whole words with optional plural: "egg" matches eggs, not "eggplant"; "pig" not "pigment"
    return re.compile(r"\b(" + "|".join(re.escape(w) for w in words) + r")(s|es)?\b", re.I)


_PROTEIN_RX = {k: _kw_regex(v) for k, v in PROTEIN_SOURCES.items()}
_OTHER_RX = {k: _kw_regex(v) for k, v in OTHER_ALLERGENS.items()}
# Phrases that contain a keyword but are not that ingredient
_FALSE_POSITIVES = re.compile(r"\b(grain[- ]free|gluten[- ]free|(no|without) (added )?(grains?|wheat|corn|soy|chicken|beef|fish|dairy))\b", re.I)


def split_ingredients(text: str | None) -> list[str]:
    """Split a label ingredient list on top-level commas/semicolons; normalize."""
    if not text:
        return []
    text = re.sub(r"^\s*(ingredients|composition|zusammensetzung)\s*:\s*", "", text.strip(), flags=re.I)
    parts, depth, buf = [], 0, []
    for ch in text:
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth = max(0, depth - 1)
        if ch in ",;" and depth == 0:
            parts.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    parts.append("".join(buf))
    out = []
    for p in parts:
        p = re.sub(r"\(?\s*\d+(\.\d+)?\s*%\s*\)?", "", p)   # drop percentages
        p = re.sub(r"[_*]", "", p)                          # OPFF allergen markup _chicken_
        p = re.sub(r"\s+", " ", p).strip(" .:").lower()
        if p:
            out.append(p)
    return out


def tag_ingredients(ingredients_text: str | None, extra_text: str = "") -> dict:
    """Return protein_sources, allergen_tags and grain_free for a product.

    The *name* (extra_text) is also scanned for proteins, e.g. "Salmon Recipe Kibble".
    """
    ing_text = ingredients_text or ""
    scan = _FALSE_POSITIVES.sub(" ", f"{ing_text} {extra_text}")
    ing_only = _FALSE_POSITIVES.sub(" ", ing_text)

    proteins = sorted(k for k, rx in _PROTEIN_RX.items() if rx.search(scan))
    others = sorted(k for k, rx in _OTHER_RX.items() if rx.search(ing_only))

    grain_free: bool | None
    if re.search(r"\bgrain[- ]free\b", f"{ing_text} {extra_text}", re.I):
        grain_free = True
    elif ing_text:
        grain_free = "grain" not in others
    else:
        grain_free = None  # unknown without an ingredient list

    allergens = sorted(set(proteins) | set(others))
    return {"protein_sources": proteins, "allergen_tags": allergens, "grain_free": grain_free}


def detect_life_stage(text: str) -> str | None:
    t = text.lower()
    if re.search(r"\b(kitten|puppy|junior|growth)\b", t):
        return "kitten" if "kitten" in t else "puppy" if "puppy" in t else "growth"
    if re.search(r"\b(senior|mature|7\+|11\+|ageing|aging)\b", t):
        return "senior"
    if re.search(r"\ball life stages\b", t):
        return "all"
    if re.search(r"\badult\b", t):
        return "adult"
    return None


def detect_food_form(text: str) -> str | None:
    t = text.lower()
    if re.search(r"\b(treats?|snacks?|chews?|biscuits?)\b", t):
        return "treat"
    if re.search(r"\b(wet|canned|cans?|pouch\w*|pat[eé]|gravy|jelly|loaf|chunks)\b", t):
        return "wet"
    if re.search(r"\b(dry|kibble|croquettes?|biscuit)\b", t):
        return "dry"
    if re.search(r"\b(raw|freeze[- ]dried)\b", t):
        return "raw"
    return None
