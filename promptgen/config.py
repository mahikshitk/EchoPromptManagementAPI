"""Tunable constants. Code lists live here so v2 audiences/topics only need edits here."""

AUDIENCE = "patient"

# Topic code -> share of n. Order here is the order topics are generated in.
TOPIC_WEIGHTS = {"safety": 0.40, "efficacy": 0.30, "access_cost": 0.30}

INTENTS = [
    "informational",
    "comparison",
    "side_effects",
    "suitability",
    "cost_access",
    "experience",
    "alternatives",
]
DEFAULT_INTENT = "informational"

# Generation
BATCH_SIZE = 12          # max prompts accepted per LLM call
OVERGENERATE = 1.3       # ask the LLM for this many x batch quota
TOPUP_CALLS = 2          # extra calls per short topic
DEDUP_THRESHOLD = 0.85
MIN_LEN, MAX_LEN = 15, 200
FEWSHOT_TOPIC, FEWSHOT_GENERAL = 6, 3

# LLM
DEFAULT_MODEL = "deepseek/deepseek-chat"
TEMPERATURE = 0.8
TIMEOUT_S = 60
RETRIES = 3

# One angle per batch, rotated. VS_COMPETITOR is expanded per competitor
# and only used when includeCompetitorPrompts is on.
VS_COMPETITOR = "vs competitor"
ANGLES = {
    "safety": [
        "pregnancy and breastfeeding",
        "alcohol",
        "long-term use",
        "other medicines",
        "missed dose",
        "stopping treatment",
        "infections",
    ],
    "efficacy": [
        "time to work",
        "what results to expect",
        "if it stops working",
        "flare-ups",
        VS_COMPETITOR,
    ],
    "access_cost": [
        "getting it on the NHS",
        "prescription charge",
        "GP vs specialist referral",
        "switching from current treatment",
        "availability",
    ],
}

# Harvested questions are tagged with the first topic (in this order) with a keyword
# starting a word in the question. No match -> GENERAL.
TOPIC_KEYWORDS = {
    "safety": ["side effect", "safe", "pregnan", "breastfeed", "alcohol", "risk", "stop",
               "dangerous", "bad for", "interact", "together", "liver", "kidney"],
    "efficacy": ["work", "effective", "how long", "results", "clear"],
    # design said "get"; too broad ("can dogs get psoriasis"), so narrower phrases
    "access_cost": ["nhs", "cost", "price", "prescription", "prescribed", "get it", "buy",
                    "over the counter", "afford", "insurance", "pharmacy"],
}
GENERAL = "general"
MIN_POOL = 10
MIN_PER_TOPIC = 2  # topics with fewer tagged examples get their fallback examples added

# Autocomplete seeds: prefix + term, term + suffix.
AC_PREFIXES = ["is", "does", "can i take", "how long does", "how much is", "should i"]
AC_SUFFIXES = ["side effects", "safe", "nhs", "cost", "vs"]
AC_MAX_SEEDS = 30
AC_SLEEP_S = 0.3
TRENDS_SLEEP_S = 2
CACHE_PATH = ".cache/harvest.sqlite"
CACHE_TTL_S = 7 * 24 * 3600
QUESTION_WORDS = (
    "what", "how", "is", "are", "can", "does", "do", "should", "will",
    "why", "when", "where", "which", "who",
)

# Used when harvesting yields fewer than MIN_POOL questions.
FALLBACK_EXAMPLES = [
    "how long does {brand} take to work",
    "is {brand} safe to take long term",
    "can i drink alcohol while on {brand}",
    "does {brand} work for {disease}",
    "what happens if i stop taking {brand}",
    "can i get {brand} on the nhs",
    "how much does {brand} cost",
    "is {brand} safe when pregnant",
    "what is the best treatment for {disease}",
    "can i take {brand} with other medicines",
]

# Validation
DOSE_PATTERN = r"\d+\s?(mg|ml|%)"
# When stage != "Launched", drop wording that implies approval or availability.
PRELAUNCH_BLOCK_PATTERN = (
    r"\b(prescribed|prescribing|(is|now|currently) available|available now"
    r"|on the nhs now|(been|is|was) approved|licensed for|in stock|already (on|taking|using))\b"
)
