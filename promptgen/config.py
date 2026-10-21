"""Tunable constants and prompt content. Adding an audience or topic only needs edits here."""

# Codes the service knows how to write for. Payload entries with other codes are ignored with a warning.
AUDIENCES = ["patient", "hcp", "policy_payer", "media_investor"]
TOPICS = ["efficacy", "safety", "moa", "population", "disease", "access_cost", "corporate"]

# Output intent codes -> meaning (the meaning is shown to the LLM).
INTENTS = {
    "informational": "general facts about the medicine or the condition",
    "comparison": "how it compares with a named competitor or another option",
    "side_effects": "adverse effects, risks and safety concerns",
    "suitability": "who it is for, eligibility, who should avoid it",
    "experience": "what taking or prescribing it is like in practice",
    "alternatives": "other options if it is unsuitable or stops working",
    "cost_access": "getting a prescription, personal cost, availability",
    "mechanism": "how the medicine works",
    "dosing_administration": "how it is taken or given, schedule, monitoring",
    "clinical_evidence": "trial data, endpoints, real-world evidence",
    "guideline": "place in clinical guidelines and treatment pathways",
    "cost_effectiveness": "value for money, cost per outcome, budget impact",
    "reimbursement": "HTA decisions, formulary listing, funding and coverage",
    "market_outlook": "sales, uptake, market share, growth, competition",
    "controversy": "criticism, safety signals, pricing disputes, litigation",
    "corporate": "the company behind the medicine: strategy, deals, reputation",
}
DEFAULT_INTENT = "informational"  # used when the LLM returns an intent not allowed for the audience

# Who is asking. "voice" comes from the payload; persona, rules and allowed intents are ours.
# block_doses: reject questions containing a dose (fine for clinicians, not for patients).
# fewshot: show harvested Google searches as style examples (real searches are patient-style).
AUDIENCE_PROFILES = {
    "patient": {
        "persona": "a patient living with the condition (or their carer)",
        "rules": [
            "First person, lay language, no medical jargon.",
            "Everyday worries and practical questions, the way people really type them.",
            "No doses, percentages or other numbers.",
            "Do not invent symptoms, other conditions or personal medical histories.",
        ],
        "intents": ["informational", "comparison", "side_effects", "suitability", "experience",
                    "alternatives", "cost_access", "mechanism", "dosing_administration"],
        "block_doses": True,
        "fewshot": True,
    },
    "hcp": {
        "persona": "a practising clinician (specialist, GP, nurse or pharmacist) deciding how to treat patients",
        "rules": [
            "Clinical language: drug classes, guideline names, endpoints and outcome measures used in this disease area.",
            "Questions about patient management, evidence and prescribing decisions, not personal experience.",
            "Ask about data; never state results.",
        ],
        "intents": ["informational", "comparison", "side_effects", "suitability", "experience", "alternatives",
                    "cost_access", "mechanism", "dosing_administration", "clinical_evidence", "guideline"],
        "block_doses": False,
        "fewshot": False,
    },
    "policy_payer": {
        "persona": "a health economist, HTA assessor, formulary or commissioning decision-maker",
        "rules": [
            "Name the HTA, reimbursement and commissioning bodies natural for the market.",
            "Focus on value, budget impact, comparative effectiveness, eligibility and reimbursement status.",
            "Ask about evidence and prices; never state figures.",
        ],
        "intents": ["informational", "comparison", "side_effects", "suitability", "alternatives",
                    "clinical_evidence", "guideline", "cost_effectiveness", "reimbursement"],
        "block_doses": False,
        "fewshot": False,
    },
    "media_investor": {
        "persona": ("an equity analyst, investor or business / pharma trade journalist covering the medicine "
                    "and its company"),
        "rules": [
            "Every question is about commercial or news impact: sales, uptake, market share, forecasts, pricing, "
            "competition, pipeline, deals, and risks to the company or the investment case.",
            "Turn clinical, safety, eligibility and reimbursement facts into what they mean for the business: "
            "ask what a reimbursement decision means for uptake or sales, never whether a patient is eligible.",
            "Use investor and trade-press vocabulary where it fits: launch trajectory, peak sales, consensus "
            "or Street expectations, guidance, net pricing, share of new patients, loss of exclusivity, "
            "catalysts, read-across to competitors.",
            "Mix analyst questions (specific, forward-looking: what is assumed, what could change the outlook) "
            "with journalist questions (why, who is affected, what happens next, what critics say). "
            "Vary the openings.",
            "Ask about allegations, risks and numbers; never assert them.",
        ],
        "intents": ["informational", "comparison", "clinical_evidence", "reimbursement",
                    "market_outlook", "controversy", "corporate"],
        "block_doses": False,
        "fewshot": False,
    },
}

# What each topic covers. brand: how often to name the brand ("most" or "some" of the questions).
TOPIC_PROFILES = {
    "efficacy": {"focus": "how well it works: results, speed, durability, loss of response", "brand": "most"},
    "safety": {"focus": "side effects, risks, interactions, special groups, long-term safety", "brand": "most"},
    "moa": {"focus": "how the medicine works in the body and why that matters", "brand": "most"},
    "population": {"focus": "who it is for: eligibility, who should not use it, special groups",
                   "brand": "most"},
    "disease": {"focus": "the condition itself and its treatment landscape; the brand is one option among several",
                "brand": "some"},
    "access_cost": {"focus": "getting it and paying for it: prescription route, price, funding, availability",
                    "brand": "most"},
    "corporate": {"focus": "the company that makes it: reputation, strategy, pipeline, news", "brand": "most"},
}

# Generation
BATCH_SIZE = 12          # max prompts accepted per LLM call
OVERGENERATE = 1.3       # ask the LLM for this many x batch quota
TOPUP_CALLS = 2          # extra calls per short pair
MAX_WORKERS = 8          # pairs generated in parallel
DEDUP_THRESHOLD = 0.85
MIN_LEN, MAX_LEN = 15, 200
FEWSHOT_TOPIC, FEWSHOT_GENERAL = 6, 3

# LLM
DEFAULT_MODEL = "deepseek/deepseek-chat"
TEMPERATURE = 0.8
TIMEOUT_S = 60
RETRIES = 3

# API jobs
REQUIRED_SCOPE = "run-management/run:management"  # same scope RunManagementAPI's UI tokens carry
MAX_ACTIVE_GENERATIONS = 2   # per user, queued or running
GENERATION_STALE_S = 900     # an unfinished job untouched this long is reported failed (worker timeout is 600s)

# ANGLES[topic][audience]: one angle per LLM call, order shuffled each run. VS_COMPETITOR is
# expanded per competitor and only used when includeCompetitorPrompts is on. Keep angles
# market-neutral; the system prompt asks for the market's own terms (NHS, NICE, ...).
VS_COMPETITOR = "vs competitor"
ANGLES = {
    "efficacy": {
        "patient": ["how long it takes to work", "what results to expect", "if it stops working",
                    "flare-ups while on it", "whether it will work for someone like me", VS_COMPETITOR],
        "hcp": ["onset of response and primary endpoints", "durability of response and long-term data",
                "loss of response and next steps", "difficult-to-treat or refractory patients",
                "real-world versus trial outcomes", VS_COMPETITOR],
        "policy_payer": ["comparative effectiveness against standard of care",
                         "strength and relevance of the trial evidence", "durability of benefit over time",
                         "outcomes that matter to patients and the health system", VS_COMPETITOR],
        "media_investor": ["clinical differentiation as a commercial driver", "upcoming data readouts as catalysts",
                           "real-world data and what it means for uptake", VS_COMPETITOR],
    },
    "safety": {
        "patient": ["common side effects", "pregnancy and breastfeeding", "alcohol and other medicines",
                    "long-term use", "infections and vaccines", "missing a dose or stopping", VS_COMPETITOR],
        "hcp": ["adverse event profile and management", "screening and monitoring before and during treatment",
                "contraindications and drug interactions", "pregnancy, older or comorbid patients",
                "long-term and post-marketing safety", VS_COMPETITOR],
        "policy_payer": ["cost of safety monitoring", "long-term safety uncertainty in the evidence",
                         "safety relative to existing options", "risk management and prescribing restrictions"],
        "media_investor": ["safety signals or regulatory warnings as a risk to sales",
                           "label changes and their commercial impact", "the safety profile as a selling point",
                           "litigation or recall risk"],
    },
    "moa": {
        "patient": ["how it works in simple terms", "how it differs from my current treatment",
                    "why it is taken the way it is", "what it does inside my body"],
        "hcp": ["molecular target and pathway", "how the mechanism differs from other drug classes",
                "mechanism-related safety considerations", "rationale for use after other therapies fail"],
        "policy_payer": ["whether the mechanism is genuinely innovative",
                         "class effect and interchangeability with similar drugs",
                         "mechanism-based rationale for patient subgroups"],
        "media_investor": ["first-in-class or best-in-class potential", "competition within the same drug class",
                           "potential in other diseases", "patent life and loss of exclusivity"],
    },
    "population": {
        "patient": ["whether I am eligible", "children and older people", "other health conditions I have",
                    "pregnancy or planning a family", "having tried other treatments first"],
        "hcp": ["licensed indication and eligibility criteria", "patient selection and who benefits most",
                "contraindicated or cautionary patient groups",
                "special populations (paediatric, elderly, renal or hepatic impairment)",
                "treatment sequencing and prior therapy requirements"],
        "policy_payer": ["size of the eligible population", "eligibility criteria in reimbursement decisions",
                         "subgroups with the best value", "restriction to a line of therapy"],
        "media_investor": ["size of the addressable patient population", "label expansion into new indications",
                           "launch uptake and share of new patients", "underserved patient groups as a growth opportunity"],
    },
    "disease": {
        "patient": ["symptoms and triggers", "best treatment options", "living with it day to day",
                    "whether it can be cured", "when to see a doctor or specialist", "new treatments"],
        "hcp": ["current treatment guidelines and pathway", "disease burden and comorbidities",
                "assessing severity and treatment goals", "unmet needs with current therapies",
                "emerging treatments in development"],
        "policy_payer": ["burden of disease and cost to the health system", "prevalence and trends",
                         "current standard of care and its cost", "unmet need and service gaps"],
        "media_investor": ["market size and growth of the disease area", "competitive landscape of treatments",
                           "pipeline and new entrants", "pricing trends in the category"],
    },
    "access_cost": {
        "patient": ["getting it prescribed", "how much it costs me", "coverage by the health system or insurance",
                    "seeing a specialist to get it", "switching from my current treatment", "availability and supply"],
        "hcp": ["prescribing criteria and local formulary status", "referral and specialist-only prescribing",
                "funding approval process", "switching patients from existing therapy",
                "lower-cost alternatives such as generics or biosimilars"],
        "policy_payer": ["HTA appraisal and recommendation", "price and cost-effectiveness threshold",
                         "budget impact", "discounts, managed access or patient access schemes", VS_COMPETITOR],
        "media_investor": ["pricing strategy and net price pressure",
                           "reimbursement decisions and what they mean for sales",
                           "market access as a driver of sales", "competition from generics or biosimilars",
                           "patient access controversies"],
    },
    "corporate": {
        "patient": ["who makes it", "whether the company can be trusted", "patient support programmes",
                    "company news that could affect my treatment"],
        "hcp": ["manufacturer medical information and support", "the company's research programme",
                "industry relationships and transparency", "supply reliability"],
        "policy_payer": ["the manufacturer's pricing and access commitments",
                         "the company's track record with health systems", "evidence generation commitments"],
        "media_investor": ["company strategy and the brand's role in it",
                           "revenue guidance and consensus expectations",
                           "pipeline and future launches", "leadership, deals and acquisitions",
                           "reputation and controversies"],
    },
}

# Harvested questions (patient few-shot only) are tagged with the first topic, in this order,
# with a keyword starting a word in the question. No match -> GENERAL.
TOPIC_KEYWORDS = {
    "safety": ["side effect", "safe", "pregnan", "breastfeed", "alcohol", "risk", "stop",
               "dangerous", "bad for", "interact", "together", "liver", "kidney"],
    "efficacy": ["work", "effective", "how long", "results", "clear"],
    # design said "get"; too broad ("can dogs get psoriasis"), so narrower phrases
    "access_cost": ["nhs", "cost", "price", "prescription", "prescribed", "get it", "buy",
                    "over the counter", "afford", "insurance", "pharmacy"],
    "population": ["children", "kids", "child", "elderly", "teen", "eligible", "who can", "who should"],
    "corporate": ["who makes", "manufacturer", "company", "made by"],
    "moa": ["mechanism", "what type of", "what kind of", "biologic", "inhibitor", "class"],
    "disease": ["symptom", "cause", "cure", "contagious", "trigger", "treatment for", "flare", "diet", "what is"],
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
HARVEST_MAX_TEXT = 500  # longer results are dropped (the AtrionDB column rejects them)
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
    r"\b(prescribed|(is|now|currently) available|available now"
    r"|on the nhs now|(been|is|was) approved|licensed for|in stock|already (on|taking|using))\b"
)
