-- Harvest store for EchoPromptManagementAPI: every Google Autocomplete / Trends result ever fetched,
-- used as few-shot style examples for patient prompts. Append-only: a refresh adds new texts and never
-- deletes, so concurrent workers cannot lose each other's rows.
-- Apply as a database owner before deploying; the application never changes the schema.
-- The application role used by the AtrionDB secret needs SELECT, INSERT and UPDATE on both tables.

BEGIN;

-- When each search was last fetched (refreshed after 7 days).
CREATE TABLE IF NOT EXISTS public.atrion_prompt_harvest_fetch
(
    source character varying(32) NOT NULL,
    query character varying(512) NOT NULL,
    lang character varying(16) NOT NULL,
    geo character varying(8) NOT NULL,
    fetchedat timestamp with time zone NOT NULL,
    PRIMARY KEY (source, query, lang, geo),
    CONSTRAINT atrion_prompt_harvest_fetch_source_ck
        CHECK (source IN ('autocomplete', 'trends'))
);

-- Every distinct result per search, in the order first seen.
CREATE TABLE IF NOT EXISTS public.atrion_prompt_harvest_result
(
    resultid bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    source character varying(32) NOT NULL,
    query character varying(512) NOT NULL,
    lang character varying(16) NOT NULL,
    geo character varying(8) NOT NULL,
    text character varying(500) NOT NULL,
    firstseen timestamp with time zone NOT NULL DEFAULT now(),
    CONSTRAINT atrion_prompt_harvest_result_uq
        UNIQUE (source, query, lang, geo, text),
    CONSTRAINT atrion_prompt_harvest_result_source_ck
        CHECK (source IN ('autocomplete', 'trends'))
);

COMMIT;
