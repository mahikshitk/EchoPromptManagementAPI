-- Prompt generation jobs for EchoPromptManagementAPI: one row per POST /generate.
-- Apply as a database owner before deploying; the application never changes the schema.
-- The application role used by the AtrionDB secret needs SELECT, INSERT and UPDATE on this table.

BEGIN;

CREATE TABLE IF NOT EXISTS public.atrion_prompt_generation
(
    generationid uuid PRIMARY KEY,
    cognitosub character varying(128) NOT NULL,
    requestid character varying(255) NOT NULL,
    runid character varying(64) NOT NULL,
    status character varying(16) NOT NULL DEFAULT 'queued',
    payload jsonb NOT NULL,
    result jsonb,
    error text,
    pairsdone integer NOT NULL DEFAULT 0,
    pairstotal integer NOT NULL DEFAULT 0,
    createdat timestamp with time zone NOT NULL DEFAULT now(),
    updatedat timestamp with time zone NOT NULL DEFAULT now(),
    startedat timestamp with time zone,
    completedat timestamp with time zone,
    CONSTRAINT atrion_prompt_generation_status_ck
        CHECK (status IN ('queued', 'running', 'completed', 'failed')),
    CONSTRAINT atrion_prompt_generation_progress_ck
        CHECK (pairsdone >= 0 AND pairstotal >= 0),
    CONSTRAINT atrion_prompt_generation_payload_ck
        CHECK (jsonb_typeof(payload) = 'object'),
    CONSTRAINT atrion_prompt_generation_result_ck
        CHECK (result IS NULL OR jsonb_typeof(result) = 'object')
);

-- Per-user limit on unfinished generations (POST /generate counts these).
CREATE INDEX IF NOT EXISTS atrion_prompt_generation_owner_active_ix
    ON public.atrion_prompt_generation (cognitosub, createdat)
    WHERE status IN ('queued', 'running');

CREATE INDEX IF NOT EXISTS atrion_prompt_generation_runid_ix
    ON public.atrion_prompt_generation (runid);

COMMIT;
