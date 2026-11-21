CREATE EXTENSION IF NOT EXISTS vector;

-- RAG store
CREATE TABLE IF NOT EXISTS documents (
    id          SERIAL PRIMARY KEY,
    source      TEXT NOT NULL UNIQUE,      -- file path or URL
    title       TEXT,
    ingested_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chunks (
    id          SERIAL PRIMARY KEY,
    document_id INT REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INT NOT NULL,
    content     TEXT NOT NULL,
    page        INT,
    embedding   vector(768) NOT NULL,
    tsv         tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
);
CREATE INDEX IF NOT EXISTS chunks_tsv_idx ON chunks USING gin (tsv);
CREATE INDEX IF NOT EXISTS chunks_embedding_idx
    ON chunks USING hnsw (embedding vector_cosine_ops);

-- "Business" data the agent's tools act on
CREATE TABLE IF NOT EXISTS records (       -- e.g. applicants / customers / orders
    id          TEXT PRIMARY KEY,
    name        TEXT,
    email       TEXT,
    status      TEXT,
    details     JSONB DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS tickets (
    id          SERIAL PRIMARY KEY,
    record_id   TEXT,
    subject     TEXT NOT NULL,
    body        TEXT NOT NULL,
    priority    TEXT DEFAULT 'normal',
    status      TEXT DEFAULT 'open',
    created_at  TIMESTAMPTZ DEFAULT now()
);

-- Tracing (step 4)
CREATE TABLE IF NOT EXISTS traces (
    id              SERIAL PRIMARY KEY,
    created_at      TIMESTAMPTZ DEFAULT now(),
    question        TEXT,
    answer          TEXT,
    retrieved       JSONB,     -- [{chunk_id, source, score}]
    tool_calls      JSONB,     -- [{name, args, result, ms}]
    latency_ms      INT,
    input_tokens    INT,
    output_tokens   INT,
    cost_usd        NUMERIC(10,6),
    error           TEXT
);

-- Sample records so the lookup tool has something to find
INSERT INTO records (id, name, email, status, details) VALUES
 ('APP-1001', 'Ananya Rao',  'ananya@example.com', 'documents_pending', '{"program":"B.Tech CSE","missing":["TC","Aadhaar"]}'),
 ('APP-1002', 'Rahul Verma', 'rahul@example.com',  'admitted',          '{"program":"B.Tech ECE","fee_paid":true}'),
 ('APP-1003', 'Sneha Iyer',  'sneha@example.com',  'under_review',      '{"program":"MBA"}')
ON CONFLICT (id) DO NOTHING;
