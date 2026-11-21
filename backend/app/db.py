from contextlib import contextmanager

from pgvector.psycopg import register_vector
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from .config import DATABASE_URL

pool = ConnectionPool(DATABASE_URL, min_size=1, max_size=5, open=False,
                      configure=lambda c: register_vector(c))


@contextmanager
def conn():
    if pool.closed:
        pool.open()
    with pool.connection() as c:
        c.row_factory = dict_row
        yield c


def ensure_schema():
    """Idempotent upgrades for databases created from an older schema.sql."""
    with conn() as c:
        c.execute("""ALTER TABLE chunks ADD COLUMN IF NOT EXISTS tsv tsvector
                     GENERATED ALWAYS AS (to_tsvector('english', content)) STORED""")
        c.execute("CREATE INDEX IF NOT EXISTS chunks_tsv_idx ON chunks USING gin (tsv)")
