"""TigerData (TimescaleDB + pgvector) integration for LedgerGone.

Manages time-series financial events and semantic vector search for kirana store ledger items.
"""
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import psycopg2
from psycopg2.extras import RealDictCursor
from pgvector.psycopg2 import register_vector
from src.config import DATABASE_URL, EMBEDDING_DIM


def get_connection():
    """Get a connected PostgreSQL client configured for Timescale and pgvector."""
    conn = psycopg2.connect(DATABASE_URL)
    register_vector(conn)
    return conn


def init_db():
    """Idempotently initialize TigerData TimescaleDB hypertables and pgvector tables."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            # 1. Ensure required PostgreSQL extensions
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;")

            # 2. Time-series master ledger table
            # Must include created_at in primary key for TimescaleDB hypertable partitioning
            cur.execute("""
                CREATE TABLE IF NOT EXISTS ledger_entries (
                    id SERIAL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    vendor_name TEXT NOT NULL,
                    entry_type VARCHAR(50) NOT NULL DEFAULT 'PURCHASE',
                    invoice_no VARCHAR(100),
                    subtotal NUMERIC(12,2) DEFAULT 0.0,
                    gst_amount NUMERIC(12,2) DEFAULT 0.0,
                    total_amount NUMERIC(12,2) NOT NULL DEFAULT 0.0,
                    paid_amount NUMERIC(12,2) NOT NULL DEFAULT 0.0,
                    pending_balance NUMERIC(12,2) NOT NULL DEFAULT 0.0,
                    payment_mode VARCHAR(50) DEFAULT 'CASH',
                    source_type VARCHAR(50) DEFAULT 'RECEIPT_IMAGE',
                    notes TEXT,
                    raw_text TEXT,
                    PRIMARY KEY (id, created_at)
                );
            """)

            # Convert ledger_entries into a TimescaleDB Hypertable partitioned on created_at
            try:
                cur.execute("""
                    SELECT create_hypertable('ledger_entries', 'created_at', if_not_exists => TRUE);
                """)
            except Exception as e:
                # May already be a hypertable or single table fallback
                print(f"[Timescale Hypertable Notice]: {e}")

            # 3. Line items table with pgvector embeddings for semantic item discovery
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS ledger_items (
                    id SERIAL PRIMARY KEY,
                    entry_id INT NOT NULL,
                    entry_created_at TIMESTAMPTZ NOT NULL,
                    item_name TEXT NOT NULL,
                    quantity NUMERIC(10,2) DEFAULT 1.0,
                    unit_price NUMERIC(12,2) DEFAULT 0.0,
                    total_price NUMERIC(12,2) DEFAULT 0.0,
                    item_embedding vector({EMBEDDING_DIM})
                );
            """)

            # 4. HNSW Vector Index for fast semantic item matching
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_ledger_items_embedding
                ON ledger_items USING hnsw (item_embedding vector_cosine_ops);
            """)

            # Index for fast vendor search and time-series range lookups
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_ledger_vendor_time
                ON ledger_entries (vendor_name, created_at DESC);
            """)

        conn.commit()


def insert_ledger_entry(
    entry_data: Dict[str, Any],
    items: List[Dict[str, Any]],
    embeddings: Optional[List[List[float]]] = None
) -> int:
    """Insert a new audited ledger entry and its vectorized line items in a single transaction."""
    created_at = entry_data.get("created_at") or datetime.now()
    
    with get_connection() as conn:
        with conn.cursor() as cur:
            # Insert into time-series ledger_entries hypertable
            cur.execute("""
                INSERT INTO ledger_entries (
                    created_at, vendor_name, entry_type, invoice_no,
                    subtotal, gst_amount, total_amount, paid_amount,
                    pending_balance, payment_mode, source_type, notes, raw_text
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id;
            """, (
                created_at,
                entry_data.get("vendor_name", "Unknown Vendor"),
                entry_data.get("entry_type", "PURCHASE"),
                entry_data.get("invoice_no", ""),
                float(entry_data.get("subtotal", 0.0) or 0.0),
                float(entry_data.get("gst_amount", 0.0) or 0.0),
                float(entry_data.get("total_amount", 0.0) or 0.0),
                float(entry_data.get("paid_amount", 0.0) or 0.0),
                float(entry_data.get("pending_balance", 0.0) or 0.0),
                entry_data.get("payment_mode", "CASH"),
                entry_data.get("source_type", "RECEIPT_IMAGE"),
                entry_data.get("notes", ""),
                entry_data.get("raw_text", "")
            ))
            
            entry_id = cur.fetchone()[0]

            # Insert line items with vector embeddings
            if items:
                for idx, item in enumerate(items):
                    emb = embeddings[idx] if embeddings and idx < len(embeddings) else None
                    cur.execute(f"""
                        INSERT INTO ledger_items (
                            entry_id, entry_created_at, item_name, quantity, unit_price, total_price, item_embedding
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s);
                    """, (
                        entry_id,
                        created_at,
                        item.get("item_name", "General Item"),
                        float(item.get("quantity", 1.0) or 1.0),
                        float(item.get("unit_price", 0.0) or 0.0),
                        float(item.get("total_price", 0.0) or 0.0),
                        emb
                    ))

        conn.commit()
        return entry_id


def get_ledger_entries(vendor_filter: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
    """Retrieve time-series ledger entries with optional vendor filtering."""
    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            if vendor_filter and vendor_filter.strip():
                cur.execute("""
                    SELECT * FROM ledger_entries
                    WHERE vendor_name ILIKE %s
                    ORDER BY created_at DESC
                    LIMIT %s;
                """, (f"%{vendor_filter.strip()}%", limit))
            else:
                cur.execute("""
                    SELECT * FROM ledger_entries
                    ORDER BY created_at DESC
                    LIMIT %s;
                """, (limit,))
            return [dict(row) for row in cur.fetchall()]


def get_summary_metrics() -> Dict[str, Any]:
    """Calculate overall Kirana store khata metrics from TimescaleDB."""
    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT 
                    COALESCE(SUM(total_amount), 0) AS total_turnover,
                    COALESCE(SUM(paid_amount), 0) AS total_paid,
                    COALESCE(SUM(pending_balance), 0) AS total_pending_udhaar,
                    COALESCE(COUNT(*), 0) AS total_entries,
                    COALESCE(COUNT(DISTINCT vendor_name), 0) AS total_parties
                FROM ledger_entries;
            """)
            summary = dict(cur.fetchone() or {})

            # Top pending vendors (Udhaar list)
            cur.execute("""
                SELECT vendor_name, 
                       SUM(pending_balance) AS pending,
                       MAX(created_at) AS last_active
                FROM ledger_entries
                GROUP BY vendor_name
                HAVING SUM(pending_balance) > 0
                ORDER BY pending DESC
                LIMIT 5;
            """)
            summary["top_creditors"] = [dict(r) for r in cur.fetchall()]

            return summary


def get_time_series_cashflow(days: int = 30) -> List[Dict[str, Any]]:
    """Use TimescaleDB time_bucket to aggregate daily financial flows."""
    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            try:
                # TimescaleDB time_bucket query
                cur.execute("""
                    SELECT 
                        time_bucket('1 day', created_at) AS bucket_day,
                        COALESCE(SUM(total_amount), 0) AS daily_total,
                        COALESCE(SUM(paid_amount), 0) AS daily_paid,
                        COALESCE(SUM(pending_balance), 0) AS daily_udhaar,
                        COUNT(*) AS entries_count
                    FROM ledger_entries
                    WHERE created_at >= NOW() - INTERVAL '%s days'
                    GROUP BY bucket_day
                    ORDER BY bucket_day ASC;
                """, (days,))
                return [dict(row) for row in cur.fetchall()]
            except Exception:
                # Fallback to standard DATE_TRUNC if hypertable is in transition
                cur.execute("""
                    SELECT 
                        DATE_TRUNC('day', created_at) AS bucket_day,
                        COALESCE(SUM(total_amount), 0) AS daily_total,
                        COALESCE(SUM(paid_amount), 0) AS daily_paid,
                        COALESCE(SUM(pending_balance), 0) AS daily_udhaar,
                        COUNT(*) AS entries_count
                    FROM ledger_entries
                    WHERE created_at >= NOW() - INTERVAL '%s days'
                    GROUP BY bucket_day
                    ORDER BY bucket_day ASC;
                """, (days,))
                return [dict(row) for row in cur.fetchall()]


def semantic_search_items(
    query_vector: List[float],
    limit: int = 10
) -> List[Dict[str, Any]]:
    """Search historical Kirana line items using pgvector cosine distance (<=>)."""
    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT 
                    li.id,
                    li.item_name,
                    li.quantity,
                    li.unit_price,
                    li.total_price,
                    le.vendor_name,
                    le.invoice_no,
                    le.created_at,
                    le.payment_mode,
                    1 - (li.item_embedding <=> %s::vector) AS similarity_score
                FROM ledger_items li
                JOIN ledger_entries le ON li.entry_id = le.id AND li.entry_created_at = le.created_at
                WHERE li.item_embedding IS NOT NULL
                ORDER BY li.item_embedding <=> %s::vector ASC
                LIMIT %s;
            """, (query_vector, query_vector, limit))
            return [dict(row) for row in cur.fetchall()]


def get_vendor_statement(vendor_name: str) -> Dict[str, Any]:
    """Retrieve full statement and running ledger for a specific vendor/customer."""
    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT * FROM ledger_entries
                WHERE vendor_name ILIKE %s
                ORDER BY created_at ASC;
            """, (vendor_name.strip(),))
            entries = [dict(row) for row in cur.fetchall()]

            cur.execute("""
                SELECT 
                    COALESCE(SUM(total_amount), 0) AS total_billed,
                    COALESCE(SUM(paid_amount), 0) AS total_paid,
                    COALESCE(SUM(pending_balance), 0) AS total_pending
                FROM ledger_entries
                WHERE vendor_name ILIKE %s;
            """, (vendor_name.strip(),))
            totals = dict(cur.fetchone() or {})

            return {
                "vendor_name": vendor_name,
                "totals": totals,
                "entries": entries
            }
