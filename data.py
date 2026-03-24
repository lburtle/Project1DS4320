"""
=============================================================================
DS 4320 - HW8 Section 8.2: Financial Data Engineering & Forecasting Pipeline
=============================================================================
GOAL:    1 GB+ Multi-Table Analytical Database
STACK:   Python 3.x, DuckDB, yfinance, Parquet
TABLES:  Companies, PriceHistory, Fundamentals, TechnicalIndicators, StockNews
=============================================================================

PROVENANCE NOTE:
    All financial data is sourced in real-time from:
    - Yahoo Finance via the `yfinance` library (price history, fundamentals, metadata)
    - Wikipedia S&P 500 constituent list (ticker universe)
    - Yahoo Finance RSS feeds (news headlines for sentiment analysis)
    No synthetic or fake data is used. Data is pulled on execution date.
=============================================================================
"""

# ─────────────────────────────────────────────────────────────────────────────
# CELL 1 – Install Dependencies
# ─────────────────────────────────────────────────────────────────────────────
# !pip install -q yfinance duckdb pandas pyarrow requests lxml

# ─────────────────────────────────────────────────────────────────────────────
# CELL 2 – Imports & Configuration
# ─────────────────────────────────────────────────────────────────────────────

import yfinance as yf
import duckdb
import pandas as pd
import numpy as np
import requests
import time
import os
import gc
import ctypes
import xml.etree.ElementTree as ET
from io import StringIO
from datetime import datetime
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────
DB_PATH       = "stock_data.db"
PARQUET_DIR   = Path("parquet_exports")
PARQUET_DIR.mkdir(exist_ok=True)

REQUEST_DELAY = 0.5   # seconds between tickers
BATCH_SIZE    = 10    # checkpoint every N tickers (not batch accumulation)
MAX_TICKERS   = 500

print(f"[config] DB={DB_PATH}  parquet={PARQUET_DIR}  max_tickers={MAX_TICKERS}")


def trim_memory():
    """
    Force Python + glibc to release freed memory back to the OS immediately.

    ROOT CAUSE OF 'free(): corrupted unsorted chunks':
    ──────────────────────────────────────────────────
    This is a glibc heap allocator crash, not a Python logic error. It fires
    when large numpy arrays (price history can be 15k rows x 30 float64 cols
    = ~3.6 MB per ticker) are allocated and freed in a tight loop without
    ever being returned to the OS. glibc's free-list grows until its internal
    bookkeeping structures become corrupted.

    The two-part fix:
    1. gc.collect() twice — catches cyclic garbage Python's ref-counter misses.
    2. malloc_trim(0) via ctypes — tells glibc to immediately return all free
       heap pages to the OS. Python's allocator normally holds them for reuse,
       which is efficient but causes heap fragmentation over 500 tickers.

    On macOS/Windows, malloc_trim doesn't exist; gc.collect() alone is enough
    because those allocators have different arena management strategies.
    """
    gc.collect()
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass  # macOS / Windows — gc suffices


# ─────────────────────────────────────────────────────────────────────────────
# CELL 3 – Ticker Universe
# ─────────────────────────────────────────────────────────────────────────────

def get_sp500_tickers() -> list[str]:
    """
    Fetches S&P 500 tickers from Wikipedia.

    Fix history:
    - v1: pd.read_html(url) → Wikipedia 403 (urllib blocked by Wikimedia CDN)
    - v2: requests + pd.read_html(resp.text) → FileNotFoundError (newer pandas
          treats long strings as file paths)
    - v3 (current): requests + StringIO(resp.text) → works on all pandas 2.x
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        )
    }
    resp    = requests.get(url, headers=headers, timeout=15)
    resp.raise_for_status()
    tables  = pd.read_html(StringIO(resp.text), attrs={"id": "constituents"})
    df      = tables[0]
    tickers = (df["Symbol"]
               .str.replace(".", "-", regex=False)
               .dropna().unique().tolist())
    tickers.sort()
    print(f"[tickers] Loaded {len(tickers)} S&P 500 constituents")
    return tickers[:MAX_TICKERS]


# ─────────────────────────────────────────────────────────────────────────────
# CELL 4 – DuckDB Schema
# ─────────────────────────────────────────────────────────────────────────────

def create_schema(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("""
        CREATE TABLE IF NOT EXISTS Companies (
            symbol                VARCHAR PRIMARY KEY,
            short_name            VARCHAR,
            long_name             VARCHAR,
            sector                VARCHAR,
            industry              VARCHAR,
            country               VARCHAR,
            exchange              VARCHAR,
            market_cap            BIGINT,
            full_time_employees   INTEGER,
            website               VARCHAR,
            long_business_summary TEXT,
            fetched_at            TIMESTAMP DEFAULT current_timestamp
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS PriceHistory (
            symbol    VARCHAR NOT NULL,
            date      DATE    NOT NULL,
            open      DOUBLE,
            high      DOUBLE,
            low       DOUBLE,
            close     DOUBLE,
            volume    BIGINT,      -- BIGINT: some tickers exceed INT32 max daily volume
            adj_close DOUBLE,
            PRIMARY KEY (symbol, date)
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS Fundamentals (
            symbol      VARCHAR NOT NULL,
            report_date DATE,
            period_type VARCHAR,
            metric      VARCHAR NOT NULL,
            value       DOUBLE,
            PRIMARY KEY (symbol, report_date, period_type, metric)
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS TechnicalIndicators (
            symbol            VARCHAR NOT NULL,
            date              DATE    NOT NULL,
            close             DOUBLE,
            daily_return      DOUBLE,
            log_return        DOUBLE,
            cumulative_return DOUBLE,
            sma_5             DOUBLE,
            sma_10            DOUBLE,
            sma_20            DOUBLE,
            sma_50            DOUBLE,
            sma_200           DOUBLE,
            ema_12            DOUBLE,
            ema_26            DOUBLE,
            ema_50            DOUBLE,
            macd              DOUBLE,
            macd_signal       DOUBLE,
            macd_histogram    DOUBLE,
            bb_upper          DOUBLE,
            bb_middle         DOUBLE,
            bb_lower          DOUBLE,
            bb_width          DOUBLE,
            bb_pct_b          DOUBLE,
            rsi_14            DOUBLE,
            volume            BIGINT,
            volume_sma_20     DOUBLE,
            volume_ratio      DOUBLE,
            atr_14            DOUBLE,
            hist_vol_20       DOUBLE,
            PRIMARY KEY (symbol, date)
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS StockNews (
            id                    INTEGER PRIMARY KEY,
            symbol                VARCHAR,
            title                 VARCHAR,
            publisher             VARCHAR,
            link                  VARCHAR,
            provider_publish_time TIMESTAMP,
            news_type             VARCHAR,
            fetched_at            TIMESTAMP DEFAULT current_timestamp
        )
    """)
    print("[schema] All tables created / verified.")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 5 – Technical Indicators (pure pandas, Python 3.14 compatible)
# ─────────────────────────────────────────────────────────────────────────────

def compute_technical_indicators(hist: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    All indicators computed with pandas .rolling()/.ewm() only.
    No Numba, no TA-Lib — fully Python 3.14 compatible.
    """
    if hist.empty or len(hist) < 30:
        return pd.DataFrame()

    # Only copy the columns we actually need — reduces peak memory
    df = hist[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index = pd.to_datetime(df.index).normalize()
    df.index.name = "date"

    df["daily_return"]      = df["Close"].pct_change()
    df["log_return"]        = np.log(df["Close"] / df["Close"].shift(1))
    df["cumulative_return"] = (1 + df["daily_return"]).cumprod() - 1

    for w in [5, 10, 20, 50, 200]:
        df[f"sma_{w}"] = df["Close"].rolling(w).mean()

    for span in [12, 26, 50]:
        df[f"ema_{span}"] = df["Close"].ewm(span=span, adjust=False).mean()

    df["macd"]           = df["ema_12"] - df["ema_26"]
    df["macd_signal"]    = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_histogram"] = df["macd"] - df["macd_signal"]

    roll20          = df["Close"].rolling(20)
    df["bb_middle"] = roll20.mean()
    bb_std          = roll20.std()
    df["bb_upper"]  = df["bb_middle"] + 2 * bb_std
    df["bb_lower"]  = df["bb_middle"] - 2 * bb_std
    df["bb_width"]  = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]
    df["bb_pct_b"]  = (df["Close"] - df["bb_lower"]) / (
                       df["bb_upper"] - df["bb_lower"])

    delta    = df["Close"].diff()
    avg_gain = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
    avg_loss = (-delta).clip(lower=0).ewm(com=13, adjust=False).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    df["volume_sma_20"] = df["Volume"].rolling(20).mean()
    df["volume_ratio"]  = df["Volume"] / df["volume_sma_20"]

    hl  = df["High"] - df["Low"]
    hcp = (df["High"] - df["Close"].shift(1)).abs()
    lcp = (df["Low"]  - df["Close"].shift(1)).abs()
    df["atr_14"] = (pd.concat([hl, hcp, lcp], axis=1)
                    .max(axis=1)
                    .ewm(com=13, adjust=False).mean())

    df["hist_vol_20"] = df["log_return"].rolling(20).std() * np.sqrt(252)

    df["symbol"] = symbol
    df["volume"] = df["Volume"].fillna(0).astype(np.int64)
    df = df.rename(columns={"Close": "close"})

    keep = [
        "symbol", "close",
        "daily_return", "log_return", "cumulative_return",
        "sma_5", "sma_10", "sma_20", "sma_50", "sma_200",
        "ema_12", "ema_26", "ema_50",
        "macd", "macd_signal", "macd_histogram",
        "bb_upper", "bb_middle", "bb_lower", "bb_width", "bb_pct_b",
        "rsi_14", "volume", "volume_sma_20", "volume_ratio",
        "atr_14", "hist_vol_20",
    ]
    result = df[keep].reset_index()
    result["date"] = result["date"].dt.date
    return result


# ─────────────────────────────────────────────────────────────────────────────
# CELL 6 – Fundamentals (wide → long EAV)
# ─────────────────────────────────────────────────────────────────────────────

def extract_fundamentals(tk: yf.Ticker, symbol: str) -> pd.DataFrame:
    records = []

    def _stack(df, period_type):
        if df is None or df.empty:
            return
        for col in df.columns:
            for metric, val in df[col].items():
                if pd.notna(val):
                    records.append({
                        "symbol":      symbol,
                        "report_date": col.date() if hasattr(col, "date") else col,
                        "period_type": period_type,
                        "metric":      str(metric),
                        "value":       float(val),
                    })

    try:
        _stack(tk.income_stmt,             "annual")
        _stack(tk.quarterly_income_stmt,   "quarterly")
        _stack(tk.balance_sheet,           "annual")
        _stack(tk.quarterly_balance_sheet, "quarterly")
        _stack(tk.cash_flow,               "annual")
        _stack(tk.quarterly_cash_flow,     "quarterly")
    except Exception as e:
        print(f"  [fundamentals warning] {symbol}: {e}")

    return pd.DataFrame(records) if records else pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# CELL 7 – News (Yahoo Finance RSS)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_news_rss(symbol: str, max_items: int = 30) -> pd.DataFrame:
    url = (f"https://feeds.finance.yahoo.com/rss/2.0/headline"
           f"?s={symbol}&region=US&lang=en-US")
    records = []
    try:
        resp = requests.get(
            url, timeout=8,
            headers={"User-Agent": "Mozilla/5.0 (research bot)"})
        if resp.status_code != 200:
            return pd.DataFrame()
        root  = ET.fromstring(resp.content)
        items = root.findall(".//item")[:max_items]
        for item in items:
            pub = item.findtext("pubDate", "")
            try:
                pt = datetime.strptime(pub, "%a, %d %b %Y %H:%M:%S %z")
            except Exception:
                pt = None
            records.append({
                "symbol":                symbol,
                "title":                 item.findtext("title", "").strip(),
                "publisher":             item.findtext("source", "Yahoo Finance"),
                "link":                  item.findtext("link", "").strip(),
                "provider_publish_time": pt,
                "news_type":             "RSS",
            })
    except Exception:
        pass
    return pd.DataFrame(records) if records else pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# CELL 8 – Per-ticker flush (write immediately, release immediately)
# ─────────────────────────────────────────────────────────────────────────────

_news_id_counter = 0


def flush_ticker(con, co_row, ph_df, ti_df, fu_df, sn_df):
    """
    Writes one ticker's data to DuckDB and returns immediately.

    KEY DESIGN DECISION — per-ticker vs. batch flush:
    ─────────────────────────────────────────────────
    The original design accumulated 50 tickers in Python lists before writing.
    That kept ~270 MB of numpy arrays alive simultaneously, which corrupted
    glibc's heap free-list. Writing per-ticker limits peak live allocations to
    one ticker's worth of data (~5-20 MB), eliminating the crash entirely.
    DuckDB buffers writes in its WAL anyway, so there is no I/O amplification.
    """
    global _news_id_counter

    if co_row:
        df_co = pd.DataFrame([co_row])
        con.execute("""
            INSERT INTO Companies
              (symbol, short_name, long_name, sector, industry, country,
               exchange, market_cap, full_time_employees, website,
               long_business_summary)
            SELECT symbol, short_name, long_name, sector, industry, country,
                   exchange, market_cap, full_time_employees, website,
                   long_business_summary
            FROM df_co
            ON CONFLICT (symbol) DO UPDATE SET
                short_name            = EXCLUDED.short_name,
                long_name             = EXCLUDED.long_name,
                sector                = EXCLUDED.sector,
                industry              = EXCLUDED.industry,
                country               = EXCLUDED.country,
                exchange              = EXCLUDED.exchange,
                market_cap            = EXCLUDED.market_cap,
                full_time_employees   = EXCLUDED.full_time_employees,
                website               = EXCLUDED.website,
                long_business_summary = EXCLUDED.long_business_summary,
                fetched_at            = now()
        """)
        del df_co

    if ph_df is not None and not ph_df.empty:
        con.execute("""
            INSERT INTO PriceHistory
              (symbol, date, open, high, low, close, volume, adj_close)
            SELECT symbol, date, open, high, low, close, volume, adj_close
            FROM ph_df
            ON CONFLICT (symbol, date) DO UPDATE SET
                open = EXCLUDED.open, high = EXCLUDED.high,
                low  = EXCLUDED.low,  close = EXCLUDED.close,
                volume = EXCLUDED.volume, adj_close = EXCLUDED.adj_close
        """)

    if ti_df is not None and not ti_df.empty:
        cols = [
            "symbol", "date", "close", "daily_return", "log_return",
            "cumulative_return", "sma_5", "sma_10", "sma_20", "sma_50",
            "sma_200", "ema_12", "ema_26", "ema_50", "macd", "macd_signal",
            "macd_histogram", "bb_upper", "bb_middle", "bb_lower",
            "bb_width", "bb_pct_b", "rsi_14", "volume", "volume_sma_20",
            "volume_ratio", "atr_14", "hist_vol_20",
        ]
        c_str   = ", ".join(cols)
        set_str = ", ".join(
            [f"{c} = EXCLUDED.{c}" for c in cols if c not in ("symbol", "date")]
        )
        con.execute(
            f"INSERT INTO TechnicalIndicators ({c_str}) "
            f"SELECT {c_str} FROM ti_df "
            f"ON CONFLICT (symbol, date) DO UPDATE SET {set_str}"
        )

    if fu_df is not None and not fu_df.empty:
        con.execute("""
            INSERT INTO Fundamentals
              (symbol, report_date, period_type, metric, value)
            SELECT symbol, report_date, period_type, metric, value
            FROM fu_df
            ON CONFLICT (symbol, report_date, period_type, metric)
            DO UPDATE SET value = EXCLUDED.value
        """)

    if sn_df is not None and not sn_df.empty:
        sn_df = sn_df.copy()
        sn_df.insert(0, "id",
                     range(_news_id_counter,
                           _news_id_counter + len(sn_df)))
        _news_id_counter += len(sn_df)
        con.execute("""
            INSERT OR IGNORE INTO StockNews
              (id, symbol, title, publisher, link,
               provider_publish_time, news_type)
            SELECT id, symbol, title, publisher, link,
                   provider_publish_time, news_type
            FROM sn_df
        """)


# ─────────────────────────────────────────────────────────────────────────────
# CELL 9 – Main Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline():
    tickers = get_sp500_tickers()
    con     = duckdb.connect(DB_PATH)
    con.execute("PRAGMA threads=4")

    # Clean slate (remove these DROP lines to resume a partial run instead)
    for t in ["Companies", "PriceHistory", "Fundamentals",
              "TechnicalIndicators", "StockNews"]:
        con.execute(f"DROP TABLE IF EXISTS {t}")
    create_schema(con)

    for i, symbol in enumerate(tickers):
        print(f"[{i+1:>4}/{len(tickers)}] {symbol}", end=" ... ", flush=True)

        # Declare all vars up front so `finally` block can always del them
        tk = info = hist = co_row = ph_df = ti_df = fu_df = sn_df = None

        try:
            tk   = yf.Ticker(symbol)
            info = tk.info or {}

            co_row = {
                "symbol":                symbol,
                "short_name":            info.get("shortName"),
                "long_name":             info.get("longName"),
                "sector":                info.get("sector"),
                "industry":              info.get("industry"),
                "country":               info.get("country"),
                "exchange":              info.get("exchange"),
                "market_cap":            info.get("marketCap"),
                "full_time_employees":   info.get("fullTimeEmployees"),
                "website":               info.get("website"),
                "long_business_summary": info.get("longBusinessSummary"),
            }

            hist = tk.history(period="max", auto_adjust=True)
            if not hist.empty:
                ph_df = hist[["Open", "High", "Low", "Close", "Volume"]].copy()
                ph_df.index = pd.to_datetime(ph_df.index).normalize()
                ph_df.index.name = "date"
                ph_df = ph_df.reset_index()
                ph_df["symbol"]    = symbol
                ph_df["date"]      = ph_df["date"].dt.date
                ph_df["volume"]    = ph_df["Volume"].fillna(0).astype(np.int64)
                ph_df["adj_close"] = ph_df["Close"]
                ph_df = ph_df.rename(columns={
                    "Open": "open", "High": "high",
                    "Low":  "low",  "Close": "close",
                })[["symbol", "date", "open", "high", "low",
                    "close", "volume", "adj_close"]]

                ti_df = compute_technical_indicators(hist, symbol)
                if ti_df is not None and ti_df.empty:
                    ti_df = None

            # Free the largest object before fetching fundamentals
            del hist
            hist = None
            info = None

            fu_df = extract_fundamentals(tk, symbol)
            if fu_df is not None and fu_df.empty:
                fu_df = None

            del tk
            tk = None

            sn_df = fetch_news_rss(symbol)
            if sn_df is not None and sn_df.empty:
                sn_df = None

            flush_ticker(con, co_row, ph_df, ti_df, fu_df, sn_df)
            print("✓")

        except Exception as e:
            print(f"✗  [{type(e).__name__}] {e}")

        finally:
            # Unconditionally release every object — even on exception paths
            del tk, info, hist, co_row, ph_df, ti_df, fu_df, sn_df
            trim_memory()   # gc.collect() × 2 + malloc_trim(0)

        time.sleep(REQUEST_DELAY)

        # Checkpoint WAL to main file every BATCH_SIZE tickers
        if (i + 1) % BATCH_SIZE == 0:
            con.execute("CHECKPOINT")
            db_mb  = os.path.getsize(DB_PATH) / 1e6
            wal_mb = (os.path.getsize(DB_PATH + ".wal")
                      if os.path.exists(DB_PATH + ".wal") else 0) / 1e6
            print(f"\n  ── checkpoint [{i+1}/{len(tickers)}] "
                  f"db={db_mb:.0f} MB  wal={wal_mb:.0f} MB ──\n")

    con.execute("CHECKPOINT")

    # ── Parquet export ─────────────────────────────────────────────────────
    print("\n[export] Writing Parquet files ...")
    for table in ["Companies", "PriceHistory", "Fundamentals",
                  "TechnicalIndicators", "StockNews"]:
        out = PARQUET_DIR / f"{table}.parquet"
        con.execute(f"COPY {table} TO '{out}' (FORMAT PARQUET, COMPRESSION ZSTD)")
        print(f"  → {out}  ({out.stat().st_size / 1e6:.1f} MB)")

    # ── Verification ───────────────────────────────────────────────────────
    print("\n══════════════════════════════════════════")
    print("  VERIFICATION QUERIES")
    print("══════════════════════════════════════════")
    queries = {
        "Row counts": """
            SELECT 'PriceHistory'        AS tbl, COUNT(*) AS rows FROM PriceHistory UNION ALL
            SELECT 'TechnicalIndicators' AS tbl, COUNT(*) AS rows FROM TechnicalIndicators UNION ALL
            SELECT 'Fundamentals'        AS tbl, COUNT(*) AS rows FROM Fundamentals UNION ALL
            SELECT 'StockNews'           AS tbl, COUNT(*) AS rows FROM StockNews UNION ALL
            SELECT 'Companies'           AS tbl, COUNT(*) AS rows FROM Companies
        """,
        "Date range": "SELECT MIN(date) AS earliest, MAX(date) AS latest FROM PriceHistory",
        "Max volume (INT32 overflow check)": """
            SELECT symbol, MAX(volume) AS max_vol
            FROM PriceHistory GROUP BY symbol ORDER BY max_vol DESC LIMIT 5
        """,
        "AAPL indicators (latest 5)": """
            SELECT * FROM TechnicalIndicators WHERE symbol='AAPL'
            ORDER BY date DESC LIMIT 5
        """,
        "Sector distribution": """
            SELECT sector, COUNT(*) AS n FROM Companies
            WHERE sector IS NOT NULL GROUP BY sector ORDER BY n DESC
        """,
    }
    for label, sql in queries.items():
        print(f"\n── {label}")
        print(con.execute(sql).df().to_string(index=False))

    db_mb = os.path.getsize(DB_PATH) / 1e6
    print(f"\n[done] stock_data.db = {db_mb:.1f} MB")
    con.close()


if __name__ == "__main__":
    run_pipeline()