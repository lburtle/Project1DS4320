import yfinance as yf
import duckdb
import pandas as pd
import time
import os
import urllib.request
import io

DB_NAME = "stock_data.db"

def init_db(con):
    # Establish precise schema to avoid Int64 and Catalog errors
    con.execute("""
    CREATE TABLE IF NOT EXISTS Companies (
        symbol VARCHAR,
        name VARCHAR,
        sector VARCHAR,
        industry VARCHAR,
        summary VARCHAR
    );
    """)
    
    con.execute("""
    CREATE TABLE IF NOT EXISTS PriceHistory (
        Date TIMESTAMP,
        Open DOUBLE,
        High DOUBLE,
        Low DOUBLE,
        Close DOUBLE,
        Volume BIGINT,
        symbol VARCHAR
    );
    """)

    con.execute("""
    CREATE TABLE IF NOT EXISTS Fundamentals (
        symbol VARCHAR,
        metric VARCHAR,
        report_date TIMESTAMP,
        value DOUBLE,
        statement_type VARCHAR
    );
    """)
    


def get_sp500_tickers():
    print("Fetching S&P 500 tickers from Wikipedia...")
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    html = urllib.request.urlopen(req).read().decode('utf-8')
    tables = pd.read_html(io.StringIO(html))
    df = tables[0]
    tickers = df['Symbol'].tolist()
    # Replace '.' with '-' for yfinance compatibility (e.g., BRK.B -> BRK-B)
    tickers = [t.replace('.', '-') for t in tickers]
    print(f"Loaded {len(tickers)} tickers.")
    return tickers

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def process_ticker(symbol, con):
    try:
        tk = yf.Ticker(symbol)
        
        # --- TABLE 1: Companies ---
        info = tk.info
        df_comp = pd.DataFrame([{
            "symbol": symbol,
            "name": info.get("longName"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "summary": info.get("longBusinessSummary", "N/A")
        }])
        
        # --- TABLE 2: PriceHistory ---
        df_h = tk.history(period="max").reset_index()
        if df_h.empty:
            return False
            
        # Standardize columns
        if 'Date' not in df_h.columns and 'Datetime' in df_h.columns:
            df_h = df_h.rename(columns={'Datetime': 'Date'})
            
        # Ensure 'Date' is tz-naive to avoid duckdb issues in some cases
        df_h['Date'] = pd.to_datetime(df_h['Date'], utc=True).dt.tz_localize(None)
            
        df_h['symbol'] = symbol
        # Drop columns like Dividends and Stock Splits if they exist to match schema
        cols_to_keep = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'symbol']
        df_h = df_h[[c for c in cols_to_keep if c in df_h.columns]]
        
        # Ensure Volume is integer and fill missing
        df_h['Volume'] = df_h['Volume'].fillna(0).astype('int64')
        
        # --- TABLE 3: Fundamentals ---
        f_list = []
        stmts = {
            "Income": tk.quarterly_financials, 
            "Balance": tk.quarterly_balance_sheet, 
            "Cashflow": tk.quarterly_cashflow
        }
        
        for stmt_name, df in stmts.items():
            if df is not None and not df.empty:
                if isinstance(df, pd.Series):
                    df = df.to_frame()
                
                # Transform: Wide -> Long
                temp_df = df.stack().reset_index()
                temp_df.columns = ['metric', 'report_date', 'value']
                temp_df['symbol'] = symbol
                temp_df['statement_type'] = stmt_name
                # Ensure correct types
                temp_df['report_date'] = pd.to_datetime(temp_df['report_date'], utc=True).dt.tz_localize(None)
                temp_df['value'] = pd.to_numeric(temp_df['value'], errors='coerce')
                temp_df = temp_df.dropna(subset=['value'])
                f_list.append(temp_df)
        
        if f_list:
            df_fund = pd.concat(f_list)
        else:
            df_fund = pd.DataFrame(columns=['symbol', 'metric', 'report_date', 'value', 'statement_type'])
            
        # --- TABLE 4: TechnicalIndicators ---
        df_ind = df_h[['Date', 'symbol', 'Close', 'Volume']].copy()
        
        # Simple Moving Averages
        for w in range(2, 502):
            df_ind[f'SMA_{w}'] = df_ind['Close'].rolling(window=w).mean()
            
        # Exponential Moving Averages
        for w in [20, 50]:
            df_ind[f'EMA_{w}'] = df_ind['Close'].ewm(span=w, adjust=False).mean()
            
        # RSI
        df_ind['RSI_14'] = calculate_rsi(df_ind['Close'], 14)
        
        # Bollinger Bands (20-day, 2 std)
        sma_20 = df_ind['Close'].rolling(window=20).mean()
        std_20 = df_ind['Close'].rolling(window=20).std()
        df_ind['BB_Upper'] = sma_20 + (std_20 * 2)
        df_ind['BB_Lower'] = sma_20 - (std_20 * 2)
        
        # Daily Return
        df_ind['Daily_Return'] = df_ind['Close'].pct_change()
        
        # Append to DB using DuckDB Appender / SQL
        con.execute("INSERT INTO Companies SELECT * FROM df_comp")
        con.execute("INSERT INTO PriceHistory SELECT * FROM df_h")
        if not df_fund.empty:
            con.execute("INSERT INTO Fundamentals SELECT * FROM df_fund")
            
        try:
            con.execute("INSERT INTO TechnicalIndicators SELECT * FROM df_ind")
        except Exception as ce:
            try:
                # Create the table on the first successful ticker
                con.execute("CREATE TABLE TechnicalIndicators AS SELECT * FROM df_ind LIMIT 0")
                con.execute("INSERT INTO TechnicalIndicators SELECT * FROM df_ind")
            except Exception as e:
                print("Failed to create TechnicalIndicators:", e)
                raise
        
        return True
    except Exception as e:
        print(f"Error processing {symbol}: {e}")
        return False

def main():
    if os.path.exists(DB_NAME):
        os.remove(DB_NAME)
        
    # We may hit rate limits, so we will use a small sleep within loop if needed
    print(f"Connecting to DuckDB: {DB_NAME}")
    con = duckdb.connect(DB_NAME)
    init_db(con)
    
    tickers = get_sp500_tickers()
    
    print("Starting data ingestion loop...")
    count = 0
    start_time = time.time()
    
    for idx, symbol in enumerate(tickers):
        success = process_ticker(symbol, con)
        if success:
            count += 1
            
        if (idx + 1) % 10 == 0:
            size_mb = os.path.getsize(DB_NAME) / (1024 * 1024)
            elapsed = time.time() - start_time
            print(f"[{idx + 1}/{len(tickers)}] DB Size: {size_mb:.2f} MB - Time Elapsed: {elapsed:.1f}s")
            
        time.sleep(0.1) # rate limit politeness
            
    print("Ingestion complete!")
    print(f"Final DB Size: {os.path.getsize(DB_NAME) / (1024 * 1024 * 1024):.2f} GB")
    
    # Export Tables to Parquet
    print("Exporting to uncompressed Parquet...")
    tables = ['Companies', 'PriceHistory', 'Fundamentals', 'TechnicalIndicators']
    for t in tables:
        con.execute(f"COPY (SELECT * FROM {t}) TO '{t.lower()}.parquet' (FORMAT PARQUET, COMPRESSION UNCOMPRESSED)")
        p_size = os.path.getsize(f"{t.lower()}.parquet") / (1024 * 1024)
        print(f"Exported {t} ({p_size:.2f} MB)")
        
    print("\nVerifying Data:")
    res = con.execute("SELECT count(*) FROM PriceHistory").fetchone()
    print(f"Total Rows in PriceHistory: {res[0]}")
    
    print("Sample Technical Indicators (AAPL):")
    sample = con.execute("SELECT * FROM TechnicalIndicators WHERE symbol='AAPL' LIMIT 5").fetchdf()
    print(sample)
    
    con.close()

if __name__ == "__main__":
    main()
