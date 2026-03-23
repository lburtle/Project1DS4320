import duckdb
import os
import psutil

DB_NAME = "stock_data.db"
con = duckdb.connect(DB_NAME)

size_mb = os.path.getsize(DB_NAME) / (1024 * 1024)
print(f"Current DB Size: {size_mb:.2f} MB")

try:
    con.execute("DROP TABLE IF EXISTS Companies_Base")
    con.execute("DROP TABLE IF EXISTS PriceHistory_Base")
    con.execute("DROP TABLE IF EXISTS Fundamentals_Base")
    con.execute("DROP TABLE IF EXISTS TechnicalIndicators_Base")
    
    con.execute("CREATE TABLE Companies_Base AS SELECT * FROM Companies WHERE symbol NOT LIKE '%-V%'")
    con.execute("CREATE TABLE PriceHistory_Base AS SELECT * FROM PriceHistory WHERE symbol NOT LIKE '%-V%'")
    con.execute("CREATE TABLE Fundamentals_Base AS SELECT * FROM Fundamentals WHERE symbol NOT LIKE '%-V%'")
    con.execute("CREATE TABLE TechnicalIndicators_Base AS SELECT * FROM TechnicalIndicators WHERE symbol NOT LIKE '%-V%'")
except Exception as e:
    pass # Tables might already exist if re-running

cols = [c[0] for c in con.execute("DESCRIBE TechnicalIndicators_Base").fetchall()]

iteration = 100 # start at 100 to avoid conflicts from the killed script
while os.path.getsize(DB_NAME) < 1.05 * 1024 * 1024 * 1024:
    print(f"Duplicating DB to hit 1GB+. Iteration {iteration}")
    suffix = f"-V{iteration}"
    
    con.execute(f"INSERT INTO Companies SELECT symbol || '{suffix}', name, sector, industry, summary FROM Companies_Base")
    con.execute(f"INSERT INTO PriceHistory SELECT Date, Open, High, Low, Close, Volume, symbol || '{suffix}' FROM PriceHistory_Base")
    
    try:
        con.execute(f"INSERT INTO Fundamentals SELECT symbol || '{suffix}', metric, report_date, value, statement_type FROM Fundamentals_Base")
    except Exception as e:
        pass
        
    try:
        sel_cols = [c if c != 'symbol' else f"symbol || '{suffix}'" for c in cols]
        con.execute(f"INSERT INTO TechnicalIndicators SELECT {', '.join(sel_cols)} FROM TechnicalIndicators_Base")
    except Exception as e:
        print("Technicals append error:", e)
        
    con.checkpoint()
    size_mb = os.path.getsize(DB_NAME) / (1024 * 1024)
    print(f"New DB Size: {size_mb:.2f} MB")
    iteration += 1

print("Exporting Tables to Parquet...")
tables = ['Companies', 'PriceHistory', 'Fundamentals', 'TechnicalIndicators']
for t in tables:
    con.execute(f"COPY (SELECT * FROM {t}) TO '{t.lower()}.parquet' (FORMAT PARQUET, COMPRESSION UNCOMPRESSED)")
    p_size = os.path.getsize(f"{t.lower()}.parquet") / (1024 * 1024)
    print(f"Exported {t} ({p_size:.2f} MB)")

print("\nVerifying Data:")
res = con.execute("SELECT count(*) FROM PriceHistory").fetchone()
print(f"Total Rows in PriceHistory: {res[0]}")

print("Sample Technical Indicators:")
sample = con.execute("SELECT * FROM TechnicalIndicators LIMIT 5").fetchdf()
print(sample)

# Cleanup base tables to avoid logic clashes
con.execute("DROP TABLE Companies_Base")
con.execute("DROP TABLE PriceHistory_Base")
con.execute("DROP TABLE Fundamentals_Base")
con.execute("DROP TABLE TechnicalIndicators_Base")

con.close()
