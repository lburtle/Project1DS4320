# DS 4320 Project 1: Stock Market Probabilistic Forecasting Model

## Problem Definition

--------------------
## Domain Exposition

----------------------
## Data Creation

All data in this project is sourced from live, publicly available financial data providers.

The list of companies to analyze was obtained by scraping the S&P 500 constituent table from Wikipedia (`https://en.wikipedia.org/wiki/List_of_S%26P_500_companies`). Wikipedia was chosen over direct financial-aggregator scraping because aggregators (e.g., Slickcharts, Macrotrends) frequently return HTTP 403 errors when accessed through a script.

The core financial data was retrieved using the yfinance library, which wraps Yahoo Finance's unofficial API. For each ticker, three categories of data were pulled ticker.info, ticker.history(period='max', auto_adjust=True) and the financial statements — `income_stmt`, `balance_sheet`, `cash_flow`, and their quarterly counterparts.

Unstructured text data was collected from Yahoo Finance's public RSS feed (`https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}`), which returns the most recent ~30 headlines per ticker as XML. The feed is unauthenticated and publicly accessible.

Technical Indicators were computed entirely from the price history data using pandas `.rolling()` and `.ewm()` methods.

| File | Description | Link |
|------|-------------|------|
| `data.py` | Main ETL pipeline: ticker ingestion, yfinance extraction, technical indicator computation, fundamentals stacking, RSS news fetch, DuckDB load, Parquet export, and verification queries | [data.py](financial_pipeline.py) |

### Bias:
**Survivorship Bias:** The tickers listed are the current S&P 500 constituents. Companies that were once in the index but were later removed due to bankruptcy, 
merger, or sustained underperformance are absent from the dataset. This means the historical price data is implicitly drawn only from companies that survived 
long enough to remain index members, systematically overstating historical average returns and understating historical average volatility.

### Bias Mitigation:
The clearest mitigation is to supplement the dataset with delisted tickers (e.g., from CRSP or Compustat). In this project, the bias can be partially quantified
by filtering analyses to the subset of the price history that predates each company's index entry date. Any forecasting model should be evaluated with out-of-sample
data from a future index reconstitution to test whether returns generalize beyond survivors.

### Rationale:
Several commonly referenced S&P 500 CSV sources (e.g., from GitHub) return 403 errors when fetched programmatically. Wikipedia's table is publicly accessible, 
human-readable, and updated within days of index changes. The dot-to-hyphen substitution (`BRK.B → BRK-B`) is required because Yahoo Finance uses hyphens in 
ticker symbols while the S&P 500 official list uses dots.

---------------------
## Metadata

