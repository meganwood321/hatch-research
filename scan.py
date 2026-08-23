"""
Hatch Research Scanner
Run: python scan.py
Scans the top ~200 largest US stocks, scores them, saves results.
"""

import json
import time
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

import yfinance as yf
import requests

# -- Config -----------------------------------------------------------------

RESULTS_JSON = Path('scan_results.json')
RESULTS_CSV  = Path('scan_results.csv')
CACHE_DIR    = Path('cache')
CACHE_HOURS  = 24
WORKERS      = 8
RETRY        = 3
RETRY_DELAY  = 5

CACHE_DIR.mkdir(exist_ok=True)

# -- Ticker universe: S&P 100 + NASDAQ 100 (~180 unique large caps) ---------

FALLBACK_TICKERS = [
    # Mega caps
    'AAPL','MSFT','NVDA','AMZN','GOOGL','META','TSLA','BRK-B','AVGO','JPM',
    # Large caps
    'LLY','UNH','XOM','V','MA','COST','HD','PG','JNJ','ABBV',
    'MRK','WMT','NFLX','CRM','BAC','AMD','ORCL','CVX','KO','PEP',
    'TMO','MCD','ACN','ABT','CSCO','GE','ADBE','TXN','QCOM','DHR',
    'LIN','PM','INTU','AMGN','IBM','GS','CAT','ISRG','SPGI','BLK',
    'UBER','NOW','AXP','SYK','BKNG','PLD','T','RTX','DE','VRTX',
    'GILD','CB','REGN','MDT','MU','LRCX','ADI','CI','SO','DUK',
    'MMC','AON','SHW','ZTS','BSX','ETN','PGR','HCA','CME','NOC',
    'PANW','TJX','KLAC','SNPS','CDNS','MCO','ECL','APH','ITW','WM',
    'EMR','FDX','ICE','WELL','CTAS','FTNT','PSA','MSI','NKE','F',
    'GM','VLO','OXY','PSX','MPC','KMB','GD','TGT','LOW','USB',
    'WFC','C','MS','SCHW','BK','PNC','TFC','COF','ALLY','KEY',
    'INTC','HPQ','DELL','STX','WDC','AMAT','MCHP','MPWR','ENPH',
    'CRWD','DDOG','ZS','NET','SNOW','PLTR','COIN','ABNB','ROKU',
    'SHOP','SQ','PYPL','HOOD','SOFI','NU','AFRM','RBLX','SNAP',
    'PINS','TTD','TRADE','GTLB','MDB','CFLT','HCP','ZM','DOCU',
]

WIKI_HEADERS = {
    'User-Agent': 'HatchResearch/2.0 (https://github.com/meganwood321; personal research tool)'
}

def get_tickers():
    """Fetch S&P 500 from Wikipedia, fall back to hardcoded list."""
    tickers = set()
    try:
        import pandas as pd
        from io import StringIO

        def _wiki(url):
            r = requests.get(url, headers=WIKI_HEADERS, timeout=20)
            r.raise_for_status()
            return pd.read_html(StringIO(r.text))

        sources = [
            ('S&P 500',    'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'),
            ('NASDAQ 100', 'https://en.wikipedia.org/wiki/Nasdaq-100'),
        ]
        for label, url in sources:
            try:
                tables = _wiki(url)
                for tbl in tables:
                    for col in tbl.columns:
                        if str(col).lower() in ('symbol', 'ticker', 'ticker symbol'):
                            for sym in tbl[col].tolist():
                                s = str(sym).replace('.', '-').strip().upper()
                                if s and s.lower() != 'nan' and 1 <= len(s) <= 6:
                                    tickers.add(s)
                            break
                print(f"  {label}: loaded (total so far: {len(tickers)})")
            except Exception as e:
                print(f"  {label} fetch failed: {e}")

    except Exception as e:
        print(f"  pandas unavailable: {e}")

    if len(tickers) < 50:
        print("  Using built-in large-cap list")
        tickers.update(FALLBACK_TICKERS)

    return sorted(tickers)


# -- Cache ------------------------------------------------------------------

def cache_path(ticker):
    return CACHE_DIR / f"{ticker.upper()}.json"

def get_cached(ticker):
    p = cache_path(ticker)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text())
        cached_at = datetime.fromisoformat(data.get('_cached_at', '2000-01-01'))
        if datetime.now() - cached_at > timedelta(hours=CACHE_HOURS):
            return None
        return data
    except Exception:
        return None

def save_cache(ticker, data):
    data['_cached_at'] = datetime.now().isoformat()
    try:
        cache_path(ticker).write_text(json.dumps(data, default=str))
    except Exception:
        pass


# -- Fetch ------------------------------------------------------------------

def fetch(ticker):
    cached = get_cached(ticker)
    if cached:
        return cached

    for attempt in range(1, RETRY + 1):
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            if not info or not info.get('marketCap'):
                return None
            if info.get('quoteType') not in ('EQUITY', None):
                return None

            data = {
                'ticker':        ticker.upper(),
                'name':          info.get('longName') or info.get('shortName', ticker),
                'sector':        info.get('sector', ''),
                'industry':      info.get('industry', ''),
                'exchange':      info.get('exchange', ''),
                'price':         info.get('currentPrice') or info.get('regularMarketPrice', 0),
                'marketCap':     info.get('marketCap', 0),
                'avgVolume':     info.get('averageVolume', 0),
                'fiftyTwoWeekHigh': info.get('fiftyTwoWeekHigh', 0),
                'fiftyTwoWeekLow':  info.get('fiftyTwoWeekLow', 0),
                'trailingPE':    info.get('trailingPE'),
                'forwardPE':     info.get('forwardPE'),
                'pegRatio':      info.get('pegRatio'),
                'priceToBook':   info.get('priceToBook'),
                'revenueGrowth': info.get('revenueGrowth'),
                'profitMargins': info.get('profitMargins'),
                'returnOnEquity':info.get('returnOnEquity'),
                'freeCashflow':  info.get('freeCashflow'),
                'totalRevenue':  info.get('totalRevenue'),
                'debtToEquity':  info.get('debtToEquity'),
                'currentRatio':  info.get('currentRatio'),
                'dividendYield': info.get('dividendYield'),
                'beta':          info.get('beta'),
                'targetMeanPrice': info.get('targetMeanPrice'),
                'numberOfAnalystOpinions': info.get('numberOfAnalystOpinions', 0),
                'recommendationKey': info.get('recommendationKey', ''),
            }
            save_cache(ticker, data)
            return data

        except Exception as e:
            err = str(e).lower()
            if ('rate' in err or 'too many' in err) and attempt < RETRY:
                time.sleep(RETRY_DELAY * attempt)
                continue
            return None
    return None


# -- Scoring ----------------------------------------------------------------

MIN_MARKET_CAP = 300_000_000
MIN_AVG_VOLUME = 100_000

def score(data):
    s = 0

    # Business Quality (35)
    fcf = data.get('freeCashflow')
    rev = data.get('totalRevenue')
    if fcf is not None and rev and rev > 0:
        m = fcf / rev
        s += 12 if m > 0.20 else 9 if m > 0.10 else 6 if m > 0.05 else 3 if m > 0 else 0

    roe = data.get('returnOnEquity')
    if roe is not None:
        s += 12 if roe > 0.25 else 9 if roe > 0.15 else 6 if roe > 0.10 else 3 if roe > 0.05 else 0

    rg = data.get('revenueGrowth')
    if rg is not None:
        s += 8 if rg > 0.25 else 6 if rg > 0.10 else 4 if rg > 0.05 else 2 if rg > 0 else 0

    mg = data.get('profitMargins')
    if mg is not None:
        s += 3 if mg > 0.20 else 2 if mg > 0.10 else 1 if mg > 0 else 0

    # Financial Strength (20)
    de = data.get('debtToEquity')
    if de is not None:
        s += 8 if de < 30 else 6 if de < 60 else 4 if de < 100 else 2 if de < 200 else 0

    cr = data.get('currentRatio')
    if cr is not None:
        s += 6 if cr > 2.0 else 5 if cr > 1.5 else 3 if cr > 1.0 else 1 if cr > 0.8 else 0

    if fcf is not None:
        s += 6 if fcf > 0 else 0

    # Valuation (25)
    peg = data.get('pegRatio')
    if peg and peg > 0:
        s += 10 if peg < 0.5 else 8 if peg < 1.0 else 5 if peg < 1.5 else 3 if peg < 2.0 else 0

    fpe = data.get('forwardPE')
    if fpe and fpe > 0:
        s += 8 if fpe < 10 else 6 if fpe < 15 else 4 if fpe < 20 else 2 if fpe < 30 else 0

    pb = data.get('priceToBook')
    if pb and pb > 0:
        s += 4 if pb < 1.5 else 3 if pb < 3.0 else 2 if pb < 5.0 else 0

    pe = data.get('trailingPE')
    if pe and pe > 0:
        s += 3 if pe < 12 else 2 if pe < 18 else 1 if pe < 25 else 0

    # Market Sentiment (20) — analyst target + 52wk position only (no recs in scan)
    price = data.get('price', 0)
    target = data.get('targetMeanPrice')
    if target and price and price > 0:
        up = (target - price) / price
        s += 6 if up > 0.30 else 4 if up > 0.15 else 2 if up > 0.05 else 1 if up > 0 else 0

    high = data.get('fiftyTwoWeekHigh', 0)
    low  = data.get('fiftyTwoWeekLow', 0)
    if high > low > 0 and price > 0:
        pos = (price - low) / (high - low)
        s += 6 if pos < 0.30 else 4 if pos < 0.50 else 2 if pos < 0.70 else 0

    return s


# -- Main -------------------------------------------------------------------

def main():
    print("\n========================================")
    print("  Hatch Research Scanner")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("========================================\n")

    print("Loading ticker list...")
    tickers = get_tickers()
    print(f"Scanning {len(tickers)} tickers ({WORKERS} workers)...\n")

    results = []
    errors = 0
    ineligible = 0
    processed = 0

    start = time.time()

    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        futures = {executor.submit(fetch, t): t for t in tickers}
        for future in as_completed(futures):
            processed += 1
            ticker = futures[future]
            try:
                data = future.result()
                if data is None:
                    errors += 1
                elif data.get('marketCap', 0) < MIN_MARKET_CAP or data.get('avgVolume', 0) < MIN_AVG_VOLUME:
                    ineligible += 1
                else:
                    data['score'] = score(data)
                    results.append(data)
            except Exception:
                errors += 1

            # Progress every 25 tickers
            if processed % 25 == 0 or processed == len(tickers):
                elapsed = time.time() - start
                print(f"  {processed}/{len(tickers)}  |  {len(results)} eligible  |  {elapsed:.0f}s elapsed")

    results.sort(key=lambda x: x.get('score', 0), reverse=True)

    elapsed = time.time() - start
    print(f"\nDone in {elapsed:.0f}s — {len(results)} stocks scored, {ineligible} ineligible, {errors} errors\n")

    # Save JSON (for web app to load)
    RESULTS_JSON.write_text(json.dumps({
        'scanned_at': datetime.now().isoformat(),
        'ticker_count': len(tickers),
        'results': results
    }, default=str))
    print(f"Saved: {RESULTS_JSON}")

    # Save CSV
    fields = ['score','ticker','name','sector','price','marketCap','returnOnEquity',
              'revenueGrowth','profitMargins','freeCashflow','debtToEquity',
              'currentRatio','pegRatio','forwardPE','trailingPE','priceToBook',
              'targetMeanPrice','dividendYield','beta']
    with open(RESULTS_CSV, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(results)
    print(f"Saved: {RESULTS_CSV}")

    # Print top 20
    print("\n--- TOP 20 ---")
    print(f"{'#':<4} {'Score':<7} {'Ticker':<8} {'Name':<35} {'Sector'}")
    print("-" * 75)
    for i, r in enumerate(results[:20], 1):
        print(f"{i:<4} {r.get('score',0):<7} {r.get('ticker',''):<8} {(r.get('name','') or '')[:34]:<35} {r.get('sector','')}")

    print()


if __name__ == '__main__':
    main()
