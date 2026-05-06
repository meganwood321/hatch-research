"""
Hatch Research - Stock research tool for NASDAQ & NYSE.
Scores stocks on fundamentals and helps narrow down purchasing decisions.
"""

import os
import json
import time
import threading
import webbrowser
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

import yfinance as yf
from news_sentiment import analyze_ticker_news
import requests
from flask import Flask, render_template, request, jsonify

# -- Configuration ----------------------------------------------------------

CACHE_DIR = Path('cache')
CACHE_DIR.mkdir(exist_ok=True)
CACHE_HOURS = 24  # How long cached data stays valid
FINNHUB_KEY_FILE = 'finnhub_key.txt'
CURATED_TICKERS_FILE = CACHE_DIR / 'curated_tickers.json'
CURATED_TTL_DAYS = 7
INELIGIBLE_FILE = CACHE_DIR / 'ineligible.json'
INELIGIBLE_TTL_DAYS = 30

# Hatch eligibility criteria
MIN_MARKET_CAP = 300_000_000   # $300M USD
MIN_AVG_VOLUME = 100_000       # 100k shares/day

# Rate limiting for yfinance
FETCH_DELAY = 0.5  # seconds between yfinance calls (lookup route)
SCAN_WORKERS = 5   # Parallel workers during exchange scan (10 triggers Yahoo rate limits)
RETRY_ATTEMPTS = 3 # Retries on rate-limit errors
RETRY_DELAY = 5    # Seconds to wait between retries

app = Flask(__name__)

# Background scan state
scan_state = {
    'running': False,
    'exchange': '',
    'total': 0,
    'processed': 0,
    'results': [],
    'error': None,
    'started': None,
    'finished': None
}


# -- Caching ----------------------------------------------------------------

def cache_path(ticker):
    return CACHE_DIR / f"{ticker.upper()}.json"


def get_cached(ticker):
    """Load cached stock data if fresh enough."""
    path = cache_path(ticker)
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    cached_time = datetime.fromisoformat(data.get('_cached_at', '2000-01-01'))
    if datetime.now() - cached_time > timedelta(hours=CACHE_HOURS):
        return None
    return data


def save_cache(ticker, data):
    """Save stock data to cache with change tracking."""
    # Store previous score before updating
    if 'score' in data:
        data['_previous_score'] = data.get('score')

    data['_cached_at'] = datetime.now().isoformat()
    cache_path(ticker).write_text(json.dumps(data, default=str))


# -- Data Fetching ----------------------------------------------------------

def fetch_stock_data(ticker, skip_recommendations=False):
    """Fetch comprehensive stock data via yfinance.

    Args:
        ticker: Stock ticker symbol.
        skip_recommendations: If True, skip the extra HTTP call for analyst
            recommendations. Used during bulk scans to roughly halve fetch time;
            the lookup/compare routes fetch full data.
    """
    cached = get_cached(ticker)
    if cached:
        return cached

    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            if not info or info.get('quoteType') not in ('EQUITY', None) or not info.get('marketCap'):
                return None

            # Pull analyst recommendations (skipped during scans for speed)
            analyst_rec = {'strongBuy': 0, 'buy': 0, 'hold': 0, 'sell': 0, 'strongSell': 0}
            if not skip_recommendations:
                try:
                    recs = stock.recommendations
                    if recs is not None and len(recs) > 0:
                        latest = recs.iloc[-1] if hasattr(recs, 'iloc') else None
                        if latest is not None:
                            for key in analyst_rec:
                                if key in latest:
                                    analyst_rec[key] = int(latest[key])
                except Exception:
                    pass

            data = {
                'ticker': ticker.upper(),
                'name': info.get('longName') or info.get('shortName', ticker),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'exchange': info.get('exchange', ''),

                # Price
                'price': info.get('currentPrice') or info.get('regularMarketPrice', 0),
                'fiftyTwoWeekHigh': info.get('fiftyTwoWeekHigh', 0),
                'fiftyTwoWeekLow': info.get('fiftyTwoWeekLow', 0),
                'marketCap': info.get('marketCap', 0),
                'avgVolume': info.get('averageVolume', 0),

                # Valuation
                'trailingPE': info.get('trailingPE'),
                'forwardPE': info.get('forwardPE'),
                'pegRatio': info.get('pegRatio'),
                'priceToBook': info.get('priceToBook'),

                # Business Quality
                'revenueGrowth': info.get('revenueGrowth'),
                'profitMargins': info.get('profitMargins'),
                'returnOnEquity': info.get('returnOnEquity'),
                'freeCashflow': info.get('freeCashflow'),
                'totalRevenue': info.get('totalRevenue'),

                # Financial Strength
                'debtToEquity': info.get('debtToEquity'),
                'currentRatio': info.get('currentRatio'),

                # Dividends (informational)
                'dividendYield': info.get('dividendYield'),
                'payoutRatio': info.get('payoutRatio'),

                # Risk
                'beta': info.get('beta'),

                # Analyst
                'targetMeanPrice': info.get('targetMeanPrice'),
                'targetMedianPrice': info.get('targetMedianPrice'),
                'targetHighPrice': info.get('targetHighPrice'),
                'targetLowPrice': info.get('targetLowPrice'),
                'numberOfAnalystOpinions': info.get('numberOfAnalystOpinions', 0),
                'recommendationKey': info.get('recommendationKey', ''),
                'analystRec': analyst_rec,
            }

            save_cache(ticker, data)
            return data

        except Exception as e:
            err_str = str(e).lower()
            if ('rate' in err_str or 'too many' in err_str) and attempt < RETRY_ATTEMPTS:
                time.sleep(RETRY_DELAY * attempt)  # Progressive backoff
                continue
            if attempt == RETRY_ATTEMPTS and ('rate' in err_str or 'too many' in err_str):
                print(f"  Error fetching {ticker}: rate limited after {RETRY_ATTEMPTS} retries")
            else:
                print(f"  Error fetching {ticker}: {e}")
            return None


# -- Ticker Lists -----------------------------------------------------------

def get_finnhub_key():
    """Load Finnhub API key from file."""
    if os.path.exists(FINNHUB_KEY_FILE):
        return Path(FINNHUB_KEY_FILE).read_text().strip()
    return None


def get_curated_tickers():
    """Load curated ticker universe covering large, mid, and small caps.

    Pulls from six Wikipedia index pages and merges/dedupes:
      - S&P 500        (~500 large caps)
      - S&P 400        (~400 mid caps)
      - S&P 600        (~600 small caps)
      - Russell 1000   (~1,000 large/mid caps)
      - Russell 2000   (~2,000 small caps)

    Result is cached to disk for CURATED_TTL_DAYS. Falls back to an embedded
    list of well-known large caps if all network fetches fail.
    """
    # Try cache first
    if CURATED_TICKERS_FILE.exists():
        try:
            cached = json.loads(CURATED_TICKERS_FILE.read_text())
            cached_at = datetime.fromisoformat(cached.get('_cached_at', '2000-01-01'))
            if datetime.now() - cached_at < timedelta(days=CURATED_TTL_DAYS):
                print(f"  Using cached curated list: {len(cached.get('tickers', []))} tickers")
                return cached.get('tickers', [])
        except Exception:
            pass

    tickers = set()

    # Fetch Wikipedia pages with a real user-agent (Wikipedia 403s the default).
    # Then hand the HTML to pandas.read_html for table parsing.
    try:
        import pandas as pd
        from io import StringIO

        wiki_headers = {
            'User-Agent': 'HatchResearch/1.2 (https://github.com/meganwood321; personal research tool)'
        }

        def _fetch_wiki(url):
            resp = requests.get(url, headers=wiki_headers, timeout=30)
            resp.raise_for_status()
            return pd.read_html(StringIO(resp.text))

        def _extract_tickers(tables, ticker_cols=('symbol', 'ticker')):
            """Pull ticker symbols from whichever column matches."""
            found = set()
            for tbl in tables:
                cols = [str(c).lower() for c in tbl.columns]
                for target in ticker_cols:
                    matching = [c for c in tbl.columns if str(c).lower() == target]
                    if matching:
                        for sym in tbl[matching[0]].tolist():
                            s = str(sym).replace('.', '-').strip().upper()
                            if (s and s.lower() != 'nan'
                                    and 1 <= len(s) <= 6
                                    and s.replace('-', '').isalpha()):
                                found.add(s)
                        return found  # Found the right table, stop
            return found

        # Each source: (label, URL)
        sources = [
            ('S&P 500',     'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'),
            ('S&P 400',     'https://en.wikipedia.org/wiki/List_of_S%26P_400_companies'),
            ('S&P 600',     'https://en.wikipedia.org/wiki/List_of_S%26P_600_companies'),
            ('Russell 1000', 'https://en.wikipedia.org/wiki/Russell_1000_Index'),
            ('Russell 2000', 'https://en.wikipedia.org/wiki/Russell_2000_Index'),
        ]

        for label, url in sources:
            try:
                tables = _fetch_wiki(url)
                before = len(tickers)
                found = _extract_tickers(tables)
                tickers.update(found)
                added = len(tickers) - before
                print(f"  {label}: +{added} new tickers (total: {len(tickers)})")
            except Exception as e:
                print(f"  {label} fetch failed: {e}")

    except Exception as e:
        print(f"  pandas.read_html unavailable: {e}")

    # Fallback to embedded list if network fetch gave us nothing useful
    if len(tickers) < 100:
        print("  Falling back to embedded large-cap list")
        tickers.update(get_fallback_tickers())

    result = sorted(tickers)
    print(f"  Total curated universe: {len(result)} tickers")

    try:
        CURATED_TICKERS_FILE.write_text(json.dumps({
            '_cached_at': datetime.now().isoformat(),
            'tickers': result
        }))
    except Exception:
        pass
    return result


def load_ineligible():
    """Load the set of tickers previously found to be Hatch-ineligible.

    These get skipped on subsequent scans to avoid re-fetching stocks we
    already know are too small / illiquid. Expires after INELIGIBLE_TTL_DAYS
    so companies that have grown get re-checked periodically.
    """
    if not INELIGIBLE_FILE.exists():
        return set()
    try:
        data = json.loads(INELIGIBLE_FILE.read_text())
        cached_at = datetime.fromisoformat(data.get('_cached_at', '2000-01-01'))
        if datetime.now() - cached_at > timedelta(days=INELIGIBLE_TTL_DAYS):
            return set()
        return set(data.get('tickers', []))
    except Exception:
        return set()


def save_ineligible(tickers):
    """Persist the known-ineligible set."""
    try:
        INELIGIBLE_FILE.write_text(json.dumps({
            '_cached_at': datetime.now().isoformat(),
            'tickers': sorted(tickers)
        }))
    except Exception as e:
        print(f"  Failed to save ineligible cache: {e}")


def get_fallback_tickers():
    """Fallback: well-known large/mid cap tickers."""
    return [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B',
        'JPM', 'JNJ', 'V', 'UNH', 'HD', 'PG', 'MA', 'DIS', 'ADBE', 'CRM',
        'NFLX', 'COST', 'PEP', 'KO', 'ABBV', 'MRK', 'TMO', 'ACN', 'LIN',
        'MCD', 'NKE', 'TXN', 'WMT', 'AMD', 'INTC', 'QCOM', 'PYPL', 'LOW',
        'AMGN', 'SBUX', 'GS', 'BA', 'CAT', 'MMM', 'IBM', 'GE', 'F', 'GM',
        'XOM', 'CVX', 'COP', 'SLB', 'PFE', 'LLY', 'BMY', 'GILD', 'ISRG',
        'NOW', 'UBER', 'SQ', 'SHOP', 'SNOW', 'PLTR', 'CRWD', 'ZS', 'NET',
        'DDOG', 'PANW', 'FTNT', 'ABNB', 'RIVN', 'LCID', 'COIN', 'MARA',
        'SOFI', 'HOOD', 'RBLX', 'SNAP', 'PINS', 'TTD', 'ROKU', 'ZM',
    ]


# -- Scoring Engine ---------------------------------------------------------

def score_stock(data, previous_score=None):
    """Score a stock on a 0-100 scale based on fundamentals."""
    scores = {
        'business_quality': 0,
        'financial_strength': 0,
        'valuation': 0,
        'market_sentiment': 0,
    }
    details = {}

    # === BUSINESS QUALITY (35 pts max) ===

    # Free cash flow (12 pts)
    fcf = data.get('freeCashflow')
    revenue = data.get('totalRevenue')
    if fcf is not None and revenue and revenue > 0:
        fcf_margin = fcf / revenue
        if fcf_margin > 0.20:
            pts = 12
        elif fcf_margin > 0.10:
            pts = 9
        elif fcf_margin > 0.05:
            pts = 6
        elif fcf_margin > 0:
            pts = 3
        else:
            pts = 0
        scores['business_quality'] += pts
        details['fcf'] = {'value': fcf_margin, 'pts': pts, 'max': 12}

    # ROE (12 pts)
    roe = data.get('returnOnEquity')
    if roe is not None:
        if roe > 0.25:
            pts = 12
        elif roe > 0.15:
            pts = 9
        elif roe > 0.10:
            pts = 6
        elif roe > 0.05:
            pts = 3
        else:
            pts = 0
        scores['business_quality'] += pts
        details['roe'] = {'value': roe, 'pts': pts, 'max': 12}

    # Revenue growth (8 pts)
    rev_growth = data.get('revenueGrowth')
    if rev_growth is not None:
        if rev_growth > 0.25:
            pts = 8
        elif rev_growth > 0.10:
            pts = 6
        elif rev_growth > 0.05:
            pts = 4
        elif rev_growth > 0:
            pts = 2
        else:
            pts = 0
        scores['business_quality'] += pts
        details['revGrowth'] = {'value': rev_growth, 'pts': pts, 'max': 8}

    # Net profit margin (3 pts)
    margin = data.get('profitMargins')
    if margin is not None:
        if margin > 0.20:
            pts = 3
        elif margin > 0.10:
            pts = 2
        elif margin > 0:
            pts = 1
        else:
            pts = 0
        scores['business_quality'] += pts
        details['margin'] = {'value': margin, 'pts': pts, 'max': 3}

    # === FINANCIAL STRENGTH (20 pts max) ===

    # Debt-to-Equity (8 pts)
    de = data.get('debtToEquity')
    if de is not None:
        if de < 30:
            pts = 8
        elif de < 60:
            pts = 6
        elif de < 100:
            pts = 4
        elif de < 200:
            pts = 2
        else:
            pts = 0
        scores['financial_strength'] += pts
        details['debtEquity'] = {'value': de, 'pts': pts, 'max': 8}

    # Current ratio (6 pts)
    cr = data.get('currentRatio')
    if cr is not None:
        if cr > 2.0:
            pts = 6
        elif cr > 1.5:
            pts = 5
        elif cr > 1.0:
            pts = 3
        elif cr > 0.8:
            pts = 1
        else:
            pts = 0
        scores['financial_strength'] += pts
        details['currentRatio'] = {'value': cr, 'pts': pts, 'max': 6}

    # FCF positive (6 pts)
    if fcf is not None:
        pts = 6 if fcf > 0 else 0
        scores['financial_strength'] += pts
        details['fcfPositive'] = {'value': fcf > 0, 'pts': pts, 'max': 6}

    # === VALUATION (25 pts max) ===

    # PEG ratio (10 pts)
    peg = data.get('pegRatio')
    if peg is not None and peg > 0:
        if peg < 0.5:
            pts = 10
        elif peg < 1.0:
            pts = 8
        elif peg < 1.5:
            pts = 5
        elif peg < 2.0:
            pts = 3
        else:
            pts = 0
        scores['valuation'] += pts
        details['peg'] = {'value': peg, 'pts': pts, 'max': 10}

    # Forward P/E (8 pts)
    fpe = data.get('forwardPE')
    if fpe is not None and fpe > 0:
        if fpe < 10:
            pts = 8
        elif fpe < 15:
            pts = 6
        elif fpe < 20:
            pts = 4
        elif fpe < 30:
            pts = 2
        else:
            pts = 0
        scores['valuation'] += pts
        details['forwardPE'] = {'value': fpe, 'pts': pts, 'max': 8}

    # Price/Book (4 pts)
    pb = data.get('priceToBook')
    if pb is not None and pb > 0:
        if pb < 1.5:
            pts = 4
        elif pb < 3.0:
            pts = 3
        elif pb < 5.0:
            pts = 2
        else:
            pts = 0
        scores['valuation'] += pts
        details['priceBook'] = {'value': pb, 'pts': pts, 'max': 4}

    # P/E ratio (3 pts)
    pe = data.get('trailingPE')
    if pe is not None and pe > 0:
        if pe < 12:
            pts = 3
        elif pe < 18:
            pts = 2
        elif pe < 25:
            pts = 1
        else:
            pts = 0
        scores['valuation'] += pts
        details['pe'] = {'value': pe, 'pts': pts, 'max': 3}

    # === MARKET SENTIMENT (20 pts max) ===

    # Analyst consensus (8 pts)
    rec = data.get('analystRec', {})
    total_rec = sum(rec.values())
    if total_rec > 0:
        buy_pct = (rec.get('strongBuy', 0) + rec.get('buy', 0)) / total_rec
        if buy_pct > 0.70:
            pts = 8
        elif buy_pct > 0.50:
            pts = 6
        elif buy_pct > 0.30:
            pts = 3
        else:
            pts = 0
        scores['market_sentiment'] += pts
        details['analystBuyPct'] = {'value': buy_pct, 'pts': pts, 'max': 8}

    # Price vs analyst target (6 pts)
    price = data.get('price', 0)
    target = data.get('targetMeanPrice')
    if target and price and price > 0:
        upside = (target - price) / price
        if upside > 0.30:
            pts = 6
        elif upside > 0.15:
            pts = 4
        elif upside > 0.05:
            pts = 2
        elif upside > 0:
            pts = 1
        else:
            pts = 0
        scores['market_sentiment'] += pts
        details['upside'] = {'value': upside, 'pts': pts, 'max': 6}

    # 52-week position (6 pts) — closer to low = more value potential
    high = data.get('fiftyTwoWeekHigh', 0)
    low = data.get('fiftyTwoWeekLow', 0)
    if high > low > 0 and price > 0:
        position = (price - low) / (high - low)  # 0 = at low, 1 = at high
        if position < 0.30:
            pts = 6
        elif position < 0.50:
            pts = 4
        elif position < 0.70:
            pts = 2
        else:
            pts = 0
        scores['market_sentiment'] += pts
        details['52wkPosition'] = {'value': position, 'pts': pts, 'max': 6}

    total = sum(scores.values())

    # Calculate changes if previous score exists
    changes = None
    if previous_score:
        changes = {
            'total': total - previous_score.get('total', 0),
            'categories': {
                cat: scores[cat] - previous_score.get('categories', {}).get(cat, 0)
                for cat in scores.keys()
            },
            'details': {}
        }

        # Calculate detailed factor changes
        prev_details = previous_score.get('details', {})
        for factor, current in details.items():
            prev = prev_details.get(factor)
            if prev:
                changes['details'][factor] = {
                    'value': current['value'] - prev.get('value', 0),
                    'pts': current['pts'] - prev.get('pts', 0)
                }

    return {
        'total': total,
        'categories': scores,
        'details': details,
        'changes': changes
    }


# -- Background Scanner ----------------------------------------------------

def run_scan(exchange_filter=None):
    """Run a full exchange scan in the background.

    Uses a curated S&P 500 + Russell 1000 universe (not the whole exchange),
    a persistent ineligibility cache, and a thread pool to fetch tickers in
    parallel. Typical scan time: a few minutes on first run, under a minute
    on subsequent runs once the cache is warm.
    """
    global scan_state
    scan_state['running'] = True
    scan_state['error'] = None
    scan_state['results'] = []
    scan_state['started'] = datetime.now().isoformat()
    scan_state['finished'] = None
    scan_state['exchange'] = exchange_filter or 'ALL'

    state_lock = threading.Lock()

    try:
        all_tickers = get_curated_tickers()
        ineligible = load_ineligible()

        # Skip tickers we already know are too small / illiquid
        tickers_to_scan = [t for t in all_tickers if t not in ineligible]
        skipped = len(all_tickers) - len(tickers_to_scan)
        if skipped:
            print(f"  Skipping {skipped} tickers from ineligibility cache")
        print(f"  Scanning {len(tickers_to_scan)} tickers with {SCAN_WORKERS} workers")

        scan_state['total'] = len(tickers_to_scan)
        scan_state['processed'] = 0

        scored_stocks = []
        new_ineligible = set()

        def process_ticker(ticker):
            """Fetch + score one ticker. Returns (status, payload)."""
            if not scan_state['running']:
                return ('cancelled', ticker)
            try:
                data = fetch_stock_data(ticker, skip_recommendations=True)
                if data is None:
                    return ('no_data', ticker)

                mcap = data.get('marketCap', 0)
                vol = data.get('avgVolume', 0)
                if mcap < MIN_MARKET_CAP or vol < MIN_AVG_VOLUME:
                    return ('ineligible', ticker)

                if exchange_filter:
                    exch = data.get('exchange', '').upper()
                    if exchange_filter == 'NASDAQ' and 'NAS' not in exch and 'NMS' not in exch and 'NGM' not in exch:
                        return ('wrong_exchange', ticker)
                    if exchange_filter == 'NYSE' and 'NYS' not in exch and 'NYQ' not in exch:
                        return ('wrong_exchange', ticker)

                score = score_stock(data, data.get('score'))
                data['score'] = score
                return ('ok', data)
            except Exception as e:
                print(f"  Scan error for {ticker}: {e}")
                return ('error', ticker)

        with ThreadPoolExecutor(max_workers=SCAN_WORKERS) as executor:
            futures = {executor.submit(process_ticker, t): t for t in tickers_to_scan}
            for future in as_completed(futures):
                if not scan_state['running']:
                    # User cancelled — stop waiting on remaining futures
                    executor.shutdown(wait=False, cancel_futures=True)
                    break
                try:
                    status, payload = future.result()
                except Exception as e:
                    print(f"  Future error: {e}")
                    status, payload = 'error', None

                with state_lock:
                    scan_state['processed'] += 1

                if status == 'ok':
                    scored_stocks.append(payload)
                elif status == 'ineligible':
                    new_ineligible.add(payload)

        # Sort by score descending
        scored_stocks.sort(key=lambda x: x.get('score', {}).get('total', 0), reverse=True)
        scan_state['results'] = scored_stocks

        # Persist the updated ineligibility cache so future scans are faster
        if new_ineligible:
            save_ineligible(ineligible | new_ineligible)
            print(f"  Added {len(new_ineligible)} tickers to ineligibility cache")

    except Exception as e:
        scan_state['error'] = str(e)
    finally:
        scan_state['running'] = False
        scan_state['finished'] = datetime.now().isoformat()


# -- Flask Routes -----------------------------------------------------------

@app.route('/')
def index():
    return render_template('dashboard.html')


@app.route('/lookup', methods=['POST'])
def lookup():
    """Look up one or more tickers."""
    data = request.get_json()
    tickers = [t.strip().upper() for t in data.get('tickers', '').split(',') if t.strip()]

    if not tickers:
        return jsonify({'status': 'error', 'message': 'No tickers provided'})

    results = []
    for ticker in tickers[:20]:  # Max 20 at once
        stock_data = fetch_stock_data(ticker)
        if stock_data:
            stock_data['score'] = score_stock(stock_data, stock_data.get('score'))
            results.append(stock_data)
        else:
            results.append({'ticker': ticker, 'error': 'Not found or no data'})
        time.sleep(FETCH_DELAY)

    return jsonify({'status': 'ok', 'results': results})


@app.route('/scan', methods=['POST'])
def start_scan():
    """Start a background exchange scan."""
    if scan_state['running']:
        return jsonify({'status': 'error', 'message': 'Scan already running'})

    data = request.get_json() or {}
    exchange = data.get('exchange')  # 'NASDAQ', 'NYSE', or None for both

    thread = threading.Thread(target=run_scan, args=(exchange,), daemon=True)
    thread.start()

    return jsonify({'status': 'ok', 'message': f'Scan started for {exchange or "ALL"}'})


@app.route('/scan/stop', methods=['POST'])
def stop_scan():
    """Stop a running scan."""
    scan_state['running'] = False
    return jsonify({'status': 'ok'})


@app.route('/scan/progress')
def scan_progress():
    """Get current scan progress."""
    return jsonify({
        'running': scan_state['running'],
        'exchange': scan_state['exchange'],
        'total': scan_state['total'],
        'processed': scan_state['processed'],
        'resultCount': len(scan_state['results']),
        'error': scan_state['error'],
        'started': scan_state['started'],
        'finished': scan_state['finished']
    })


@app.route('/scan/results')
def scan_results():
    """Get scan results (top N)."""
    limit = request.args.get('limit', 50, type=int)
    results = scan_state['results'][:limit]
    return jsonify({'status': 'ok', 'results': results, 'total': len(scan_state['results'])})


@app.route('/compare', methods=['POST'])
def compare():
    """Get data for multiple tickers for comparison."""
    data = request.get_json()
    tickers = data.get('tickers', [])

    results = []
    for ticker in tickers[:10]:
        stock_data = fetch_stock_data(ticker)
        if stock_data:
            stock_data['score'] = score_stock(stock_data, stock_data.get('score'))
            results.append(stock_data)

    return jsonify({'status': 'ok', 'results': results})


@app.route('/export')
def export_csv():
    """Export scan results as CSV."""
    import csv
    from io import StringIO
    from flask import Response

    results = scan_state.get('results', [])
    if not results:
        return jsonify({'status': 'error', 'message': 'No results to export'})

    output = StringIO()
    fields = ['ticker', 'name', 'sector', 'price', 'marketCap', 'trailingPE',
              'forwardPE', 'pegRatio', 'priceToBook', 'returnOnEquity',
              'revenueGrowth', 'profitMargins', 'freeCashflow', 'debtToEquity',
              'currentRatio', 'dividendYield', 'beta', 'targetMeanPrice',
              'fiftyTwoWeekHigh', 'fiftyTwoWeekLow']

    writer = csv.DictWriter(output, fieldnames=fields + ['score'], extrasaction='ignore')
    writer.writeheader()
    for r in results:
        row = {k: r.get(k, '') for k in fields}
        row['score'] = r.get('score', {}).get('total', 0)
        writer.writerow(row)

    return Response(
        output.getvalue(),
        mimetype='text/csv',
        headers={'Content-Disposition': f'attachment; filename=hatch_research_{datetime.now().strftime("%Y%m%d")}.csv'}
    )


@app.route('/api/news-sentiment/<ticker>')
def news_sentiment(ticker):
    """Get news sentiment analysis for a ticker."""
    result = analyze_ticker_news(ticker.upper())
    return jsonify(result)


@app.route('/api/news-sentiment/<ticker>/refresh')
def news_sentiment_refresh(ticker):
    """Force-refresh news sentiment (bypass cache)."""
    result = analyze_ticker_news(ticker.upper(), force_refresh=True)
    return jsonify(result)


# -- Main -------------------------------------------------------------------

if __name__ == '__main__':
    print("\n+==========================================+")
    print("|       Hatch Research v1.0                |")
    print("+==========================================+")
    print("|  Opening browser at http://localhost:5001 |")
    print("|  Press Ctrl+C to stop                    |")
    print("+==========================================+\n")

    print(f"  Cache directory: {CACHE_DIR.absolute()}")
    cached_count = len(list(CACHE_DIR.glob('*.json')))
    if cached_count:
        print(f"  Cached stocks: {cached_count}")
    if CURATED_TICKERS_FILE.exists():
        try:
            ct = json.loads(CURATED_TICKERS_FILE.read_text())
            print(f"  Curated ticker list: {len(ct.get('tickers', []))} stocks (S&P 500 + Russell 1000)")
        except Exception:
            pass
    ineligible = load_ineligible()
    if ineligible:
        print(f"  Ineligibility cache: {len(ineligible)} tickers will be skipped on next scan")
    print()

    webbrowser.open('http://localhost:5001')
    app.run(debug=False, port=5001)
