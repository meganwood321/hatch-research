"""
Hatch Research - Stock research tool for NASDAQ & NYSE.
Scores stocks on fundamentals and helps narrow down purchasing decisions.
"""

import os
import json
import time
import threading
import webbrowser
from datetime import datetime, timedelta
from pathlib import Path

import yfinance as yf
import requests
from flask import Flask, render_template, request, jsonify

# -- Configuration ----------------------------------------------------------

CACHE_DIR = Path('cache')
CACHE_DIR.mkdir(exist_ok=True)
CACHE_HOURS = 24  # How long cached data stays valid
FINNHUB_KEY_FILE = 'finnhub_key.txt'

# Hatch eligibility criteria
MIN_MARKET_CAP = 300_000_000   # $300M USD
MIN_AVG_VOLUME = 100_000       # 100k shares/day

# Rate limiting for yfinance
FETCH_DELAY = 0.5  # seconds between yfinance calls

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
    """Save stock data to cache."""
    data['_cached_at'] = datetime.now().isoformat()
    cache_path(ticker).write_text(json.dumps(data, default=str))


# -- Data Fetching ----------------------------------------------------------

def fetch_stock_data(ticker):
    """Fetch comprehensive stock data via yfinance."""
    cached = get_cached(ticker)
    if cached:
        return cached

    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        if not info or info.get('quoteType') not in ('EQUITY', None) or not info.get('marketCap'):
            return None

        # Pull analyst recommendations
        analyst_rec = {'strongBuy': 0, 'buy': 0, 'hold': 0, 'sell': 0, 'strongSell': 0}
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
        print(f"  Error fetching {ticker}: {e}")
        return None


# -- Ticker Lists -----------------------------------------------------------

def get_finnhub_key():
    """Load Finnhub API key from file."""
    if os.path.exists(FINNHUB_KEY_FILE):
        return Path(FINNHUB_KEY_FILE).read_text().strip()
    return None


def fetch_exchange_tickers(exchange='US'):
    """Fetch ticker list from Finnhub."""
    key = get_finnhub_key()
    if not key:
        return get_fallback_tickers()

    try:
        url = f"https://finnhub.io/api/v1/stock/symbol?exchange={exchange}&token={key}"
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        symbols = resp.json()

        # Filter to common stocks only (not warrants, preferred, etc.)
        tickers = []
        for s in symbols:
            sym = s.get('symbol', '')
            stype = s.get('type', '')
            # Skip symbols with dots (preferred shares), warrants, etc.
            if '.' in sym or '-' in sym or len(sym) > 5:
                continue
            if stype in ('Common Stock', 'EQS', ''):
                tickers.append(sym)

        return sorted(set(tickers))

    except Exception as e:
        print(f"  Finnhub error: {e}, using fallback list")
        return get_fallback_tickers()


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

def score_stock(data):
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
    return {
        'total': total,
        'categories': scores,
        'details': details
    }


# -- Background Scanner ----------------------------------------------------

def run_scan(exchange_filter=None):
    """Run a full exchange scan in the background."""
    global scan_state
    scan_state['running'] = True
    scan_state['error'] = None
    scan_state['results'] = []
    scan_state['started'] = datetime.now().isoformat()
    scan_state['finished'] = None
    scan_state['exchange'] = exchange_filter or 'ALL'

    try:
        # Get all tickers
        all_tickers = fetch_exchange_tickers('US')
        scan_state['total'] = len(all_tickers)
        scan_state['processed'] = 0

        scored_stocks = []

        for i, ticker in enumerate(all_tickers):
            if not scan_state['running']:
                break  # Allow cancellation

            scan_state['processed'] = i + 1

            try:
                data = fetch_stock_data(ticker)
                if data is None:
                    continue

                # Hatch eligibility filter
                mcap = data.get('marketCap', 0)
                vol = data.get('avgVolume', 0)
                if mcap < MIN_MARKET_CAP or vol < MIN_AVG_VOLUME:
                    continue

                # Exchange filter
                if exchange_filter:
                    exch = data.get('exchange', '').upper()
                    if exchange_filter == 'NASDAQ' and 'NAS' not in exch and 'NMS' not in exch and 'NGM' not in exch:
                        continue
                    if exchange_filter == 'NYSE' and 'NYS' not in exch and 'NYQ' not in exch:
                        continue

                score = score_stock(data)
                data['score'] = score
                scored_stocks.append(data)

            except Exception as e:
                print(f"  Scan error for {ticker}: {e}")

            # Rate limiting
            if not get_cached(ticker):
                time.sleep(FETCH_DELAY)

        # Sort by score descending
        scored_stocks.sort(key=lambda x: x.get('score', {}).get('total', 0), reverse=True)
        scan_state['results'] = scored_stocks

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
            stock_data['score'] = score_stock(stock_data)
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
            stock_data['score'] = score_stock(stock_data)
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


# -- Main -------------------------------------------------------------------

if __name__ == '__main__':
    print("\n+==========================================+")
    print("|       Hatch Research v1.0                |")
    print("+==========================================+")
    print("|  Opening browser at http://localhost:5001 |")
    print("|  Press Ctrl+C to stop                    |")
    print("+==========================================+\n")

    # Check for Finnhub key
    if get_finnhub_key():
        print("  Finnhub API key loaded (full exchange scanning enabled)")
    else:
        print("  No Finnhub key found (using fallback ticker list)")
        print("  To enable full scanning, save your free key to finnhub_key.txt")

    print(f"  Cache directory: {CACHE_DIR.absolute()}")
    cached_count = len(list(CACHE_DIR.glob('*.json')))
    if cached_count:
        print(f"  Cached stocks: {cached_count}")
    print()

    webbrowser.open('http://localhost:5001')
    app.run(debug=False, port=5001)
