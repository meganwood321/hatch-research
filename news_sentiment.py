"""
News Sentiment Engine for Hatch Research.

Fetches financial news articles via Finnhub and scores sentiment using FinBERT.
Provides per-article sentiment, daily trends, and source breakdowns.
"""

import json
import os
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import requests

# -- Configuration ----------------------------------------------------------

CACHE_DIR = Path('cache')
NEWS_CACHE_HOURS = 6
FINNHUB_KEY_FILE = 'finnhub_key.txt'
NEWS_LOOKBACK_DAYS = 14

# FinBERT model — downloaded on first use (~200MB), cached by Hugging Face
FINBERT_MODEL = 'ProsusAI/finbert'

# -- Finnhub Key ------------------------------------------------------------

def _get_finnhub_key():
    if os.path.exists(FINNHUB_KEY_FILE):
        return Path(FINNHUB_KEY_FILE).read_text().strip()
    return None


# -- Caching ----------------------------------------------------------------

def _news_cache_path(ticker):
    return CACHE_DIR / f"news_{ticker.upper()}.json"


def _get_cached_news(ticker):
    path = _news_cache_path(ticker)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        cached_at = datetime.fromisoformat(data.get('cached_at', '2000-01-01'))
        if datetime.now() - cached_at > timedelta(hours=NEWS_CACHE_HOURS):
            return None
        return data
    except Exception:
        return None


def _save_news_cache(ticker, data):
    data['cached_at'] = datetime.now().isoformat()
    try:
        _news_cache_path(ticker).write_text(json.dumps(data, default=str))
    except Exception as e:
        print(f"  Failed to cache news for {ticker}: {e}")


# -- FinBERT Sentiment Analyzer ---------------------------------------------

class SentimentAnalyzer:
    """Lazy-loading FinBERT wrapper. Model is loaded on first analyze() call."""

    def __init__(self):
        self._pipeline = None
        self._available = None

    def _load(self):
        if self._available is not None:
            return self._available
        try:
            from transformers import pipeline
            print("  Loading FinBERT model (first time may download ~200MB)...")
            self._pipeline = pipeline(
                'sentiment-analysis',
                model=FINBERT_MODEL,
                tokenizer=FINBERT_MODEL,
                device=-1,  # CPU
                top_k=None  # Return all class scores
            )
            self._available = True
            print("  FinBERT loaded successfully")
        except Exception as e:
            print(f"  FinBERT not available: {e}")
            self._available = False
        return self._available

    def analyze(self, text):
        """Analyze text sentiment.

        Returns:
            dict: {label: 'bullish'|'bearish'|'neutral', score: -1.0 to 1.0,
                   confidence: 0.0 to 1.0}
        """
        if not text or not text.strip():
            return {'label': 'neutral', 'score': 0.0, 'confidence': 0.0}

        if not self._load():
            return self._fallback_analyze(text)

        try:
            # Truncate to FinBERT's max token length (~512 tokens, ~380 words)
            truncated = ' '.join(text.split()[:350])
            results = self._pipeline(truncated)

            # results is a list of lists: [[{label, score}, ...]]
            scores_list = results[0] if results else []
            scores = {}
            for item in scores_list:
                scores[item['label'].lower()] = item['score']

            pos = scores.get('positive', 0)
            neg = scores.get('negative', 0)
            neu = scores.get('neutral', 0)

            # Score: -1 (bearish) to +1 (bullish)
            sentiment_score = pos - neg

            # Label based on highest probability
            if pos > neg and pos > neu:
                label = 'bullish'
            elif neg > pos and neg > neu:
                label = 'bearish'
            else:
                label = 'neutral'

            confidence = max(pos, neg, neu)

            return {
                'label': label,
                'score': round(sentiment_score, 3),
                'confidence': round(confidence, 3)
            }
        except Exception as e:
            print(f"  FinBERT error: {e}")
            return self._fallback_analyze(text)

    def _fallback_analyze(self, text):
        """Simple keyword-based fallback when FinBERT isn't available."""
        text_lower = text.lower()
        bullish = ['upgrade', 'outperform', 'buy', 'strong', 'growth', 'beat',
                    'surge', 'rally', 'bullish', 'record', 'positive', 'soar',
                    'profit', 'exceed', 'gain', 'optimistic', 'momentum']
        bearish = ['downgrade', 'underperform', 'sell', 'weak', 'decline',
                    'miss', 'drop', 'fall', 'bearish', 'loss', 'negative',
                    'cut', 'risk', 'concern', 'warning', 'plunge', 'crash']

        bull_count = sum(1 for w in bullish if w in text_lower)
        bear_count = sum(1 for w in bearish if w in text_lower)
        total = bull_count + bear_count

        if total == 0:
            return {'label': 'neutral', 'score': 0.0, 'confidence': 0.3}

        score = (bull_count - bear_count) / total
        if score > 0.15:
            label = 'bullish'
        elif score < -0.15:
            label = 'bearish'
        else:
            label = 'neutral'

        return {'label': label, 'score': round(score, 3), 'confidence': 0.4}


# Module-level singleton — lazy-loaded on first use
_analyzer = SentimentAnalyzer()


# -- Finnhub News Fetching --------------------------------------------------

def _is_relevant(ticker, company_name, headline, summary):
    """Check if an article actually mentions the ticker or company."""
    text = (headline + ' ' + summary).lower()
    # Check ticker symbol (word boundary — avoid matching partial words)
    tick = ticker.lower()
    if f' {tick} ' in f' {text} ' or text.startswith(tick + ' ') or text.endswith(' ' + tick):
        return True
    # Check company name and its main keywords
    if company_name:
        name_lower = company_name.lower()
        # Try full name
        if name_lower in text:
            return True
        # Try significant words from the name (skip generic suffixes)
        skip = {'inc', 'inc.', 'corp', 'corp.', 'co', 'co.', 'ltd', 'ltd.',
                'llc', 'plc', 'group', 'holdings', 'international', 'the',
                'and', '&', 'of', 'class', 'a', 'b', 'c'}
        words = [w for w in name_lower.replace(',', '').split() if w not in skip and len(w) > 2]
        # If any distinctive word from the company name appears, it's likely relevant
        if words and any(w in text for w in words[:3]):
            return True
    return False


def _get_company_name(ticker):
    """Try to get the company name from the stock data cache."""
    cache_file = CACHE_DIR / f"{ticker.upper()}.json"
    if cache_file.exists():
        try:
            data = json.loads(cache_file.read_text())
            return data.get('name', '')
        except Exception:
            pass
    return ''


def fetch_news(ticker, days=NEWS_LOOKBACK_DAYS):
    """Fetch recent news articles for a ticker from Finnhub.

    Filters out articles that don't actually mention the ticker or company
    name, since Finnhub's free tier often returns loosely associated articles.

    Returns:
        list[dict]: Articles with keys: headline, summary, source, datetime, url, image
        None: If Finnhub key is missing or fetch fails
    """
    key = _get_finnhub_key()
    if not key:
        return None

    to_date = datetime.now().strftime('%Y-%m-%d')
    from_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    company_name = _get_company_name(ticker)

    try:
        url = (f"https://finnhub.io/api/v1/company-news"
               f"?symbol={ticker.upper()}&from={from_date}&to={to_date}&token={key}")
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        raw = resp.json()

        if not isinstance(raw, list):
            return []

        articles = []
        filtered = 0
        for item in raw:
            headline = item.get('headline', '').strip()
            if not headline:
                continue
            summary = item.get('summary', '').strip()

            # Skip articles that don't mention the ticker or company
            if not _is_relevant(ticker, company_name, headline, summary):
                filtered += 1
                continue

            articles.append({
                'headline': headline,
                'summary': summary,
                'source': item.get('source', 'Unknown'),
                'datetime': datetime.fromtimestamp(item.get('datetime', 0)).isoformat(),
                'url': item.get('url', ''),
                'image': item.get('image', ''),
            })

        if filtered:
            print(f"  {ticker}: filtered out {filtered} irrelevant articles, kept {len(articles)}")

        # Sort newest first
        articles.sort(key=lambda a: a['datetime'], reverse=True)
        return articles

    except Exception as e:
        print(f"  Finnhub news fetch failed for {ticker}: {e}")
        return None


# -- Aggregation & Analysis -------------------------------------------------

def analyze_ticker_news(ticker, force_refresh=False):
    """Full news sentiment analysis for a ticker.

    Returns a dict with overall sentiment, daily trend, source breakdown,
    and individual article sentiments.
    """
    # Check cache
    if not force_refresh:
        cached = _get_cached_news(ticker)
        if cached:
            return cached

    # Check for Finnhub key
    if not _get_finnhub_key():
        return {
            'ticker': ticker.upper(),
            'error': 'no_finnhub_key',
            'message': 'Finnhub API key required. Save your free key to finnhub_key.txt'
        }

    # Fetch articles
    articles = fetch_news(ticker)
    if articles is None:
        return {
            'ticker': ticker.upper(),
            'error': 'fetch_failed',
            'message': 'Failed to fetch news from Finnhub'
        }

    if not articles:
        result = {
            'ticker': ticker.upper(),
            'article_count': 0,
            'overall_sentiment': {'label': 'neutral', 'score': 0.0, 'confidence': 0.0},
            'trend': [],
            'source_breakdown': [],
            'articles': [],
            'model': 'none',
        }
        _save_news_cache(ticker, result)
        return result

    # Score each article
    scored_articles = []
    for article in articles:
        text = f"{article['headline']}. {article['summary']}"
        sentiment = _analyzer.analyze(text)
        scored_articles.append({**article, 'sentiment': sentiment})

    # Overall sentiment — weighted by confidence
    total_weight = sum(a['sentiment']['confidence'] for a in scored_articles) or 1
    overall_score = sum(
        a['sentiment']['score'] * a['sentiment']['confidence']
        for a in scored_articles
    ) / total_weight

    if overall_score > 0.15:
        overall_label = 'bullish'
    elif overall_score < -0.15:
        overall_label = 'bearish'
    else:
        overall_label = 'neutral'

    overall_confidence = sum(a['sentiment']['confidence'] for a in scored_articles) / len(scored_articles)

    # Daily trend
    daily = defaultdict(lambda: {'scores': [], 'count': 0})
    for a in scored_articles:
        date = a['datetime'][:10]  # YYYY-MM-DD
        daily[date]['scores'].append(a['sentiment']['score'])
        daily[date]['count'] += 1

    trend = []
    for date in sorted(daily.keys()):
        d = daily[date]
        avg = sum(d['scores']) / len(d['scores'])
        trend.append({
            'date': date,
            'score': round(avg, 3),
            'count': d['count']
        })

    # Source breakdown
    sources = defaultdict(lambda: {'scores': [], 'count': 0})
    for a in scored_articles:
        src = a['source']
        sources[src]['scores'].append(a['sentiment']['score'])
        sources[src]['count'] += 1

    source_breakdown = []
    for src, data in sorted(sources.items(), key=lambda x: x[1]['count'], reverse=True):
        avg = sum(data['scores']) / len(data['scores'])
        if avg > 0.15:
            label = 'bullish'
        elif avg < -0.15:
            label = 'bearish'
        else:
            label = 'neutral'
        source_breakdown.append({
            'source': src,
            'avg_score': round(avg, 3),
            'count': data['count'],
            'label': label
        })

    # Determine which model was used
    model_name = 'finbert' if _analyzer._available else 'keyword_fallback'

    result = {
        'ticker': ticker.upper(),
        'article_count': len(scored_articles),
        'overall_sentiment': {
            'label': overall_label,
            'score': round(overall_score, 3),
            'confidence': round(overall_confidence, 3)
        },
        'trend': trend,
        'source_breakdown': source_breakdown,
        'articles': scored_articles,
        'model': model_name,
    }

    _save_news_cache(ticker, result)
    return result
