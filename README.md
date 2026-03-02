# Hatch Research

Stock research tool for NASDAQ and NYSE shares (via Hatch platform). Scores stocks on fundamentals and helps narrow down purchasing decisions.

## Features
- **Ticker Lookup** — Enter any ticker(s) to see full fundamental profile with score
- **Side-by-Side Comparison** — Compare up to 8 stocks with colour-coded metrics
- **Exchange Scanner** — Scan NASDAQ/NYSE for top-scoring Hatch-eligible stocks
- **CSV Export** — Export scan results for offline analysis

## Scoring (100 points)
- Business Quality (35 pts): FCF, ROE, revenue growth, margins
- Valuation (25 pts): PEG, forward P/E, P/B, trailing P/E
- Financial Strength (20 pts): debt/equity, current ratio, FCF positive
- Market Sentiment (20 pts): analyst consensus, price target upside, 52-wk position

## How to Run
1. Open terminal in project folder
2. Run: `python hatch_research.py`
3. Browser opens at http://localhost:5001

## Optional: Full Exchange Scanning
1. Get a free API key from https://finnhub.io
2. Save it to `finnhub_key.txt` in the project folder
3. Restart the app — "Scan NASDAQ" / "Scan NYSE" buttons will now work with full ticker lists
