# Kalshi CPI Trading Bot

Automated systematic trading system for Kalshi CPI prediction markets.

## Architecture

```
kalshi_bot/
├── data/
│   ├── database.py      — SQLAlchemy models, DB init
│   └── ingestion.py     — FRED, BLS, EIA, Cleveland Fed data pipelines
├── models/
│   └── forecaster.py    — Ridge + XGBoost CPI ensemble model
├── execution/
│   ├── kalshi_client.py — Kalshi API client + paper trading mock
│   └── risk_manager.py  — Kelly sizing, circuit breakers, exposure limits
├── dashboard/
│   └── index.html       — Trading terminal dashboard
├── engine.py            — Main trading orchestrator
├── scheduler.py         — Cron-style job runner
├── api.py               — FastAPI backend for dashboard
├── setup.py             — One-time setup script
└── .env.template        — Configuration template
```

## Quickstart

### Step 1: Prerequisites

You need Python 3.10+ installed. Check with:
```
python --version
```

### Step 2: Download the code

If you're using the zip file, extract it. Then open your terminal (Mac: Terminal app, Windows: Command Prompt or PowerShell) and navigate to the folder:

```
cd kalshi_bot
```

### Step 3: Create a virtual environment

**Mac/Linux:**
```
python -m venv venv
source venv/bin/activate
```

**Windows:**
```
python -m venv venv
venv\Scripts\activate
```

You should see `(venv)` at the start of your prompt.

### Step 4: Get your free API keys

You need these before the bot can pull data:

**FRED API Key** (pulls all economic data):
1. Go to https://fred.stlouisfed.org/docs/api/api_key.html
2. Click "Request API Key"
3. Fill out the form (2 minutes)
4. Copy the key

**BLS API Key** (BLS inflation data):
1. Go to https://data.bls.gov/registrationEngine/
2. Register (2 minutes)
3. They email you a key within minutes

**Kalshi API Key** (live market prices):
1. Log into https://kalshi.com
2. Go to Account → API Keys
3. Create a new key, copy it and the secret

### Step 5: Configure your .env file

Open `.env.template`, save it as `.env`, then fill in your keys:

```
FRED_API_KEY=your_key_here
BLS_API_KEY=your_key_here
KALSHI_API_KEY=your_key_here
KALSHI_API_SECRET=your_secret_here
TRADING_MODE=paper
STARTING_PAPER_CAPITAL=25000
```

Leave `TRADING_MODE=paper` until you've validated the model.

### Step 6: Run setup

```
python setup.py
```

This will:
- Install all dependencies (~2 min)
- Initialize the database
- Pull historical CPI data back to 1995
- Train the forecasting model (~3-5 min)
- Create startup scripts

### Step 7: Start the dashboard

Open a terminal window:
```
./start_dashboard.sh     (Mac/Linux)
start_dashboard.bat      (Windows)
```

Then open your browser to: **http://localhost:8000**

### Step 8: Start the bot

Open a second terminal window:
```
./start_bot.sh           (Mac/Linux)
start_bot.bat            (Windows)
```

The bot will now run automatically on this schedule:
- **6:00 AM ET** — Pull fresh economic data
- **7:00 AM ET** — Scan CPI markets, execute paper trades
- **8:45 AM ET** — Check for settled contracts
- **12:00 PM ET** — Midday market refresh
- **6:00 PM ET** — Risk checks
- **1st of month, 2 AM** — Retrain model

## Dashboard Features

- **Equity Curve** — Track paper portfolio value over time
- **Live Markets** — See which Kalshi CPI buckets have edge
- **Trade Log** — All paper trades with P&L
- **Manual Trade Entry** — Log trades you place yourself
- **Settle Trades** — Mark open positions as win/loss
- **Forecasts** — See what the model predicts for next CPI
- **System Status** — Circuit breaker states

## Paper Trading

The bot starts in paper trading mode with $25,000 virtual capital. Every trade the bot would place is logged to the database and tracked in the dashboard. No real money moves.

To manually log a trade you placed on Kalshi yourself:
1. Open the dashboard
2. Click the "TRADE LOG" panel
3. Click "+ LOG TRADE" tab
4. Fill in the details and click "LOG PAPER TRADE"
5. After the CPI release, go to "OPEN" tab and click WIN or LOSS to settle

## Going Live

Only do this after:
1. At least 3 paper CPI releases
2. Model Brier score is better than consensus (dashboard shows this)
3. You understand the risk model

Then in `.env`, change:
```
TRADING_MODE=live
```

And restart both processes.

## Risk Parameters (in .env)

| Parameter | Default | Meaning |
|-----------|---------|---------|
| MAX_SINGLE_CONTRACT_DOLLARS | 5000 | Max $ per bucket contract |
| MAX_SINGLE_RELEASE_DOLLARS | 15000 | Max $ total per CPI release |
| MAX_MONTHLY_DRAWDOWN_PCT | 0.15 | 15% drawdown triggers halt |
| MIN_NET_EDGE_TO_TRADE | 0.04 | 4¢ minimum edge required |
| KELLY_FRACTION | 0.25 | Use 25% of Kelly sizing |

## Database

The bot uses SQLite by default (no setup needed). For production, use PostgreSQL:

```
DATABASE_URL=postgresql://user:password@localhost:5432/kalshi_bot
```

Create the PostgreSQL database:
```sql
CREATE DATABASE kalshi_bot;
CREATE USER kalshi_user WITH PASSWORD 'yourpassword';
GRANT ALL PRIVILEGES ON DATABASE kalshi_bot TO kalshi_user;
```

## Troubleshooting

**"No module named X"** — Run `pip install -r requirements.txt`

**"Database error"** — Delete `kalshi_bot.db` and re-run `python setup.py`

**"FRED API error"** — Check your FRED_API_KEY in .env

**Bot not placing trades** — Check logs/scheduler.log. Most likely cause: no Kalshi API key or no edge found (that's fine, means model doesn't see mispricing)

**Dashboard shows "OFFLINE"** — The API server isn't running. Start it with `./start_dashboard.sh`

## Logs

All activity logged to `logs/scheduler.log`. Rotates weekly.

```
tail -f logs/scheduler.log    (Mac/Linux, live log view)
```
