#!/usr/bin/env python3
"""
KALSHI BOT — AUTOMATED SETUP SCRIPT
Run this once after cloning. Sets up database, trains model, verifies everything.
Usage: python setup.py
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def header(text):
    print(f"\n{'═'*60}")
    print(f"  {text}")
    print('═'*60)

def step(text):
    print(f"\n▶  {text}")

def ok(text):
    print(f"   ✅  {text}")

def warn(text):
    print(f"   ⚠️   {text}")

def fail(text):
    print(f"   ❌  {text}")

def run(cmd, cwd=None):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=cwd)
    if result.returncode != 0:
        print(result.stderr)
        return False
    return True


header("KALSHI CPI BOT — SETUP")
print("""
This script will:
  1. Create your .env configuration file
  2. Install Python dependencies
  3. Set up the database (SQLite for dev, PostgreSQL for prod)
  4. Verify API connections
  5. Train the forecasting model
  6. Launch the dashboard

Estimated time: 5–15 minutes (model training takes a few minutes)
""")

# ─── STEP 1: ENV FILE ───────────────────────────────────────────

step("Setting up environment configuration")

env_file = Path(".env")
template = Path(".env.template")

if env_file.exists():
    warn(".env already exists, skipping creation")
else:
    shutil.copy(template, env_file)
    ok(".env created from template")
    print("""
   ┌─────────────────────────────────────────────────────┐
   │  IMPORTANT: Edit .env with your API keys before     │
   │  running the full bot. For now, setup will proceed  │
   │  in paper trading mode with SQLite (no DB needed).  │
   └─────────────────────────────────────────────────────┘
   
   Required API keys (get them free):
   
   1. FRED API key:
      → https://fred.stlouisfed.org/docs/api/api_key.html
      → Takes 2 minutes, instant approval
   
   2. BLS API key:
      → https://data.bls.gov/registrationEngine/
      → Takes 2 minutes, instant approval
   
   3. Kalshi API key (for live market data):
      → https://kalshi.com/account/api
      → Requires Kalshi account
   
   For paper trading, only FRED + BLS are required.
   Kalshi key needed to pull real market prices.
    """)

# ─── STEP 2: DEPENDENCIES ───────────────────────────────────────

step("Installing Python dependencies")

# Detect if in virtual environment
in_venv = sys.prefix != sys.base_prefix
if not in_venv:
    warn("Not in a virtual environment. Recommend: python -m venv venv && source venv/bin/activate")

success = run(f"{sys.executable} -m pip install -r requirements.txt -q")
if success:
    ok("Dependencies installed")
else:
    fail("Dependency installation failed. Check requirements.txt and your Python version.")
    sys.exit(1)

# ─── STEP 3: DATABASE ───────────────────────────────────────────

step("Initializing database")

# Check if Postgres URL is set
from dotenv import load_dotenv
load_dotenv()

db_url = os.getenv("DATABASE_URL", "")
if not db_url or db_url.startswith("postgresql://kalshi_user:yourpassword"):
    warn("No real database configured. Using SQLite (good for development).")
    os.environ["DATABASE_URL"] = "sqlite:///kalshi_bot.db"
    print("   → Set DATABASE_URL=sqlite:///kalshi_bot.db")
else:
    ok(f"Using database: {db_url[:30]}...")

# Init DB
try:
    sys.path.insert(0, str(Path.cwd()))
    from data.database import init_db
    init_db()
    ok("Database initialized")
except Exception as e:
    fail(f"Database init failed: {e}")
    sys.exit(1)

# ─── STEP 4: API VERIFICATION ───────────────────────────────────

step("Verifying API connections")

fred_key = os.getenv("FRED_API_KEY", "")
if fred_key and fred_key != "your_fred_api_key_here":
    try:
        import requests
        r = requests.get(
            "https://api.stlouisfed.org/fred/series/observations",
            params={"series_id": "CPIAUCSL", "api_key": fred_key, "file_type": "json", 
                    "limit": 1, "sort_order": "desc"},
            timeout=10
        )
        if r.status_code == 200:
            ok("FRED API ✓")
        else:
            warn(f"FRED API returned {r.status_code}. Check your key.")
    except Exception as e:
        warn(f"FRED API check failed: {e}")
else:
    warn("FRED API key not set. Edit .env to add FRED_API_KEY")

bls_key = os.getenv("BLS_API_KEY", "")
if bls_key and bls_key != "your_bls_api_key_here":
    ok("BLS API key found ✓")
else:
    warn("BLS API key not set. Bot will use FRED as fallback.")

kalshi_key = os.getenv("KALSHI_API_KEY", "")
if kalshi_key and kalshi_key != "your_kalshi_api_key_here":
    ok("Kalshi API key found ✓")
else:
    warn("Kalshi API key not set. Running in full paper mode (no live market prices).")

# ─── STEP 5: DATA INGESTION ─────────────────────────────────────

step("Running initial data ingestion (pulling CPI history)")

try:
    if fred_key and fred_key != "your_fred_api_key_here":
        from data.ingestion import run_data_ingestion
        run_data_ingestion()
        ok("Historical data loaded")
    else:
        warn("Skipping data ingestion (no FRED key). Add key to .env and re-run.")
except Exception as e:
    warn(f"Data ingestion skipped: {e}")

# ─── STEP 6: MODEL TRAINING ─────────────────────────────────────

step("Training CPI forecasting model")

if fred_key and fred_key != "your_fred_api_key_here":
    print("   This takes 2–5 minutes...")
    try:
        from engine import TradingEngine
        engine = TradingEngine()
        engine.train_model(start_year=1995)
        ok("Model trained and saved to models/cpi_forecaster.pkl")
    except Exception as e:
        warn(f"Model training failed: {e}")
        warn("You can train manually later: python engine.py train")
else:
    warn("Skipping model training (no FRED key). Add key to .env and run: python engine.py train")

# ─── STEP 7: CREATE STARTUP SCRIPTS ─────────────────────────────

step("Creating startup scripts")

# Dashboard start script
with open("start_dashboard.sh", "w") as f:
    f.write(f"""#!/bin/bash
cd "$(dirname "$0")"
source venv/bin/activate 2>/dev/null || true
echo "Starting Kalshi Bot Dashboard..."
echo "Open: http://localhost:8000"
{sys.executable} api.py
""")
os.chmod("start_dashboard.sh", 0o755)
ok("Created: start_dashboard.sh")

# Bot scheduler start script  
with open("start_bot.sh", "w") as f:
    f.write(f"""#!/bin/bash
cd "$(dirname "$0")"
source venv/bin/activate 2>/dev/null || true
echo "Starting Kalshi Bot Scheduler..."
echo "The bot will run daily trades automatically."
echo "Press Ctrl+C to stop."
{sys.executable} scheduler.py
""")
os.chmod("start_bot.sh", 0o755)
ok("Created: start_bot.sh")

# Windows batch files
with open("start_dashboard.bat", "w") as f:
    f.write(f"""@echo off
cd /d "%~dp0"
call venv\\Scripts\\activate 2>nul
echo Starting Kalshi Bot Dashboard...
echo Open: http://localhost:8000
python api.py
pause
""")
ok("Created: start_dashboard.bat (Windows)")

with open("start_bot.bat", "w") as f:
    f.write(f"""@echo off
cd /d "%~dp0"
call venv\\Scripts\\activate 2>nul
echo Starting Kalshi Bot Scheduler...
python scheduler.py
pause
""")
ok("Created: start_bot.bat (Windows)")

# ─── DONE ───────────────────────────────────────────────────────

header("SETUP COMPLETE")
print(f"""
Everything is ready. Here's how to use the system:

╔═══════════════════════════════════════════════════════════╗
║                     DAILY WORKFLOW                       ║
╠═══════════════════════════════════════════════════════════╣
║                                                           ║
║  TERMINAL 1 — Start the dashboard:                       ║
║  Mac/Linux:  ./start_dashboard.sh                        ║
║  Windows:    start_dashboard.bat                         ║
║  Then open:  http://localhost:8000                        ║
║                                                           ║
║  TERMINAL 2 — Start the bot (auto-trades on schedule):   ║
║  Mac/Linux:  ./start_bot.sh                              ║
║  Windows:    start_bot.bat                               ║
║                                                           ║
╠═══════════════════════════════════════════════════════════╣
║                   BEFORE GOING LIVE                      ║
╠═══════════════════════════════════════════════════════════╣
║                                                           ║
║  1. Add your API keys to .env                            ║
║  2. Re-run: python setup.py                              ║
║  3. Paper trade for 2-3 CPI releases                     ║
║  4. Check Brier scores in dashboard                      ║
║  5. Only then change TRADING_MODE=live in .env           ║
║                                                           ║
╠═══════════════════════════════════════════════════════════╣
║                   MANUAL COMMANDS                        ║
╠═══════════════════════════════════════════════════════════╣
║                                                           ║
║  Retrain model:    python engine.py train                ║
║  Run one scan:     python engine.py                      ║
║  Check DB:         python -c "from data.database import  ║
║                    get_session; print('DB OK')"           ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝

Log files: logs/scheduler.log
Database: kalshi_bot.db (SQLite) or your PostgreSQL DB

Mode: {os.getenv('TRADING_MODE', 'paper').upper()} TRADING
""")
