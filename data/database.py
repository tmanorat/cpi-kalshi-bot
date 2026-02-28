"""
Database schema for Kalshi CPI Trading Bot.
Uses SQLAlchemy ORM with PostgreSQL.
"""

from sqlalchemy import (
    create_engine, Column, Integer, String, Float, DateTime, 
    Boolean, Text, JSON, ForeignKey, Date
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.sql import func
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

Base = declarative_base()

# ─────────────────────────────────────────────
# CORE TABLES
# ─────────────────────────────────────────────

class EconomicRelease(Base):
    """Stores actual economic release values (CPI, PCE, etc.)"""
    __tablename__ = "economic_releases"
    
    id = Column(Integer, primary_key=True)
    series_id = Column(String(50), nullable=False)       # e.g. 'CPIAUCSL'
    release_date = Column(Date, nullable=False)           # date BLS released it
    reference_period = Column(Date, nullable=False)       # month it covers
    value = Column(Float, nullable=False)                 # actual value
    value_mom = Column(Float)                             # month-over-month change
    value_yoy = Column(Float)                             # year-over-year change
    source = Column(String(50))
    revision_number = Column(Integer, default=0)         # 0 = initial release
    ingested_at = Column(DateTime, default=func.now())


class FeatureSnapshot(Base):
    """Daily snapshot of all model features at prediction time."""
    __tablename__ = "feature_snapshots"
    
    id = Column(Integer, primary_key=True)
    snapshot_date = Column(Date, nullable=False)
    reference_period = Column(Date, nullable=False)       # CPI month being forecast
    features = Column(JSON, nullable=False)               # all feature values as dict
    created_at = Column(DateTime, default=func.now())


class ModelForecast(Base):
    """Model probability forecasts for each CPI release."""
    __tablename__ = "model_forecasts"
    
    id = Column(Integer, primary_key=True)
    series_id = Column(String(50), nullable=False)
    forecast_date = Column(Date, nullable=False)
    reference_period = Column(Date, nullable=False)
    mu_model = Column(Float, nullable=False)             # mean forecast (MoM %)
    sigma_model = Column(Float, nullable=False)          # uncertainty (std dev)
    mu_cleveland = Column(Float)
    mu_ridge = Column(Float)
    mu_xgboost = Column(Float)
    mu_spf = Column(Float)
    weights_used = Column(JSON)
    model_version = Column(String(20))
    created_at = Column(DateTime, default=func.now())
    
    trades = relationship("Trade", back_populates="model_forecast")


class KalshiMarketSnapshot(Base):
    """Point-in-time snapshot of Kalshi market prices."""
    __tablename__ = "kalshi_market_snapshots"
    
    id = Column(Integer, primary_key=True)
    market_id = Column(String(150), nullable=False)
    series_id = Column(String(50))
    reference_period = Column(Date)
    snapshot_time = Column(DateTime, nullable=False)
    bucket_low = Column(Float)
    bucket_high = Column(Float)
    best_bid = Column(Float)
    best_ask = Column(Float)
    mid_price = Column(Float)
    spread = Column(Float)
    volume_24h = Column(Integer)
    open_interest = Column(Integer)
    p_model = Column(Float)                              # our model's probability
    net_edge = Column(Float)                             # p_model - mid_price - fees


class Trade(Base):
    """All trades — both paper and live."""
    __tablename__ = "trades"
    
    id = Column(Integer, primary_key=True)
    market_id = Column(String(150), nullable=False)
    reference_period = Column(Date)
    trade_time = Column(DateTime, default=func.now())
    side = Column(String(10), nullable=False)            # 'BUY' or 'SELL'
    contracts = Column(Integer, nullable=False)
    price = Column(Float, nullable=False)
    total_cost = Column(Float)                           # contracts * price
    net_edge_at_entry = Column(Float)
    p_model_at_entry = Column(Float)
    p_market_at_entry = Column(Float)
    kelly_fraction_used = Column(Float)
    mode = Column(String(10), default='paper')          # 'paper' or 'live'
    status = Column(String(20), default='open')         # 'open', 'settled', 'cancelled'
    outcome = Column(Boolean)                           # True = won, False = lost
    settlement_price = Column(Float)
    pnl_gross = Column(Float)
    pnl_net = Column(Float)                             # after fees
    fees_paid = Column(Float)
    model_forecast_id = Column(Integer, ForeignKey("model_forecasts.id"))
    notes = Column(Text)
    
    model_forecast = relationship("ModelForecast", back_populates="trades")


class PerformanceLog(Base):
    """Daily/monthly performance metrics."""
    __tablename__ = "performance_log"
    
    id = Column(Integer, primary_key=True)
    log_date = Column(Date, nullable=False)
    mode = Column(String(10), default='paper')
    brier_score = Column(Float)
    brier_skill_score = Column(Float)
    calibration_error = Column(Float)
    realized_pnl = Column(Float)
    cumulative_pnl = Column(Float)
    portfolio_value = Column(Float)
    num_contracts_resolved = Column(Integer)
    win_rate = Column(Float)
    avg_edge = Column(Float)
    sharpe_ratio = Column(Float)
    max_drawdown = Column(Float)
    model_version = Column(String(20))
    created_at = Column(DateTime, default=func.now())


class CircuitBreakerEvent(Base):
    """Log of all circuit breaker triggers."""
    __tablename__ = "circuit_breaker_events"
    
    id = Column(Integer, primary_key=True)
    triggered_at = Column(DateTime, default=func.now())
    reason = Column(String(100))
    details = Column(JSON)
    halt_days = Column(Integer)
    resume_at = Column(DateTime)
    resolved = Column(Boolean, default=False)


class PaperPortfolio(Base):
    """Tracks paper trading portfolio state."""
    __tablename__ = "paper_portfolio"
    
    id = Column(Integer, primary_key=True)
    updated_at = Column(DateTime, default=func.now())
    cash_balance = Column(Float, nullable=False)
    total_exposure = Column(Float, default=0)
    unrealized_pnl = Column(Float, default=0)
    realized_pnl = Column(Float, default=0)
    total_value = Column(Float)
    num_open_positions = Column(Integer, default=0)


# ─────────────────────────────────────────────
# DATABASE INITIALIZATION
# ─────────────────────────────────────────────

def get_engine():
    db_url = os.getenv("DATABASE_URL", "sqlite:///kalshi_bot.db")
    return create_engine(db_url, echo=False)

def get_session():
    engine = get_engine()
    Session = sessionmaker(bind=engine)
    return Session()

def init_db():
    """Create all tables and seed paper portfolio if needed."""
    engine = get_engine()
    Base.metadata.create_all(engine)
    
    session = sessionmaker(bind=engine)()
    
    # Seed paper portfolio if none exists
    existing = session.query(PaperPortfolio).first()
    if not existing:
        starting_capital = float(os.getenv("STARTING_PAPER_CAPITAL", 25000))
        portfolio = PaperPortfolio(
            cash_balance=starting_capital,
            total_value=starting_capital,
            realized_pnl=0,
            unrealized_pnl=0,
            total_exposure=0,
            num_open_positions=0
        )
        session.add(portfolio)
        session.commit()
        print(f"✅ Paper portfolio initialized with ${starting_capital:,.0f}")
    
    session.close()
    print("✅ Database initialized successfully")

if __name__ == "__main__":
    init_db()
