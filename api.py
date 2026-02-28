"""
FastAPI backend — serves data to the dashboard.
Endpoints for portfolio, trades, forecasts, performance.
"""

import os
import numpy as np
from datetime import datetime, date, timedelta
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from sqlalchemy import desc, func
from pydantic import BaseModel
from loguru import logger
from dotenv import load_dotenv

from data.database import (
    get_session, Trade, ModelForecast, KalshiMarketSnapshot,
    PerformanceLog, PaperPortfolio, CircuitBreakerEvent, EconomicRelease
)

load_dotenv()

app = FastAPI(title="Kalshi CPI Bot Dashboard", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# PORTFOLIO ENDPOINTS
# ─────────────────────────────────────────────

@app.get("/api/portfolio")
def get_portfolio():
    session = get_session()
    try:
        portfolio = session.query(PaperPortfolio).order_by(
            PaperPortfolio.updated_at.desc()
        ).first()
        
        if not portfolio:
            starting = float(os.getenv("STARTING_PAPER_CAPITAL", 25000))
            return {
                "cash_balance": starting,
                "total_value": starting,
                "realized_pnl": 0,
                "unrealized_pnl": 0,
                "total_exposure": 0,
                "num_open_positions": 0,
                "pnl_pct": 0,
                "mode": os.getenv("TRADING_MODE", "paper")
            }
        
        starting = float(os.getenv("STARTING_PAPER_CAPITAL", 25000))
        pnl_pct = ((portfolio.total_value - starting) / starting * 100) if starting > 0 else 0
        
        return {
            "cash_balance": round(portfolio.cash_balance, 2),
            "total_value": round(portfolio.total_value, 2),
            "realized_pnl": round(portfolio.realized_pnl or 0, 2),
            "unrealized_pnl": round(portfolio.unrealized_pnl or 0, 2),
            "total_exposure": round(portfolio.total_exposure or 0, 2),
            "num_open_positions": portfolio.num_open_positions or 0,
            "pnl_pct": round(pnl_pct, 2),
            "starting_capital": starting,
            "mode": os.getenv("TRADING_MODE", "paper"),
            "updated_at": portfolio.updated_at.isoformat() if portfolio.updated_at else None
        }
    finally:
        session.close()


@app.get("/api/portfolio/equity-curve")
def get_equity_curve(days: int = 90):
    """Return daily portfolio value history for chart."""
    session = get_session()
    try:
        since = date.today() - timedelta(days=days)
        logs = session.query(PerformanceLog).filter(
            PerformanceLog.log_date >= since
        ).order_by(PerformanceLog.log_date).all()
        
        starting = float(os.getenv("STARTING_PAPER_CAPITAL", 25000))
        
        # Build equity curve
        points = [{"date": str(date.today() - timedelta(days=days)), "value": starting}]
        
        cumulative = starting
        for log in logs:
            if log.portfolio_value:
                points.append({
                    "date": str(log.log_date),
                    "value": round(log.portfolio_value, 2),
                    "pnl": round(log.realized_pnl or 0, 2)
                })
        
        return {"curve": points, "starting_capital": starting}
    finally:
        session.close()


# ─────────────────────────────────────────────
# TRADES ENDPOINTS
# ─────────────────────────────────────────────

@app.get("/api/trades")
def get_trades(
    limit: int = Query(50, le=200),
    offset: int = 0,
    status: Optional[str] = None,
    mode: Optional[str] = None
):
    session = get_session()
    try:
        query = session.query(Trade).order_by(desc(Trade.trade_time))
        
        if status:
            query = query.filter(Trade.status == status)
        if mode:
            query = query.filter(Trade.mode == mode)
        
        total = query.count()
        trades = query.offset(offset).limit(limit).all()
        
        return {
            "total": total,
            "trades": [
                {
                    "id": t.id,
                    "market_id": t.market_id,
                    "reference_period": str(t.reference_period) if t.reference_period else None,
                    "trade_time": t.trade_time.isoformat() if t.trade_time else None,
                    "side": t.side,
                    "contracts": t.contracts,
                    "price": round(t.price, 4) if t.price else None,
                    "total_cost": round(t.total_cost, 2) if t.total_cost else None,
                    "net_edge_at_entry": round(t.net_edge_at_entry, 4) if t.net_edge_at_entry else None,
                    "p_model_at_entry": round(t.p_model_at_entry, 4) if t.p_model_at_entry else None,
                    "p_market_at_entry": round(t.p_market_at_entry, 4) if t.p_market_at_entry else None,
                    "mode": t.mode,
                    "status": t.status,
                    "outcome": t.outcome,
                    "pnl_net": round(t.pnl_net, 2) if t.pnl_net else None,
                    "pnl_gross": round(t.pnl_gross, 2) if t.pnl_gross else None,
                    "fees_paid": round(t.fees_paid, 2) if t.fees_paid else None,
                }
                for t in trades
            ]
        }
    finally:
        session.close()


@app.get("/api/trades/stats")
def get_trade_stats():
    """Aggregate trade statistics."""
    session = get_session()
    try:
        all_settled = session.query(Trade).filter_by(status="settled").all()
        
        if not all_settled:
            return {
                "total_trades": 0,
                "win_rate": 0,
                "total_pnl": 0,
                "avg_pnl_per_trade": 0,
                "avg_edge": 0,
                "sharpe": 0,
                "best_trade": 0,
                "worst_trade": 0,
                "total_fees": 0
            }
        
        pnls = [t.pnl_net for t in all_settled if t.pnl_net is not None]
        wins = [t for t in all_settled if t.outcome is True]
        edges = [t.net_edge_at_entry for t in all_settled if t.net_edge_at_entry]
        fees = [t.fees_paid for t in all_settled if t.fees_paid]
        
        sharpe = 0.0
        if len(pnls) > 1 and np.std(pnls) > 0:
            sharpe = float(np.mean(pnls) / np.std(pnls) * np.sqrt(12))
        
        return {
            "total_trades": len(all_settled),
            "total_wins": len(wins),
            "win_rate": round(len(wins) / len(all_settled) * 100, 1),
            "total_pnl": round(sum(pnls), 2),
            "avg_pnl_per_trade": round(np.mean(pnls), 2) if pnls else 0,
            "avg_edge": round(np.mean(edges), 4) if edges else 0,
            "sharpe": round(sharpe, 2),
            "best_trade": round(max(pnls), 2) if pnls else 0,
            "worst_trade": round(min(pnls), 2) if pnls else 0,
            "total_fees": round(sum(fees), 2) if fees else 0,
            "edge_realization_rate": _compute_edge_realization(all_settled)
        }
    finally:
        session.close()


def _compute_edge_realization(trades) -> float:
    """What % of theoretical edge did we actually capture?"""
    theoretical_edges = [t.net_edge_at_entry * t.total_cost 
                        for t in trades if t.net_edge_at_entry and t.total_cost]
    actual_pnls = [t.pnl_net for t in trades if t.pnl_net is not None]
    
    if not theoretical_edges or not actual_pnls:
        return 0.0
    
    total_theoretical = sum(theoretical_edges)
    total_actual = sum(actual_pnls)
    
    if total_theoretical == 0:
        return 0.0
    
    return round(total_actual / total_theoretical, 3)


# ─────────────────────────────────────────────
# FORECASTS ENDPOINTS
# ─────────────────────────────────────────────

@app.get("/api/forecasts/latest")
def get_latest_forecasts(limit: int = 10):
    """Get most recent model forecasts."""
    session = get_session()
    try:
        forecasts = session.query(ModelForecast).order_by(
            desc(ModelForecast.created_at)
        ).limit(limit).all()
        
        return {
            "forecasts": [
                {
                    "id": f.id,
                    "reference_period": str(f.reference_period),
                    "forecast_date": str(f.forecast_date),
                    "mu_model": round(f.mu_model, 4),
                    "sigma_model": round(f.sigma_model, 4),
                    "mu_cleveland": f.mu_cleveland,
                    "mu_ridge": f.mu_ridge,
                    "mu_xgboost": f.mu_xgboost,
                    "model_version": f.model_version,
                    "created_at": f.created_at.isoformat()
                }
                for f in forecasts
            ]
        }
    finally:
        session.close()


@app.get("/api/forecasts/accuracy")
def get_forecast_accuracy():
    """Compare model forecasts vs actual releases."""
    session = get_session()
    try:
        forecasts = session.query(ModelForecast).all()
        releases = {r.reference_period: r for r in session.query(EconomicRelease).filter_by(series_id="CPIAUCSL").all()}
        
        accuracy_data = []
        for f in forecasts:
            actual = releases.get(f.reference_period)
            if actual and actual.value_mom is not None:
                error = f.mu_model - actual.value_mom
                accuracy_data.append({
                    "period": str(f.reference_period),
                    "forecast": round(f.mu_model, 4),
                    "actual": round(actual.value_mom, 4),
                    "error": round(error, 4),
                    "abs_error": round(abs(error), 4),
                    "within_1sigma": abs(error) <= f.sigma_model,
                    "within_2sigma": abs(error) <= 2 * f.sigma_model,
                })
        
        maes = [d["abs_error"] for d in accuracy_data]
        
        return {
            "accuracy": accuracy_data,
            "mae": round(np.mean(maes), 4) if maes else None,
            "rmse": round(np.sqrt(np.mean([x**2 for x in maes])), 4) if maes else None,
            "within_1sigma_pct": round(np.mean([d["within_1sigma"] for d in accuracy_data]) * 100, 1) if accuracy_data else None,
            "n_forecasts": len(accuracy_data)
        }
    finally:
        session.close()


# ─────────────────────────────────────────────
# MARKET SNAPSHOTS
# ─────────────────────────────────────────────

@app.get("/api/markets/current")
def get_current_markets():
    """Get latest Kalshi market snapshots with edge analysis."""
    session = get_session()
    try:
        # Get most recent snapshot per market
        subq = session.query(
            KalshiMarketSnapshot.market_id,
            func.max(KalshiMarketSnapshot.snapshot_time).label("max_time")
        ).group_by(KalshiMarketSnapshot.market_id).subquery()
        
        snapshots = session.query(KalshiMarketSnapshot).join(
            subq,
            (KalshiMarketSnapshot.market_id == subq.c.market_id) &
            (KalshiMarketSnapshot.snapshot_time == subq.c.max_time)
        ).all()
        
        return {
            "markets": [
                {
                    "market_id": s.market_id,
                    "bucket_low": s.bucket_low,
                    "bucket_high": s.bucket_high,
                    "mid_price": s.mid_price,
                    "best_bid": s.best_bid,
                    "best_ask": s.best_ask,
                    "spread": s.spread,
                    "volume_24h": s.volume_24h,
                    "p_model": s.p_model,
                    "net_edge": round(s.net_edge, 4) if s.net_edge else None,
                    "snapshot_time": s.snapshot_time.isoformat(),
                    "reference_period": str(s.reference_period) if s.reference_period else None
                }
                for s in snapshots
            ]
        }
    finally:
        session.close()


# ─────────────────────────────────────────────
# SYSTEM STATUS
# ─────────────────────────────────────────────

@app.get("/api/system/status")
def get_system_status():
    """System health check."""
    from execution.risk_manager import RiskManager
    rm = RiskManager()
    
    session = get_session()
    try:
        # Check for active circuit breakers
        active_breakers = session.query(CircuitBreakerEvent).filter_by(
            resolved=False
        ).filter(CircuitBreakerEvent.resume_at > datetime.utcnow()).all()
        
        # Last data ingestion time (most recent economic release)
        last_release = session.query(EconomicRelease).order_by(
            desc(EconomicRelease.ingested_at)
        ).first()
        
        # Recent trade count
        last_24h_trades = session.query(Trade).filter(
            Trade.trade_time >= datetime.utcnow() - timedelta(hours=24)
        ).count()
        
        return {
            "status": "HALTED" if active_breakers else "RUNNING",
            "mode": os.getenv("TRADING_MODE", "paper").upper(),
            "circuit_breakers_active": len(active_breakers),
            "circuit_breakers": [
                {
                    "reason": b.reason,
                    "triggered_at": b.triggered_at.isoformat(),
                    "resume_at": b.resume_at.isoformat()
                }
                for b in active_breakers
            ],
            "last_data_update": last_release.ingested_at.isoformat() if last_release else None,
            "trades_last_24h": last_24h_trades,
            "timestamp": datetime.utcnow().isoformat()
        }
    finally:
        session.close()


@app.get("/api/system/circuit-breakers")
def get_circuit_breakers():
    session = get_session()
    try:
        events = session.query(CircuitBreakerEvent).order_by(
            desc(CircuitBreakerEvent.triggered_at)
        ).limit(20).all()
        
        return {
            "events": [
                {
                    "id": e.id,
                    "reason": e.reason,
                    "details": e.details,
                    "triggered_at": e.triggered_at.isoformat(),
                    "resume_at": e.resume_at.isoformat() if e.resume_at else None,
                    "resolved": e.resolved,
                    "halt_days": e.halt_days
                }
                for e in events
            ]
        }
    finally:
        session.close()


@app.post("/api/system/circuit-breakers/{event_id}/resolve")
def resolve_circuit_breaker(event_id: int):
    """Manually resolve a circuit breaker."""
    from execution.risk_manager import RiskManager
    rm = RiskManager()
    rm.resolve_circuit_breaker(event_id)
    return {"status": "resolved", "event_id": event_id}


# ─────────────────────────────────────────────
# PAPER TRADING MANUAL ENTRY
# ─────────────────────────────────────────────

class ManualTradeRequest(BaseModel):
    market_id: str
    side: str
    contracts: int
    price: float
    net_edge: float
    p_model: float
    p_market: float
    reference_period: str
    notes: Optional[str] = ""

@app.post("/api/trades/manual")
def add_manual_trade(trade_req: ManualTradeRequest):
    """Manually log a paper trade."""
    session = get_session()
    try:
        ref_period = date.fromisoformat(trade_req.reference_period)
        trade = Trade(
            market_id=trade_req.market_id,
            reference_period=ref_period,
            side=trade_req.side.upper(),
            contracts=trade_req.contracts,
            price=trade_req.price,
            total_cost=trade_req.contracts * trade_req.price,
            net_edge_at_entry=trade_req.net_edge,
            p_model_at_entry=trade_req.p_model,
            p_market_at_entry=trade_req.p_market,
            mode="paper",
            status="open",
            notes=trade_req.notes
        )
        session.add(trade)
        
        # Update portfolio exposure
        portfolio = session.query(PaperPortfolio).order_by(
            PaperPortfolio.updated_at.desc()
        ).first()
        if portfolio:
            portfolio.cash_balance -= trade.total_cost
            portfolio.total_exposure = (portfolio.total_exposure or 0) + trade.total_cost
            portfolio.total_value = portfolio.cash_balance + portfolio.total_exposure
            portfolio.num_open_positions = (portfolio.num_open_positions or 0) + 1
        
        session.commit()
        return {"status": "created", "trade_id": trade.id}
    except Exception as e:
        session.rollback()
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        session.close()


class SettleTradeRequest(BaseModel):
    outcome: bool

@app.post("/api/trades/{trade_id}/settle")
def settle_trade(trade_id: int, req: SettleTradeRequest):
    """Manually settle a paper trade."""
    session = get_session()
    try:
        trade = session.query(Trade).get(trade_id)
        if not trade:
            raise HTTPException(status_code=404, detail="Trade not found")
        
        won = req.outcome
        if won:
            gross_pnl = trade.contracts * (1.0 - trade.price)
        else:
            gross_pnl = -trade.contracts * trade.price
        
        fees = trade.total_cost * 0.07
        net_pnl = gross_pnl - fees
        
        trade.status = "settled"
        trade.outcome = won
        trade.settlement_price = 1.0 if won else 0.0
        trade.pnl_gross = gross_pnl
        trade.pnl_net = net_pnl
        trade.fees_paid = fees
        
        # Update portfolio
        portfolio = session.query(PaperPortfolio).order_by(
            PaperPortfolio.updated_at.desc()
        ).first()
        if portfolio:
            portfolio.realized_pnl = (portfolio.realized_pnl or 0) + net_pnl
            portfolio.cash_balance += net_pnl + trade.total_cost  # Return stake + profit
            portfolio.total_exposure = max(0, (portfolio.total_exposure or 0) - trade.total_cost)
            portfolio.total_value = portfolio.cash_balance + portfolio.total_exposure
            portfolio.num_open_positions = max(0, (portfolio.num_open_positions or 0) - 1)
        
        session.commit()
        return {
            "status": "settled",
            "outcome": won,
            "pnl_net": round(net_pnl, 2),
            "pnl_gross": round(gross_pnl, 2),
            "fees": round(fees, 2)
        }
    except HTTPException:
        raise
    except Exception as e:
        session.rollback()
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        session.close()


# ─────────────────────────────────────────────
# SERVE DASHBOARD
# ─────────────────────────────────────────────

@app.get("/")
def serve_dashboard():
    return FileResponse("dashboard/index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api:app",
        host=os.getenv("DASHBOARD_HOST", "0.0.0.0"),
        port=int(os.getenv("DASHBOARD_PORT", 8000)),
        reload=False,
        log_level="info"
    )
