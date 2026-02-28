"""
Risk Management System.
Circuit breakers, position sizing, exposure limits.
"""

import os
import numpy as np
from datetime import datetime, date, timedelta
from loguru import logger
from dotenv import load_dotenv
from data.database import get_session, Trade, CircuitBreakerEvent, PaperPortfolio

load_dotenv()


class RiskManager:
    
    def __init__(self):
        self.max_single_contract = float(os.getenv("MAX_SINGLE_CONTRACT_DOLLARS", 5000))
        self.max_single_release = float(os.getenv("MAX_SINGLE_RELEASE_DOLLARS", 15000))
        self.max_monthly_drawdown = float(os.getenv("MAX_MONTHLY_DRAWDOWN_PCT", 0.15))
        self.min_net_edge = float(os.getenv("MIN_NET_EDGE_TO_TRADE", 0.04))
        self.kelly_fraction = float(os.getenv("KELLY_FRACTION", 0.25))
        self.max_spread = 0.08       # Don't trade if spread > 8 cents
        self.min_volume = 200         # Don't trade if < 200 contracts daily volume
        self.blackout_minutes = 30    # Cancel orders 30 min before release
        self.mode = os.getenv("TRADING_MODE", "paper")
    
    # ─────────────────────────────────────────────
    # POSITION SIZING
    # ─────────────────────────────────────────────
    
    def compute_position_dollars(self, net_edge: float, p_market: float,
                                   side: str, portfolio_value: float) -> float:
        """
        Fractional Kelly position sizing in dollars.
        Returns recommended $ to risk on this contract.
        """
        if net_edge <= self.min_net_edge:
            return 0.0
        
        if side == "buy":
            raw_kelly = net_edge / max(1 - p_market, 0.01)
        else:
            raw_kelly = net_edge / max(p_market, 0.01)
        
        fractional = raw_kelly * self.kelly_fraction
        fractional = max(0.0, min(fractional, 0.25))  # Hard cap: never > 25% of capital
        
        raw_dollars = fractional * portfolio_value
        
        # Apply hard caps
        dollars = min(
            raw_dollars,
            self.max_single_contract,
            portfolio_value * 0.10  # Max 10% of portfolio in one contract
        )
        
        return round(dollars, 2)
    
    def compute_contracts(self, position_dollars: float, price: float) -> int:
        """Convert dollar allocation to number of contracts."""
        if price <= 0 or position_dollars <= 0:
            return 0
        contracts = int(position_dollars / price)
        return max(0, min(contracts, 1000))  # Hard cap at 1000 contracts per order
    
    # ─────────────────────────────────────────────
    # PRE-TRADE CHECKS
    # ─────────────────────────────────────────────
    
    def pre_trade_check(self, market_snapshot: dict, net_edge: float,
                         reference_period: date, release_datetime: datetime = None) -> dict:
        """
        Run all pre-trade checks. Returns {approved: bool, reason: str}.
        """
        # 1. Edge threshold
        if abs(net_edge) < self.min_net_edge:
            return {"approved": False, "reason": f"Edge {net_edge:.3f} below minimum {self.min_net_edge}"}
        
        # 2. Spread check
        spread = market_snapshot.get("spread", 1.0)
        if spread > self.max_spread:
            return {"approved": False, "reason": f"Spread {spread:.3f} too wide (max {self.max_spread})"}
        
        # 3. Volume check
        volume = market_snapshot.get("volume_24h", 0)
        if volume < self.min_volume:
            return {"approved": False, "reason": f"Volume {volume} below minimum {self.min_volume}"}
        
        # 4. Blackout window check
        if release_datetime:
            minutes_to_release = (release_datetime - datetime.utcnow()).total_seconds() / 60
            if minutes_to_release <= self.blackout_minutes:
                return {"approved": False, "reason": f"Within {self.blackout_minutes}min blackout window"}
            if minutes_to_release < 0:
                return {"approved": False, "reason": "Release has already occurred"}
        
        # 5. Market status check
        if market_snapshot.get("status") != "open":
            return {"approved": False, "reason": f"Market status: {market_snapshot.get('status')}"}
        
        # 6. Circuit breaker check
        if self.is_halted():
            return {"approved": False, "reason": "Circuit breaker active"}
        
        # 7. Release-level exposure check
        session = get_session()
        try:
            existing_exposure = self._get_release_exposure(session, reference_period)
            if existing_exposure >= self.max_single_release:
                return {"approved": False, "reason": f"Release exposure limit reached: ${existing_exposure:.0f}"}
        finally:
            session.close()
        
        return {"approved": True, "reason": "All checks passed"}
    
    def _get_release_exposure(self, session, reference_period: date) -> float:
        """Get total $ exposure for a given CPI release."""
        trades = session.query(Trade).filter_by(
            reference_period=reference_period,
            status="open",
            mode=self.mode
        ).all()
        return sum(t.total_cost for t in trades if t.total_cost)
    
    # ─────────────────────────────────────────────
    # CIRCUIT BREAKERS
    # ─────────────────────────────────────────────
    
    def is_halted(self) -> bool:
        """Check if any circuit breaker is currently active."""
        session = get_session()
        try:
            active = session.query(CircuitBreakerEvent).filter_by(
                resolved=False
            ).filter(
                CircuitBreakerEvent.resume_at > datetime.utcnow()
            ).first()
            return active is not None
        finally:
            session.close()
    
    def trigger_circuit_breaker(self, reason: str, details: dict, halt_days: int):
        """Trigger a circuit breaker halt."""
        session = get_session()
        try:
            resume_at = datetime.utcnow() + timedelta(days=halt_days)
            event = CircuitBreakerEvent(
                reason=reason,
                details=details,
                halt_days=halt_days,
                resume_at=resume_at,
                resolved=False
            )
            session.add(event)
            session.commit()
            
            logger.warning(f"🚨 CIRCUIT BREAKER: {reason} | Halted for {halt_days} days")
            self._send_alert(f"CIRCUIT BREAKER TRIGGERED: {reason}", details)
            
        finally:
            session.close()
    
    def check_circuit_breakers(self):
        """Run all circuit breaker checks. Call daily."""
        session = get_session()
        try:
            portfolio = session.query(PaperPortfolio).order_by(
                PaperPortfolio.updated_at.desc()
            ).first()
            
            if not portfolio:
                return
            
            # Get monthly PnL
            thirty_days_ago = datetime.utcnow() - timedelta(days=30)
            monthly_trades = session.query(Trade).filter(
                Trade.trade_time >= thirty_days_ago,
                Trade.status == "settled",
                Trade.mode == self.mode
            ).all()
            
            monthly_pnl = sum(t.pnl_net for t in monthly_trades if t.pnl_net)
            
            # 1. Monthly drawdown check
            if portfolio.total_value > 0:
                drawdown_pct = abs(min(0, monthly_pnl)) / portfolio.total_value
                if drawdown_pct > self.max_monthly_drawdown:
                    self.trigger_circuit_breaker(
                        "MAX_MONTHLY_DRAWDOWN",
                        {"drawdown_pct": drawdown_pct, "monthly_pnl": monthly_pnl},
                        halt_days=30
                    )
            
            # 2. Consecutive losses check
            recent_trades = session.query(Trade).filter(
                Trade.status == "settled",
                Trade.mode == self.mode
            ).order_by(Trade.trade_time.desc()).limit(5).all()
            
            if len(recent_trades) >= 5:
                all_losses = all(t.pnl_net and t.pnl_net < 0 for t in recent_trades)
                if all_losses:
                    self.trigger_circuit_breaker(
                        "CONSECUTIVE_LOSSES",
                        {"count": 5},
                        halt_days=14
                    )
            
        except Exception as e:
            logger.error(f"Circuit breaker check failed: {e}")
        finally:
            session.close()
    
    def resolve_circuit_breaker(self, event_id: int):
        """Manually resolve a circuit breaker."""
        session = get_session()
        try:
            event = session.query(CircuitBreakerEvent).get(event_id)
            if event:
                event.resolved = True
                session.commit()
                logger.info(f"Circuit breaker {event_id} resolved")
        finally:
            session.close()
    
    def _send_alert(self, title: str, details: dict):
        """Send alert via Slack webhook if configured."""
        webhook_url = os.getenv("SLACK_WEBHOOK_URL")
        if not webhook_url:
            return
        
        try:
            import requests
            payload = {
                "text": f"🚨 *Kalshi Bot Alert*\n*{title}*\n```{details}```"
            }
            requests.post(webhook_url, json=payload, timeout=5)
        except Exception as e:
            logger.error(f"Alert send failed: {e}")
    
    # ─────────────────────────────────────────────
    # PORTFOLIO STATE
    # ─────────────────────────────────────────────
    
    def get_portfolio_value(self) -> float:
        """Get current portfolio value from DB."""
        session = get_session()
        try:
            portfolio = session.query(PaperPortfolio).order_by(
                PaperPortfolio.updated_at.desc()
            ).first()
            if portfolio:
                return portfolio.total_value
            return float(os.getenv("STARTING_PAPER_CAPITAL", 25000))
        finally:
            session.close()
    
    def update_portfolio(self, pnl_delta: float, exposure_delta: float = 0):
        """Update portfolio after trade settlement."""
        session = get_session()
        try:
            portfolio = session.query(PaperPortfolio).order_by(
                PaperPortfolio.updated_at.desc()
            ).first()
            if portfolio:
                portfolio.realized_pnl = (portfolio.realized_pnl or 0) + pnl_delta
                portfolio.cash_balance += pnl_delta
                portfolio.total_exposure = max(0, (portfolio.total_exposure or 0) + exposure_delta)
                portfolio.total_value = portfolio.cash_balance + portfolio.total_exposure
                portfolio.num_open_positions = max(0, (portfolio.num_open_positions or 0))
                session.commit()
        finally:
            session.close()
