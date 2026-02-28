"""
Main Trading Engine.
Orchestrates: forecast → market scan → edge calculation → order execution.
Runs on scheduled cron jobs.
"""

import os
import pickle
import numpy as np
from datetime import datetime, date, timedelta
from typing import List, Optional, Dict
from loguru import logger
from dotenv import load_dotenv

from data.database import (
    get_session, ModelForecast, KalshiMarketSnapshot, 
    Trade, PaperPortfolio, PerformanceLog
)
from data.ingestion import FeatureBuilder, run_data_ingestion
from models.forecaster import CPIForecaster, compute_brier_score, compute_brier_skill_score
from execution.kalshi_client import get_kalshi_client
from execution.risk_manager import RiskManager

load_dotenv()

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "cpi_forecaster.pkl")
FEE_RATE = 0.07  # Kalshi ~7% fee round-trip


class TradingEngine:
    
    def __init__(self):
        self.kalshi = get_kalshi_client()
        self.risk = RiskManager()
        self.feature_builder = FeatureBuilder()
        self.model = self._load_model()
        self.mode = os.getenv("TRADING_MODE", "paper")
        logger.info(f"🚀 TradingEngine initialized | Mode: {self.mode.upper()}")
    
    def _load_model(self) -> Optional[CPIForecaster]:
        """Load trained model, or return None if not yet trained."""
        try:
            m = CPIForecaster()
            m.load(MODEL_PATH)
            return m
        except FileNotFoundError:
            logger.warning("No trained model found. Run train_model() first.")
            return None
    
    # ─────────────────────────────────────────────
    # DAILY RUN (7:00 AM ET, Monday-Friday)
    # ─────────────────────────────────────────────
    
    def run_daily(self):
        """
        Main daily job. Runs at 7 AM ET on trading days.
        1. Ingest fresh data
        2. Update forecasts
        3. Scan markets for edge
        4. Execute trades where approved
        5. Check circuit breakers
        """
        logger.info(f"=== Daily Run: {datetime.utcnow().isoformat()} ===")
        
        # Step 1: Fresh data
        try:
            run_data_ingestion()
        except Exception as e:
            logger.error(f"Data ingestion failed: {e}")
        
        # Step 2: Check circuit breakers before anything
        self.risk.check_circuit_breakers()
        if self.risk.is_halted():
            logger.warning("System halted by circuit breaker. Skipping trade scan.")
            return
        
        # Step 3: Get current CPI markets
        try:
            markets = self.kalshi.get_cpi_markets()
            logger.info(f"Found {len(markets)} CPI markets")
        except Exception as e:
            logger.error(f"Market fetch failed: {e}")
            return
        
        # Step 4: For each market, evaluate and potentially trade
        for market in markets:
            try:
                self._evaluate_market(market)
            except Exception as e:
                logger.error(f"Market evaluation failed for {market.get('id')}: {e}")
        
        logger.info("=== Daily Run Complete ===")
    
    def _evaluate_market(self, market: dict):
        """Evaluate a single market and execute if edge found."""
        market_id = market.get("id", "")
        title = market.get("title", "")
        
        logger.info(f"Evaluating: {title} ({market_id})")
        
        # Parse bucket bounds from market title
        bucket = self._parse_bucket(market)
        if bucket is None:
            logger.debug(f"Could not parse bucket for {title}")
            return
        
        # Get fresh snapshot
        snapshot = self.kalshi.get_market_snapshot(market_id)
        if not snapshot:
            return
        
        # Store snapshot
        self._store_snapshot(snapshot, bucket)
        
        # Get or generate forecast for this release period
        reference_period = bucket.get("reference_period")
        if not reference_period:
            return
        
        forecast = self._get_or_generate_forecast(reference_period)
        if not forecast:
            return
        
        # Calculate model probability for this bucket
        p_model = self.model.bucket_probability(
            bucket["low"], bucket["high"],
            forecast["mu_model"], forecast["sigma_model"]
        )
        
        p_market = snapshot.get("mid_price", 0.5)
        
        # Calculate edge
        net_edge = p_model - p_market - FEE_RATE
        
        logger.info(f"  P_model={p_model:.3f} | P_market={p_market:.3f} | NetEdge={net_edge:.3f}")
        
        # Determine trade side
        if net_edge > self.risk.min_net_edge:
            side = "buy"   # We think it's underpriced
            trade_edge = net_edge
        elif net_edge < -self.risk.min_net_edge:
            side = "sell"  # We think it's overpriced
            trade_edge = abs(net_edge)
        else:
            logger.info(f"  No edge. Skipping.")
            return
        
        # Pre-trade risk checks
        release_dt = self._get_release_datetime(reference_period)
        check = self.risk.pre_trade_check(snapshot, net_edge, reference_period, release_dt)
        
        if not check["approved"]:
            logger.info(f"  Trade rejected: {check['reason']}")
            return
        
        # Size position
        portfolio_value = self.risk.get_portfolio_value()
        position_dollars = self.risk.compute_position_dollars(
            trade_edge, p_market, side, portfolio_value
        )
        
        if position_dollars < 10:
            logger.info(f"  Position too small (${position_dollars:.2f}). Skipping.")
            return
        
        price = snapshot["best_ask"] if side == "buy" else snapshot["best_bid"]
        contracts = self.risk.compute_contracts(position_dollars, price)
        
        if contracts < 1:
            return
        
        # Execute
        self._execute_trade(
            market_id=market_id,
            side=side,
            contracts=contracts,
            price=price,
            p_model=p_model,
            p_market=p_market,
            net_edge=net_edge,
            reference_period=reference_period,
            forecast=forecast
        )
    
    def _execute_trade(self, market_id: str, side: str, contracts: int,
                        price: float, p_model: float, p_market: float,
                        net_edge: float, reference_period: date, forecast: dict):
        """Execute trade and log to database."""
        session = get_session()
        try:
            price_cents = int(price * 100)
            kalshi_side = "yes" if side == "buy" else "no"
            
            # Place order
            result = self.kalshi.place_order(
                market_id=market_id,
                side=kalshi_side,
                action="buy",
                count=contracts,
                price_cents=price_cents
            )
            
            order_id = result.get("order", {}).get("id", "UNKNOWN")
            
            # Log trade to DB
            trade = Trade(
                market_id=market_id,
                reference_period=reference_period,
                side=side.upper(),
                contracts=contracts,
                price=price,
                total_cost=contracts * price,
                net_edge_at_entry=net_edge,
                p_model_at_entry=p_model,
                p_market_at_entry=p_market,
                kelly_fraction_used=float(os.getenv("KELLY_FRACTION", 0.25)),
                mode=self.mode,
                status="open",
                notes=f"order_id:{order_id}"
            )
            session.add(trade)
            session.commit()
            
            logger.info(f"✅ Trade placed: {side.upper()} {contracts}x {market_id} @ ${price:.2f} | NetEdge={net_edge:.3f}")
            
        except Exception as e:
            logger.error(f"Trade execution failed: {e}")
            session.rollback()
        finally:
            session.close()
    
    # ─────────────────────────────────────────────
    # SETTLEMENT
    # ─────────────────────────────────────────────
    
    def run_settlement_check(self):
        """
        Check for settled markets and record PnL.
        Runs at 9:00 AM ET on CPI release days.
        """
        logger.info("Running settlement check...")
        session = get_session()
        try:
            open_trades = session.query(Trade).filter_by(
                status="open", mode=self.mode
            ).all()
            
            for trade in open_trades:
                market = self.kalshi.get_market(trade.market_id)
                if market.get("status") in ["settled", "finalized"]:
                    result = market.get("result")
                    self._settle_trade(session, trade, result)
            
            session.commit()
            self._update_performance_metrics()
            
        finally:
            session.close()
    
    def _settle_trade(self, session, trade: Trade, result: str):
        """Record settlement outcome and PnL."""
        if result is None:
            return
        
        # Kalshi result is 'yes' or 'no'
        won = (
            (result == "yes" and trade.side == "BUY") or
            (result == "no" and trade.side == "SELL")
        )
        
        if won:
            gross_pnl = trade.contracts * (1.0 - trade.price)  # Profit on winning contracts
        else:
            gross_pnl = -trade.contracts * trade.price  # Loss
        
        fees = trade.contracts * trade.price * 0.07  # 7% fee
        net_pnl = gross_pnl - fees
        
        trade.status = "settled"
        trade.outcome = won
        trade.settlement_price = 1.0 if won else 0.0
        trade.pnl_gross = gross_pnl
        trade.pnl_net = net_pnl
        trade.fees_paid = fees
        
        # Update portfolio
        self.risk.update_portfolio(net_pnl, -trade.total_cost)
        
        emoji = "💰" if won else "📉"
        logger.info(f"{emoji} Settled: {trade.market_id} | Won={won} | PnL=${net_pnl:.2f}")
    
    # ─────────────────────────────────────────────
    # FORECAST MANAGEMENT
    # ─────────────────────────────────────────────
    
    def _get_or_generate_forecast(self, reference_period: date) -> Optional[dict]:
        """Get cached forecast or generate new one."""
        session = get_session()
        try:
            # Check for recent forecast (within 24 hours)
            existing = session.query(ModelForecast).filter_by(
                series_id="CPIAUCSL",
                reference_period=reference_period
            ).order_by(ModelForecast.created_at.desc()).first()
            
            if existing and (datetime.utcnow() - existing.created_at).total_seconds() < 86400:
                return {
                    "mu_model": existing.mu_model,
                    "sigma_model": existing.sigma_model
                }
            
            # Generate fresh forecast
            return self._generate_forecast(reference_period, session)
            
        finally:
            session.close()
    
    def _generate_forecast(self, reference_period: date, session=None) -> Optional[dict]:
        """Generate and store a new forecast."""
        if not self.model:
            logger.error("No model available")
            return None
        
        close_session = session is None
        if session is None:
            session = get_session()
        
        try:
            features = self.feature_builder.build_features(
                reference_period, as_of_date=date.today()
            )
            
            result = self.model.predict(
                features,
                cleveland_nowcast=features.get("cleveland_nowcast_mom")
            )
            
            # Store forecast
            forecast_record = ModelForecast(
                series_id="CPIAUCSL",
                forecast_date=date.today(),
                reference_period=reference_period,
                mu_model=result["mu_model"],
                sigma_model=result["sigma_model"],
                mu_cleveland=result.get("mu_cleveland"),
                mu_ridge=result.get("mu_ridge"),
                mu_xgboost=result.get("mu_xgboost"),
                weights_used=result.get("weights_used"),
                model_version=self.model.VERSION
            )
            session.add(forecast_record)
            session.commit()
            
            logger.info(f"Forecast generated: μ={result['mu_model']:.4f}, σ={result['sigma_model']:.4f}")
            return result
            
        except Exception as e:
            logger.error(f"Forecast generation failed: {e}")
            if close_session:
                session.rollback()
            return None
        finally:
            if close_session:
                session.close()
    
    # ─────────────────────────────────────────────
    # PERFORMANCE METRICS
    # ─────────────────────────────────────────────
    
    def _update_performance_metrics(self):
        """Compute and store performance metrics."""
        session = get_session()
        try:
            settled = session.query(Trade).filter_by(
                status="settled", mode=self.mode
            ).all()
            
            if not settled:
                return
            
            pnls = [t.pnl_net for t in settled if t.pnl_net is not None]
            outcomes = [1 if t.outcome else 0 for t in settled if t.outcome is not None]
            
            if not pnls:
                return
            
            total_pnl = sum(pnls)
            win_rate = sum(outcomes) / len(outcomes) if outcomes else 0
            
            # Sharpe (monthly resolution)
            if len(pnls) > 1:
                sharpe = np.mean(pnls) / (np.std(pnls) + 1e-8) * np.sqrt(12)
            else:
                sharpe = 0.0
            
            # Max drawdown
            cumulative = np.cumsum(pnls)
            max_drawdown = 0.0
            peak = cumulative[0]
            for val in cumulative:
                peak = max(peak, val)
                dd = (peak - val) / (peak + 1e-8)
                max_drawdown = max(max_drawdown, dd)
            
            portfolio = session.query(PaperPortfolio).order_by(
                PaperPortfolio.updated_at.desc()
            ).first()
            
            log = PerformanceLog(
                log_date=date.today(),
                mode=self.mode,
                realized_pnl=total_pnl,
                cumulative_pnl=total_pnl,
                portfolio_value=portfolio.total_value if portfolio else None,
                num_contracts_resolved=len(settled),
                win_rate=win_rate,
                sharpe_ratio=float(sharpe),
                max_drawdown=float(max_drawdown),
                avg_edge=float(np.mean([t.net_edge_at_entry for t in settled if t.net_edge_at_entry]))
            )
            session.add(log)
            session.commit()
            
            logger.info(f"📊 Performance: PnL=${total_pnl:.2f} | WinRate={win_rate:.1%} | Sharpe={sharpe:.2f}")
            
        except Exception as e:
            logger.error(f"Performance metrics update failed: {e}")
        finally:
            session.close()
    
    # ─────────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────────
    
    def _parse_bucket(self, market: dict) -> Optional[dict]:
        """
        Parse CPI bucket bounds from Kalshi market title/parameters.
        Markets look like: 'Will CPI be 3.1% to 3.2% in January?'
        """
        import re
        title = market.get("title", "")
        
        # Try to extract percentage range
        pattern = r'(\d+\.?\d*)\s*%?\s*(?:to|-)\s*(\d+\.?\d*)\s*%'
        match = re.search(pattern, title)
        
        if match:
            low = float(match.group(1))
            high = float(match.group(2))
        else:
            # Try "above X%" pattern
            above_match = re.search(r'above\s+(\d+\.?\d*)\s*%', title, re.IGNORECASE)
            below_match = re.search(r'below\s+(\d+\.?\d*)\s*%', title, re.IGNORECASE)
            
            if above_match:
                low = float(above_match.group(1))
                high = np.inf
            elif below_match:
                low = -np.inf
                high = float(below_match.group(1))
            else:
                return None
        
        # Try to extract reference period (month/year)
        reference_period = self._extract_reference_period(title, market)
        
        return {
            "low": low,
            "high": high,
            "reference_period": reference_period,
            "title": title
        }
    
    def _extract_reference_period(self, title: str, market: dict) -> Optional[date]:
        """Extract the CPI reference month from market title."""
        import re
        from dateutil.parser import parse as dateparse
        
        months = ["january", "february", "march", "april", "may", "june",
                  "july", "august", "september", "october", "november", "december"]
        
        title_lower = title.lower()
        for i, month in enumerate(months):
            if month in title_lower:
                year_match = re.search(r'20\d{2}', title)
                year = int(year_match.group()) if year_match else date.today().year
                return date(year, i + 1, 1)
        
        # Fall back to close_time from market metadata
        close_time = market.get("close_time")
        if close_time:
            try:
                dt = dateparse(close_time)
                return date(dt.year, dt.month, 1)
            except Exception:
                pass
        
        return None
    
    def _store_snapshot(self, snapshot: dict, bucket: dict):
        """Persist market snapshot to DB."""
        session = get_session()
        try:
            s = KalshiMarketSnapshot(
                market_id=snapshot["market_id"],
                series_id="CPIAUCSL",
                reference_period=bucket.get("reference_period"),
                snapshot_time=datetime.utcnow(),
                bucket_low=bucket.get("low"),
                bucket_high=bucket.get("high") if bucket.get("high") != np.inf else 999.0,
                best_bid=snapshot.get("best_bid"),
                best_ask=snapshot.get("best_ask"),
                mid_price=snapshot.get("mid_price"),
                spread=snapshot.get("spread"),
                volume_24h=snapshot.get("volume_24h"),
                open_interest=snapshot.get("open_interest")
            )
            session.add(s)
            session.commit()
        except Exception:
            session.rollback()
        finally:
            session.close()
    
    def _get_release_datetime(self, reference_period: date) -> Optional[datetime]:
        """BLS releases CPI at 8:30 AM ET. Return next expected release datetime."""
        # BLS releases on specific dates — approximate: ~2nd or 3rd week of following month
        # In production, pull exact dates from BLS release calendar
        next_month = date(reference_period.year + (reference_period.month // 12),
                         (reference_period.month % 12) + 1, 1)
        # Approximate: 12th of the following month at 13:30 UTC (8:30 AM ET)
        release_date = next_month.replace(day=12)
        return datetime(release_date.year, release_date.month, release_date.day, 13, 30, 0)
    
    # ─────────────────────────────────────────────
    # TRAINING
    # ─────────────────────────────────────────────
    
    def train_model(self, start_year: int = 1995):
        """Train the forecasting model. Run once to initialize."""
        logger.info("Starting model training...")
        
        df = self.feature_builder.build_training_dataset(start_year=start_year)
        
        model = CPIForecaster()
        model.train(df, verbose=True)
        model.save(MODEL_PATH)
        
        self.model = model
        logger.info("✅ Model training complete")
        return model


# ─────────────────────────────────────────────
# ENTRYPOINTS (called by scheduler)
# ─────────────────────────────────────────────

def run_morning_scan():
    """7:00 AM ET — main daily scan."""
    engine = TradingEngine()
    engine.run_daily()

def run_settlement_check():
    """9:00 AM ET on release days."""
    engine = TradingEngine()
    engine.run_settlement_check()

def train_and_save_model():
    """One-time model training."""
    engine = TradingEngine()
    engine.train_model()

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        train_and_save_model()
    else:
        run_morning_scan()
