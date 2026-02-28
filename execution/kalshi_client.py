"""
Kalshi API Client.
Handles authentication, market data, order placement.
"""

import os
import time
import hmac
import hashlib
import base64
import requests
import json
from datetime import datetime, date
from typing import Optional, List, Dict
from loguru import logger
from dotenv import load_dotenv

load_dotenv()


class KalshiClient:
    """
    Kalshi REST API v2 client.
    Handles auth, market data polling, and order execution.
    """
    
    BASE_URL = os.getenv("KALSHI_BASE_URL", "https://trading.kalshi.com/trade-api/v2")
    
    def __init__(self):
        self.api_key = os.getenv("KALSHI_API_KEY", "")
        self.api_secret = os.getenv("KALSHI_API_SECRET", "")
        self.session = requests.Session()
        self._token = None
        self._token_expiry = 0
    
    def _get_headers(self, method: str, path: str) -> dict:
        """Generate authentication headers for Kalshi API."""
        timestamp = str(int(time.time() * 1000))
        
        # HMAC signature
        message = timestamp + method.upper() + path
        signature = hmac.new(
            self.api_secret.encode("utf-8"),
            message.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()
        
        return {
            "Content-Type": "application/json",
            "KALSHI-ACCESS-KEY": self.api_key,
            "KALSHI-ACCESS-TIMESTAMP": timestamp,
            "KALSHI-ACCESS-SIGNATURE": signature
        }
    
    def _request(self, method: str, endpoint: str, params: dict = None, 
                  body: dict = None, retries: int = 3) -> dict:
        """Make authenticated API request with retry logic."""
        url = f"{self.BASE_URL}{endpoint}"
        headers = self._get_headers(method, f"/trade-api/v2{endpoint}")
        
        for attempt in range(retries):
            try:
                if method.upper() == "GET":
                    r = self.session.get(url, headers=headers, params=params, timeout=15)
                elif method.upper() == "POST":
                    r = self.session.post(url, headers=headers, json=body, timeout=15)
                elif method.upper() == "DELETE":
                    r = self.session.delete(url, headers=headers, timeout=15)
                else:
                    raise ValueError(f"Unsupported method: {method}")
                
                if r.status_code == 429:
                    wait_time = 2 ** attempt
                    logger.warning(f"Rate limited. Waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                
                r.raise_for_status()
                return r.json()
                
            except requests.exceptions.RequestException as e:
                if attempt == retries - 1:
                    logger.error(f"API request failed after {retries} attempts: {e}")
                    raise
                time.sleep(1)
        
        return {}
    
    # ─────────────────────────────────────────────
    # ACCOUNT
    # ─────────────────────────────────────────────
    
    def get_balance(self) -> dict:
        """Get account balance."""
        return self._request("GET", "/portfolio/balance")
    
    def get_positions(self) -> list:
        """Get all open positions."""
        result = self._request("GET", "/portfolio/positions")
        return result.get("market_positions", [])
    
    # ─────────────────────────────────────────────
    # MARKETS
    # ─────────────────────────────────────────────
    
    def search_markets(self, keyword: str = "CPI", status: str = "open", 
                        limit: int = 50) -> list:
        """Search for active markets by keyword."""
        params = {
            "status": status,
            "limit": limit,
        }
        result = self._request("GET", "/markets", params=params)
        markets = result.get("markets", [])
        
        # Filter by keyword in title
        if keyword:
            markets = [m for m in markets if keyword.lower() in m.get("title", "").lower()]
        
        return markets
    
    def get_market(self, market_id: str) -> dict:
        """Get details for a specific market."""
        return self._request("GET", f"/markets/{market_id}")
    
    def get_orderbook(self, market_id: str) -> dict:
        """Get full order book for a market."""
        return self._request("GET", f"/markets/{market_id}/orderbook")
    
    def get_market_snapshot(self, market_id: str) -> dict:
        """
        Get best bid/ask, volume, open interest for a market.
        Returns clean snapshot dict.
        """
        try:
            market = self.get_market(market_id)
            book = self.get_orderbook(market_id)
            
            yes_bids = book.get("orderbook", {}).get("yes", [])
            no_bids = book.get("orderbook", {}).get("no", [])
            
            # Best bid/ask in cents (Kalshi uses cents, we normalize to 0-1)
            best_bid_cents = yes_bids[0][0] if yes_bids else 0
            best_ask_cents = (100 - no_bids[0][0]) if no_bids else 100
            
            best_bid = best_bid_cents / 100
            best_ask = best_ask_cents / 100
            mid = (best_bid + best_ask) / 2
            spread = best_ask - best_bid
            
            return {
                "market_id": market_id,
                "title": market.get("title", ""),
                "status": market.get("status", ""),
                "best_bid": round(best_bid, 4),
                "best_ask": round(best_ask, 4),
                "mid_price": round(mid, 4),
                "spread": round(spread, 4),
                "volume_24h": market.get("volume_24h", 0),
                "open_interest": market.get("open_interest", 0),
                "close_time": market.get("close_time"),
                "result": market.get("result"),
                "snapshot_time": datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"Failed to get snapshot for {market_id}: {e}")
            return {}
    
    def get_cpi_markets(self) -> list:
        """Find all open CPI-related markets."""
        markets = self.search_markets(keyword="CPI", status="open")
        # Also search for "inflation" 
        inflation_markets = self.search_markets(keyword="inflation", status="open")
        all_markets = {m["id"]: m for m in markets + inflation_markets}
        return list(all_markets.values())
    
    # ─────────────────────────────────────────────
    # ORDERS
    # ─────────────────────────────────────────────
    
    def place_order(self, market_id: str, side: str, action: str,
                     count: int, price_cents: int) -> dict:
        """
        Place a limit order.
        
        side: 'yes' or 'no'
        action: 'buy' or 'sell'  
        count: number of contracts
        price_cents: limit price in cents (1-99)
        """
        body = {
            "market_id": market_id,
            "side": side,
            "action": action,
            "count": count,
            "type": "limit",
            "yes_price": price_cents if side == "yes" else (100 - price_cents),
            "no_price": (100 - price_cents) if side == "yes" else price_cents,
        }
        
        logger.info(f"Placing order: {market_id} | {side} {action} x{count} @ {price_cents}¢")
        return self._request("POST", "/portfolio/orders", body=body)
    
    def cancel_order(self, order_id: str) -> dict:
        """Cancel a specific order."""
        return self._request("DELETE", f"/portfolio/orders/{order_id}")
    
    def cancel_all_orders(self) -> int:
        """Cancel all open orders. Returns count cancelled."""
        orders = self._request("GET", "/portfolio/orders", params={"status": "resting"})
        open_orders = orders.get("orders", [])
        
        cancelled = 0
        for order in open_orders:
            try:
                self.cancel_order(order["id"])
                cancelled += 1
            except Exception as e:
                logger.error(f"Failed to cancel order {order['id']}: {e}")
        
        logger.info(f"Cancelled {cancelled} orders")
        return cancelled
    
    def get_order_status(self, order_id: str) -> dict:
        """Check status of an order."""
        return self._request("GET", f"/portfolio/orders/{order_id}")
    
    def get_fills(self, market_id: str = None) -> list:
        """Get trade history / fills."""
        params = {}
        if market_id:
            params["market_id"] = market_id
        result = self._request("GET", "/portfolio/fills", params=params)
        return result.get("fills", [])


# ─────────────────────────────────────────────
# MOCK CLIENT FOR PAPER TRADING
# ─────────────────────────────────────────────

class MockKalshiClient(KalshiClient):
    """
    Paper trading mock — simulates Kalshi API without real money.
    Uses real market data but doesn't execute real orders.
    """
    
    def __init__(self):
        super().__init__()
        self._mock_orders = {}
        self._order_counter = 1000
        logger.info("🎭 MockKalshiClient initialized (PAPER TRADING MODE)")
    
    def place_order(self, market_id: str, side: str, action: str,
                     count: int, price_cents: int) -> dict:
        """Simulate order placement."""
        order_id = f"PAPER_{self._order_counter}"
        self._order_counter += 1
        
        order = {
            "id": order_id,
            "market_id": market_id,
            "side": side,
            "action": action,
            "count": count,
            "price_cents": price_cents,
            "status": "resting",
            "created_time": datetime.utcnow().isoformat(),
            "is_paper": True
        }
        self._mock_orders[order_id] = order
        
        logger.info(f"📝 PAPER ORDER: {market_id} | {side} {action} x{count} @ {price_cents}¢ → {order_id}")
        return {"order": order}
    
    def cancel_order(self, order_id: str) -> dict:
        if order_id in self._mock_orders:
            self._mock_orders[order_id]["status"] = "cancelled"
            logger.info(f"📝 PAPER CANCEL: {order_id}")
        return {"status": "ok"}
    
    def get_balance(self) -> dict:
        """Return mock balance."""
        return {"balance": 25000_00}  # In cents
    
    def get_positions(self) -> list:
        return []


def get_kalshi_client():
    """Factory — returns mock or live client based on TRADING_MODE env var."""
    mode = os.getenv("TRADING_MODE", "paper").lower()
    if mode == "paper":
        return MockKalshiClient()
    else:
        return KalshiClient()
