"""
Crypto Intelligence WebSocket Client Library
Easy-to-use Python client for real-time price streaming
"""
import asyncio
import json
import logging
from typing import Callable, List, Optional, Dict, Any
import websockets
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CryptoWebSocketClient:
    """
    Async WebSocket client for real-time crypto price streaming.
    
    Example:
        client = CryptoWebSocketClient("ws://localhost:8000/ws")
        await client.connect()
        await client.subscribe(["BTC", "ETH", "SOL"])
        
        # Receive updates
        await client.listen()
    """
    
    def __init__(self, uri: str = "ws://localhost:8000/ws"):
        """
        Initialize WebSocket client.
        
        Args:
            uri: WebSocket server URI (default: localhost:8000)
        """
        self.uri = uri
        self.ws = None
        self.connected = False
        self.subscribed_symbols: set = set()
        self.callbacks: Dict[str, List[Callable]] = {
            "price_update": [],
            "connection_open": [],
            "connection_close": [],
            "error": []
        }
        self.price_cache: Dict[str, Dict[str, Any]] = {}
    
    async def connect(self, auto_reconnect: bool = True):
        """
        Connect to WebSocket server with optional auto-reconnect.
        
        Args:
            auto_reconnect: Automatically reconnect on disconnect
        """
        max_retries = 5
        retry_count = 0
        retry_delay = 1
        
        while retry_count < max_retries:
            try:
                logger.info(f"🔌 Connecting to {self.uri}")
                self.ws = await websockets.connect(self.uri)
                self.connected = True
                logger.info("✅ WebSocket connected")
                
                # Trigger connection callbacks
                for callback in self.callbacks["connection_open"]:
                    await self._run_callback(callback)
                
                if auto_reconnect:
                    # Auto-reconnect on disconnect
                    await self._handle_disconnect_and_reconnect()
                
                break
            
            except Exception as e:
                retry_count += 1
                logger.warning(f"⚠️ Connection failed (attempt {retry_count}/{max_retries}): {e}")
                
                if retry_count < max_retries:
                    await asyncio.sleep(retry_delay)
                    retry_delay = min(retry_delay * 2, 30)  # Exponential backoff
                else:
                    logger.error("❌ Failed to connect after max retries")
                    raise
    
    async def _handle_disconnect_and_reconnect(self):
        """Handle disconnection and reconnection logic."""
        try:
            await self.listen()
        except websockets.exceptions.ConnectionClosed:
            self.connected = False
            logger.warning("🔌 Connection closed, reconnecting in 3s...")
            
            for callback in self.callbacks["connection_close"]:
                await self._run_callback(callback)
            
            await asyncio.sleep(3)
            await self.connect(auto_reconnect=True)
    
    async def listen(self):
        """
        Listen for incoming messages from server.
        Should be run in background: asyncio.create_task(client.listen())
        """
        if not self.connected:
            raise RuntimeError("Not connected. Call connect() first")
        
        try:
            async for message in self.ws:
                try:
                    data = json.loads(message)
                    await self._handle_message(data)
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON: {message}")
        
        except websockets.exceptions.ConnectionClosed:
            self.connected = False
            logger.warning("Connection closed by server")
            
            for callback in self.callbacks["connection_close"]:
                await self._run_callback(callback)
    
    async def _handle_message(self, data: Dict[str, Any]):
        """Process incoming message from server."""
        msg_type = data.get("type", "unknown")
        
        if msg_type == "price_update":
            symbol = data.get("symbol")
            price_data = data.get("data", {})
            
            # Update cache
            self.price_cache[symbol] = price_data
            
            # Trigger callbacks
            for callback in self.callbacks["price_update"]:
                await self._run_callback(callback, symbol=symbol, data=price_data)
            
            logger.debug(f"💰 {symbol}: ${price_data.get('price', 0):,.2f}")
        
        elif msg_type == "subscription_confirmed":
            symbols = data.get("symbols", [])
            self.subscribed_symbols.update(symbols)
            logger.info(f"✅ Subscribed to: {symbols}")
        
        elif msg_type == "error":
            error = data.get("error", "Unknown error")
            logger.error(f"❌ Server error: {error}")
            
            for callback in self.callbacks["error"]:
                await self._run_callback(callback, error=error)
    
    async def subscribe(self, symbols: List[str]):
        """
        Subscribe to price updates for symbols.
        
        Args:
            symbols: List of symbol names (e.g., ["BTC", "ETH", "SOL"])
        """
        if not self.connected:
            raise RuntimeError("Not connected")
        
        symbols = [s.upper() for s in symbols]
        message = {
            "action": "subscribe",
            "symbols": symbols
        }
        
        await self.ws.send(json.dumps(message))
        logger.info(f"📡 Requested subscription to: {symbols}")
    
    async def unsubscribe(self, symbols: List[str]):
        """Unsubscribe from price updates."""
        if not self.connected:
            raise RuntimeError("Not connected")
        
        symbols = [s.upper() for s in symbols]
        message = {
            "action": "unsubscribe",
            "symbols": symbols
        }
        
        await self.ws.send(json.dumps(message))
        logger.info(f"Unsubscribed from: {symbols}")
    
    async def get_current_price(self, symbols: List[str]) -> Dict[str, Dict]:
        """
        Get current price for symbols.
        
        Args:
            symbols: List of symbol names
            
        Returns:
            Dictionary mapping symbols to price data
        """
        if not self.connected:
            raise RuntimeError("Not connected")
        
        symbols = [s.upper() for s in symbols]
        message = {
            "action": "get_current_price",
            "symbols": symbols
        }
        
        await self.ws.send(json.dumps(message))
        
        # Wait for responses (simplified - in production use callbacks)
        await asyncio.sleep(0.5)
        
        return {s: self.price_cache.get(s, {}) for s in symbols}
    
    def on_price_update(self, callback: Callable):
        """Register callback for price updates."""
        self.callbacks["price_update"].append(callback)
        return self
    
    def on_connect(self, callback: Callable):
        """Register callback for connection."""
        self.callbacks["connection_open"].append(callback)
        return self
    
    def on_disconnect(self, callback: Callable):
        """Register callback for disconnection."""
        self.callbacks["connection_close"].append(callback)
        return self
    
    def on_error(self, callback: Callable):
        """Register callback for errors."""
        self.callbacks["error"].append(callback)
        return self
    
    async def _run_callback(self, callback: Callable, **kwargs):
        """Safely run callback function."""
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(**kwargs)
            else:
                callback(**kwargs)
        except Exception as e:
            logger.error(f"Callback error: {e}")
    
    async def disconnect(self):
        """Disconnect from server."""
        if self.ws:
            await self.ws.close()
            self.connected = False
            logger.info("Disconnected")


# ========== SIMPLE SYNCHRONOUS WRAPPER ==========
class CryptoClient:
    """Simplified synchronous wrapper for WebSocket client."""
    
    def __init__(self, uri: str = "ws://localhost:8000/ws"):
        self.client = CryptoWebSocketClient(uri)
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
    
    def connect(self, auto_reconnect: bool = True):
        """Connect to server."""
        self.loop.run_until_complete(self.client.connect(auto_reconnect))
    
    def subscribe(self, symbols: List[str]):
        """Subscribe to price updates."""
        self.loop.run_until_complete(self.client.subscribe(symbols))
    
    def unsubscribe(self, symbols: List[str]):
        """Unsubscribe from price updates."""
        self.loop.run_until_complete(self.client.unsubscribe(symbols))
    
    def get_price(self, symbol: str) -> Optional[float]:
        """Get cached price for symbol."""
        return self.client.price_cache.get(symbol, {}).get("price")
    
    def get_prices(self) -> Dict[str, float]:
        """Get all cached prices."""
        return {
            symbol: data.get("price")
            for symbol, data in self.client.price_cache.items()
        }
    
    def on_update(self, callback: Callable):
        """Register callback for price updates."""
        self.client.on_price_update(callback)
        
        # Start listening in background thread
        import threading
        listener = threading.Thread(
            target=lambda: self.loop.run_until_complete(
                self.client.listen()
            ),
            daemon=True
        )
        listener.start()
    
    def disconnect(self):
        """Disconnect from server."""
        self.loop.run_until_complete(self.client.disconnect())


# ========== EXAMPLE USAGE ==========
async def main_async():
    """Example: Async usage (recommended)."""
    client = CryptoWebSocketClient()
    
    # Register callbacks
    async def on_price_update(symbol: str, data: Dict):
        print(f"💰 {symbol}: ${data.get('price', 0):,.2f} ({data.get('change_24h', 0):+.2f}%)")
    
    async def on_connect():
        print("✅ Connected to WebSocket server")
    
    client.on_price_update(on_price_update)
    client.on_connect(on_connect)
    
    # Connect and subscribe
    await client.connect()
    await client.subscribe(["BTC", "ETH", "SOL"])
    
    # Listen for updates
    await client.listen()


def main_sync():
    """Example: Synchronous usage (simpler)."""
    client = CryptoClient()
    
    def on_price_update(symbol: str, data: Dict):
        print(f"💰 {symbol}: ${data['price']:,.2f}")
    
    client.connect()
    client.subscribe(["BTC", "ETH"])
    client.on_update(on_price_update)
    
    # Keep running
    import time
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        client.disconnect()


if __name__ == "__main__":
    # Run async example
    # asyncio.run(main_async())
    
    # Or run sync example
    # main_sync()
    
    print("Import this module in your code:")
    print("from crypto_client import CryptoClient, CryptoWebSocketClient")
