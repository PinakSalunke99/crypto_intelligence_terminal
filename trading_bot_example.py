"""
Example Trading Bot using WebSocket Real-Time Price Streaming
Demonstrates how to use the crypto_client for trading strategies
"""
import asyncio
import json
from datetime import datetime
from crypto_client import CryptoWebSocketClient
from typing import Dict, Any


class SimpleTradingBot:
    """
    Example trading bot that reacts to price movements.
    
    Strategies:
    1. Price spike detection: Alert on 5% change in 2 minutes
    2. Support/Resistance: Track high/low levels
    3. Momentum: Simple moving average crossover
    """
    
    def __init__(self, symbols: list, alert_threshold: float = 5.0):
        """
        Initialize trading bot.
        
        Args:
            symbols: List of symbols to monitor (e.g., ["BTC", "ETH"])
            alert_threshold: Alert if price changes by this % (default: 5%)
        """
        self.client = CryptoWebSocketClient("ws://localhost:8000/ws")
        self.symbols = [s.upper() for s in symbols]
        self.alert_threshold = alert_threshold
        
        # Price history for moving average
        self.price_history: Dict[str, list] = {s: [] for s in self.symbols}
        self.moving_average_period = 10  # 10 updates
        
        # Entry/exit prices for position tracking
        self.positions: Dict[str, Dict[str, Any]] = {}
        
        # Statistics
        self.stats = {
            "total_alerts": 0,
            "price_updates": 0,
            "profit_trades": 0,
            "loss_trades": 0,
            "total_pnl": 0.0
        }
    
    async def start(self):
        """Connect and start bot."""
        print("🤖 Starting Trading Bot...")
        print(f"📡 Monitoring: {', '.join(self.symbols)}")
        print(f"⚠️  Alert threshold: {self.alert_threshold}%")
        print("-" * 60)
        
        # Register callbacks
        self.client.on_price_update(self.on_price_update)
        self.client.on_connect(self.on_connect)
        self.client.on_error(self.on_error)
        
        # Connect and subscribe
        await self.client.connect()
        await self.client.subscribe(self.symbols)
        
        # Start listening
        await self.client.listen()
    
    async def on_connect(self):
        """Called when connected."""
        print(f"✅ Connected to WebSocket server at {datetime.now().strftime('%H:%M:%S')}")
    
    async def on_price_update(self, symbol: str, data: Dict[str, Any]):
        """
        Called every time price updates.
        This is where your trading logic goes.
        """
        price = data.get('price', 0)
        change_24h = data.get('change_24h', 0)
        self.stats["price_updates"] += 1
        
        # ===== STRATEGY 1: PRICE SPIKE DETECTION =====
        if abs(change_24h) >= self.alert_threshold:
            self.stats["total_alerts"] += 1
            direction = "🔴 DOWN" if change_24h < 0 else "🟢 UP"
            await self._alert(
                f"{direction} SPIKE | {symbol}: ${price:,.2f} ({change_24h:+.2f}%)"
            )
        
        # ===== STRATEGY 2: SUPPORT/RESISTANCE =====
        await self._track_levels(symbol, price)
        
        # ===== STRATEGY 3: MOVING AVERAGE CROSSOVER =====
        await self._moving_average_strategy(symbol, price)
        
        # ===== DISPLAY REAL-TIME DATA =====
        await self._display_price(symbol, price, change_24h)
    
    async def _alert(self, message: str):
        """Send trading alert."""
        self.stats["total_alerts"] += 1
        print(f"\n🚨 ALERT: {message}")
        print(f"   Time: {datetime.now().strftime('%H:%M:%S')}")
        # In production, you would:
        # - Send email/SMS
        # - Place order
        # - Update database
        # - Trigger webhook
    
    async def _track_levels(self, symbol: str, price: float):
        """Track support and resistance levels."""
        if symbol not in self.price_history:
            return
        
        history = self.price_history[symbol]
        history.append(price)
        
        # Keep last 100 prices
        if len(history) > 100:
            history.pop(0)
        
        if len(history) < 5:
            return
        
        # Calculate support and resistance
        high = max(history)
        low = min(history)
        current = price
        
        # Alert if near support/resistance
        if abs(current - low) < 50 and current > low * 1.01:  # Near support
            print(f"📊 {symbol}: Approaching SUPPORT at ${low:,.2f}")
        elif abs(current - high) < 50 and current < high * 0.99:  # Near resistance
            print(f"📊 {symbol}: Approaching RESISTANCE at ${high:,.2f}")
    
    async def _moving_average_strategy(self, symbol: str, price: float):
        """Simple moving average crossover strategy."""
        if symbol not in self.price_history:
            return
        
        history = self.price_history[symbol]
        history.append(price)
        
        if len(history) > 100:
            history.pop(0)
        
        if len(history) < self.moving_average_period:
            return
        
        # Calculate moving average
        ma_fast = sum(history[-5:]) / 5
        ma_slow = sum(history[-self.moving_average_period:]) / self.moving_average_period
        
        # Crossover signals
        if len(history) > self.moving_average_period + 1:
            prev_price = history[-2]
            prev_ma_fast = sum(history[-6:-1]) / 5 if len(history) >= 6 else ma_fast
            
            # Golden cross: Fast MA crosses above Slow MA
            if prev_ma_fast <= prev_ma_fast and ma_fast > ma_slow:
                await self._alert(f"{symbol}: 🟢 GOLDEN CROSS - BUY SIGNAL")
                self._open_position(symbol, price, "LONG")
            
            # Death cross: Fast MA crosses below Slow MA
            elif prev_ma_fast >= ma_slow and ma_fast < ma_slow:
                await self._alert(f"{symbol}: 🔴 DEATH CROSS - SELL SIGNAL")
                self._close_position(symbol, price)
    
    def _open_position(self, symbol: str, entry_price: float, side: str):
        """Open a trading position."""
        self.positions[symbol] = {
            "entry_price": entry_price,
            "side": side,
            "entry_time": datetime.now(),
            "quantity": 1.0  # Simplified
        }
        print(f"📈 OPENED {side} position on {symbol} @ ${entry_price:,.2f}")
    
    def _close_position(self, symbol: str, exit_price: float):
        """Close a trading position."""
        if symbol not in self.positions:
            return
        
        pos = self.positions[symbol]
        entry = pos['entry_price']
        pnl = (exit_price - entry) * pos['quantity']
        pnl_pct = (pnl / entry) * 100
        
        self.stats["total_pnl"] += pnl
        
        if pnl > 0:
            self.stats["profit_trades"] += 1
            print(f"📊 CLOSED position on {symbol}")
            print(f"   Entry: ${entry:,.2f} | Exit: ${exit_price:,.2f}")
            print(f"   P&L: ${pnl:+.2f} ({pnl_pct:+.2f}%) ✅")
        else:
            self.stats["loss_trades"] += 1
            print(f"📊 CLOSED position on {symbol}")
            print(f"   Entry: ${entry:,.2f} | Exit: ${exit_price:,.2f}")
            print(f"   P&L: ${pnl:+.2f} ({pnl_pct:+.2f}%) ❌")
        
        del self.positions[symbol]
    
    async def _display_price(self, symbol: str, price: float, change_24h: float):
        """Display current price (non-blocking)."""
        emoji = "🟢" if change_24h >= 0 else "🔴"
        # Don't print every update to avoid spam
        if self.stats["price_updates"] % 10 == 0:
            print(f"{emoji} {symbol}: ${price:,.2f} ({change_24h:+.2f}%)", end="\r")
    
    async def on_error(self, error: str):
        """Called on error."""
        print(f"❌ Error: {error}")
    
    def print_stats(self):
        """Print trading statistics."""
        print("\n" + "=" * 60)
        print("📊 TRADING BOT STATISTICS")
        print("=" * 60)
        print(f"Price Updates: {self.stats['price_updates']}")
        print(f"Total Alerts: {self.stats['total_alerts']}")
        print(f"Profit Trades: {self.stats['profit_trades']}")
        print(f"Loss Trades: {self.stats['loss_trades']}")
        print(f"Total P&L: ${self.stats['total_pnl']:+,.2f}")
        print(f"Win Rate: {self._calculate_win_rate():.1f}%")
        print("=" * 60)
    
    def _calculate_win_rate(self) -> float:
        """Calculate win rate."""
        total = self.stats["profit_trades"] + self.stats["loss_trades"]
        if total == 0:
            return 0.0
        return (self.stats["profit_trades"] / total) * 100


# ========== MARKET ANALYZER BOT ==========
class MarketAnalyzerBot:
    """Analyzes market conditions and provides insights."""
    
    def __init__(self, symbols: list):
        self.client = CryptoWebSocketClient()
        self.symbols = [s.upper() for s in symbols]
        self.market_data = {}
    
    async def start(self):
        """Start market analysis."""
        print("🧠 Starting Market Analyzer...")
        
        self.client.on_price_update(self.analyze)
        
        await self.client.connect()
        await self.client.subscribe(self.symbols)
        await self.client.listen()
    
    async def analyze(self, symbol: str, data: Dict[str, Any]):
        """Analyze market data."""
        self.market_data[symbol] = data
        
        # Calculate market metrics
        total_change = sum(d.get('change_24h', 0) for d in self.market_data.values())
        avg_change = total_change / len(self.market_data) if self.market_data else 0
        
        # Market sentiment
        if avg_change > 2:
            sentiment = "🟢 BULLISH"
        elif avg_change < -2:
            sentiment = "🔴 BEARISH"
        else:
            sentiment = "🟡 NEUTRAL"
        
        if [v for v in self.market_data.values() if v.get('change_24h', 0) > 0]:
            if len(self.market_data) > 0:
                bullish_count = sum(1 for v in self.market_data.values() if v.get('change_24h', 0) > 0)
                print(f"{sentiment} | {bullish_count}/{len(self.market_data)} symbols up")


# ========== MAIN ==========
async def main():
    """Run example trading bot."""
    
    # Example 1: Simple Trading Bot
    print("\n" + "=" * 60)
    print("🤖 EXAMPLE 1: SIMPLE TRADING BOT")
    print("=" * 60)
    
    bot = SimpleTradingBot(symbols=["BTC", "ETH", "SOL"], alert_threshold=2.0)
    
    # Run for demonstration
    try:
        await asyncio.wait_for(bot.start(), timeout=60)
    except asyncio.TimeoutError:
        print("\n⏱️ Demo timeout - stopping bot")
        bot.print_stats()
    except KeyboardInterrupt:
        print("\n⏹️ Bot stopped by user")
        bot.print_stats()


async def main_analyzer():
    """Run market analyzer."""
    print("\n" + "=" * 60)
    print("🧠 EXAMPLE 2: MARKET ANALYZER")
    print("=" * 60)
    
    analyzer = MarketAnalyzerBot(symbols=["BTC", "ETH", "SOL", "BNB", "ADA"])
    
    try:
        await asyncio.wait_for(analyzer.start(), timeout=60)
    except (asyncio.TimeoutError, KeyboardInterrupt):
        print("\nMarket analyzer stopped")


if __name__ == "__main__":
    """Uncomment to run examples"""
    
    print("\n📖 TRADING BOT EXAMPLES")
    print("Uncomment the code below to run examples:\n")
    print("1. Simple Trading Bot: asyncio.run(main())")
    print("2. Market Analyzer: asyncio.run(main_analyzer())")
    print("\nMake sure websocket_server.py is running first!")
    
    # Uncomment to run:
    # import sys
    # if len(sys.argv) > 1 and sys.argv[1] == "trading":
    #     asyncio.run(main())
    # else:
    #     asyncio.run(main_analyzer())
