"""
WebSocket-based Real-Time Crypto Data Server
Streams live price updates to connected clients with 0 delay
"""
import asyncio
import json
import logging
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import numpy as np
from datetime import datetime, timedelta
from data_ingestion import InstitutionalDataEngine
import threading

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ========== GLOBAL STATE ==========
class ConnectionManager:
    """Manages WebSocket connections and broadcasts data to all clients."""
    
    def __init__(self):
        self.active_connections: list[WebSocket] = []
        self.subscribed_symbols: dict[WebSocket, set[str]] = {}
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        self.subscribed_symbols[websocket] = set()
        logger.info(f"✅ Client connected. Total: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        if websocket in self.subscribed_symbols:
            del self.subscribed_symbols[websocket]
        logger.info(f"❌ Client disconnected. Total: {len(self.active_connections)}")
    
    async def broadcast_price(self, symbol: str, price_data: dict):
        """Send price update to all subscribed clients."""
        message = {
            "type": "price_update",
            "symbol": symbol,
            "data": price_data,
            "timestamp": datetime.now().isoformat()
        }
        
        disconnected = []
        for connection in self.active_connections:
            if symbol in self.subscribed_symbols.get(connection, set()):
                try:
                    await connection.send_json(message)
                except Exception as e:
                    logger.warning(f"⚠️ Failed to send to client: {e}")
                    disconnected.append(connection)
        
        # Clean up disconnected clients
        for conn in disconnected:
            self.disconnect(conn)
    
    async def send_personal_message(self, websocket: WebSocket, message: dict):
        """Send message to specific client."""
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.warning(f"⚠️ Failed to send personal message: {e}")
    
    def subscribe_client(self, websocket: WebSocket, symbol: str):
        """Subscribe a client to price updates for a symbol."""
        if websocket not in self.subscribed_symbols:
            self.subscribed_symbols[websocket] = set()
        self.subscribed_symbols[websocket].add(symbol)
        logger.info(f"📡 Client subscribed to {symbol}")
    
    def unsubscribe_client(self, websocket: WebSocket, symbol: str):
        """Unsubscribe a client from a symbol."""
        if websocket in self.subscribed_symbols:
            self.subscribed_symbols[websocket].discard(symbol)


manager = ConnectionManager()
data_engine = None
price_stream_task = None


# ========== BACKGROUND PRICE STREAM ==========
async def stream_prices(symbols: list[str], interval: float = 3.0):
    """
    Background task that continuously streams price updates to all subscribed clients.
    
    Args:
        symbols: List of crypto symbols to stream (e.g., ["BTC", "ETH", "SOL"])
        interval: Update interval in seconds (default: 3s for real-time feel)
    """
    logger.info(f"🚀 Price stream started for {symbols} with {interval}s interval")
    
    while True:
        try:
            for symbol in symbols:
                if not manager.active_connections:
                    await asyncio.sleep(interval)
                    continue
                
                try:
                    # Fetch latest price data
                    price_data = data_engine.get_sentiment_score(symbol)
                    
                    # Broadcast to all connected clients
                    await manager.broadcast_price(symbol, price_data)
                    
                except Exception as e:
                    logger.error(f"❌ Error fetching {symbol}: {e}")
            
            await asyncio.sleep(interval)
        
        except asyncio.CancelledError:
            logger.info("⏹️ Price stream cancelled")
            break
        except Exception as e:
            logger.error(f"❌ Stream error: {e}")
            await asyncio.sleep(interval)


# ========== LIFESPAN CONTEXT ==========
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage server startup and shutdown."""
    global data_engine, price_stream_task
    
    # Startup
    logger.info("🚀 Starting WebSocket Server...")
    data_engine = InstitutionalDataEngine()
    
    # Start background price streaming
    symbols = ["BTC", "ETH", "SOL", "BNB", "ADA"]
    price_stream_task = asyncio.create_task(stream_prices(symbols, interval=2.0))
    
    logger.info("✅ WebSocket Server Ready")
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down...")
    if price_stream_task:
        price_stream_task.cancel()
    logger.info("✅ Server shutdown complete")


# ========== FASTAPI APP ==========
app = FastAPI(
    title="Crypto Intelligence WebSocket Server",
    description="Real-time cryptocurrency price streaming via WebSocket",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ========== API ROUTES ==========
@app.get("/")
async def root():
    """Return API info."""
    return {
        "name": "Crypto Intelligence WebSocket Server",
        "version": "1.0.0",
        "endpoints": {
            "websocket": "ws://localhost:8000/ws",
            "health": "GET /health",
            "prices": "GET /prices/{symbols}",
            "price_history": "GET /price-history/{symbol}?days=7"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "connected_clients": len(manager.active_connections),
        "server_time": datetime.now().isoformat()
    }


@app.get("/prices/{symbols}")
async def get_prices(symbols: str):
    """
    Get current prices for multiple symbols.
    
    Example: GET /prices/BTC,ETH,SOL
    """
    symbol_list = [s.strip().upper() for s in symbols.split(",")]
    prices = {}
    
    for symbol in symbol_list:
        try:
            prices[symbol] = data_engine.get_sentiment_score(symbol)
        except Exception as e:
            prices[symbol] = {"error": str(e)}
    
    return {
        "timestamp": datetime.now().isoformat(),
        "prices": prices
    }


@app.get("/price-history/{symbol}")
async def get_price_history(symbol: str, days: int = Query(7, ge=1, le=365)):
    """
    Get historical price data for a symbol.
    
    Example: GET /price-history/BTC?days=7
    """
    symbol = symbol.upper()
    
    try:
        # Generate mock historical data (replace with real data from DB if available)
        prices = []
        now = datetime.now()
        
        for i in range(days):
            date = (now - timedelta(days=days-i-1)).isoformat()
            # Mock data - replace with actual historical data
            price = np.random.normal(loc=50000 if symbol == "BTC" else 2000, scale=5000)
            prices.append({
                "timestamp": date,
                "price": max(100, price)
            })
        
        return {
            "symbol": symbol,
            "period_days": days,
            "data": prices
        }
    except Exception as e:
        return {"error": str(e)}


# ========== WEBSOCKET ENDPOINT ==========
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    Main WebSocket endpoint for real-time price streaming.
    
    Client should send messages like:
    {"action": "subscribe", "symbols": ["BTC", "ETH", "SOL"]}
    or
    {"action": "unsubscribe", "symbols": ["BTC"]}
    """
    await manager.connect(websocket)
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_json()
            action = data.get("action", "").lower()
            symbols = data.get("symbols", [])
            
            logger.info(f"📨 Client action: {action}, symbols: {symbols}")
            
            # Handle subscribe action
            if action == "subscribe":
                for symbol in symbols:
                    manager.subscribe_client(websocket, symbol.upper())
                
                await manager.send_personal_message(
                    websocket,
                    {
                        "type": "subscription_confirmed",
                        "symbols": [s.upper() for s in symbols],
                        "message": f"Subscribed to {', '.join(symbols)}"
                    }
                )
            
            # Handle unsubscribe action
            elif action == "unsubscribe":
                for symbol in symbols:
                    manager.unsubscribe_client(websocket, symbol.upper())
                
                await manager.send_personal_message(
                    websocket,
                    {
                        "type": "unsubscribed",
                        "symbols": [s.upper() for s in symbols]
                    }
                )
            
            # Handle get_current_price action
            elif action == "get_current_price":
                for symbol in symbols:
                    try:
                        price_data = data_engine.get_sentiment_score(symbol.upper())
                        await manager.send_personal_message(
                            websocket,
                            {
                                "type": "current_price",
                                "symbol": symbol.upper(),
                                "data": price_data,
                                "timestamp": datetime.now().isoformat()
                            }
                        )
                    except Exception as e:
                        await manager.send_personal_message(
                            websocket,
                            {
                                "type": "error",
                                "symbol": symbol.upper(),
                                "error": str(e)
                            }
                        )
            
            else:
                await manager.send_personal_message(
                    websocket,
                    {
                        "type": "error",
                        "error": f"Unknown action: {action}"
                    }
                )
    
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"❌ WebSocket error: {e}")
        manager.disconnect(websocket)


if __name__ == "__main__":
    import uvicorn
    
    logger.info("🚀 Starting Crypto Intelligence WebSocket Server")
    logger.info("📡 WebSocket endpoint: ws://localhost:8000/ws")
    logger.info("🌐 API docs: http://localhost:8000/docs")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
