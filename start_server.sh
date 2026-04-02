#!/bin/bash
# WebSocket Real-Time Crypto Data Server Launcher
# Starts both the FastAPI WebSocket backend and HTML frontend

echo "=========================================="
echo "🚀 Crypto Intelligence Real-Time Server"
echo "=========================================="
echo ""

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "❌ Python is not installed"
    exit 1
fi

echo "📦 Installing/Updating dependencies..."
pip install -r requirements.txt -q

echo ""
echo "🔄 Starting services..."
echo ""

# Start the FastAPI WebSocket server in background
echo "📡 Starting WebSocket Server on http://localhost:8000"
python websocket_server.py &
SERVER_PID=$!

# Wait for server to start
sleep 3

# Start the Streamlit dashboard in the foreground
echo "🎨 Starting Streamlit Dashboard on http://localhost:8501"
streamlit run streamlit_dashboard.py --client.showErrorDetails=true

# Cleanup on exit
trap "kill $SERVER_PID" EXIT
