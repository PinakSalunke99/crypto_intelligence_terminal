@echo off
REM WebSocket Real-Time Crypto Data Server Launcher (Windows)
REM Starts both the FastAPI WebSocket backend and Streamlit frontend

echo.
echo ==========================================
echo 🚀 Crypto Intelligence Real-Time Server
echo ==========================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    exit /b 1
)

echo 📦 Installing/Updating dependencies...
pip install -r requirements.txt -q

echo.
echo 🔄 Starting services...
echo.

REM Start the FastAPI WebSocket server in a new window
echo 📡 Starting WebSocket Server on http://localhost:8000
start "Crypto WS Server" cmd /k python websocket_server.py

REM Wait for server to start
timeout /t 3 /nobreak

REM Start the Streamlit dashboard
echo 🎨 Starting Streamlit Dashboard on http://localhost:8501
streamlit run streamlit_dashboard.py --client.showErrorDetails=true

pause
