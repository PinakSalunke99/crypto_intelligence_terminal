#!/usr/bin/env python3
"""
Setup Verification Script
Checks if all dependencies and services are properly configured
"""
import sys
import subprocess
import importlib
import socket
from pathlib import Path
from datetime import datetime


def print_header(text: str):
    """Print colored header."""
    print(f"\n{'=' * 60}")
    print(f"🔍 {text}")
    print(f"{'=' * 60}")


def print_check(status: bool, message: str, details: str = ""):
    """Print check result."""
    icon = "✅" if status else "❌"
    print(f"{icon} {message}")
    if details:
        print(f"   └─ {details}")


def check_python_version():
    """Check Python version."""
    print_header("PYTHON VERSION")
    version = sys.version_info
    required_version = (3, 8)
    
    status = version >= required_version
    print_check(
        status,
        f"Python {version.major}.{version.minor}.{version.micro}",
        f"Required: >= {required_version[0]}.{required_version[1]}"
    )
    return status


def check_dependencies():
    """Check if all required packages are installed."""
    print_header("DEPENDENCIES")
    
    required_packages = {
        'fastapi': 'FastAPI Web Framework',
        'uvicorn': 'ASGI Server',
        'websockets': 'WebSocket Protocol',
        'streamlit': 'Dashboard Framework',
        'pandas': 'Data Processing',
        'numpy': 'Numerical Computing',
        'requests': 'HTTP Client',
        'plotly': 'Plotting Library',
        'psycopg2': 'PostgreSQL Driver',
        'python-binance': 'Binance API',
        'prophet': 'Time Series Forecasting',
        'ollama': 'AI Integration',
        'tweepy': 'Twitter API',
    }
    
    all_installed = True
    for package, description in required_packages.items():
        try:
            importlib.import_module(package)
            print_check(True, f"{package:20} - {description}")
        except ImportError:
            print_check(False, f"{package:20} - {description}")
            all_installed = False
    
    return all_installed


def check_ports():
    """Check if required ports are available."""
    print_header("PORT AVAILABILITY")
    
    ports = {
        8000: "WebSocket Server",
        8501: "Streamlit Dashboard",
    }
    
    all_available = True
    for port, service in ports.items():
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('127.0.0.1', port))
        sock.close()
        
        if result == 0:
            # Port is in use
            print_check(False, f"Port {port:5} - {service:25} (IN USE)")
            all_available = False
        else:
            # Port is free
            print_check(True, f"Port {port:5} - {service:25} (Available)")
    
    return all_available


def check_files():
    """Check if all required files exist."""
    print_header("PROJECT FILES")
    
    required_files = {
        'websocket_server.py': 'WebSocket Server',
        'streamlit_dashboard.py': 'Streamlit Dashboard',
        'frontend.html': 'HTML Frontend',
        'crypto_client.py': 'WebSocket Client Library',
        'trading_bot_example.py': 'Trading Bot Example',
        'requirements.txt': 'Dependencies List',
        'data_ingestion.py': 'Data Engine',
        'brain.py': 'AI Brain',
        'start_server.bat': 'Windows Launcher',
        'start_server.sh': 'Linux/Mac Launcher',
    }
    
    all_exist = True
    for filename, description in required_files.items():
        path = Path(filename)
        exists = path.exists()
        print_check(exists, f"{filename:30} - {description}")
        if not exists:
            all_exist = False
    
    return all_exist


def check_env_file():
    """Check for .env configuration file."""
    print_header("ENVIRONMENT CONFIGURATION")
    
    env_path = Path('.env')
    
    if env_path.exists():
        print_check(True, ".env file exists")
        
        # Check for key environment variables
        with open(env_path, 'r') as f:
            content = f.read()
        
        keys = ['COINGECKO_API_KEY', 'TWITTER_BEARER_TOKEN', 'ETHERSCAN_API_KEY']
        for key in keys:
            if key in content:
                print_check(True, f"{key:25} configured")
            else:
                print_check(False, f"{key:25} not configured (optional)")
    else:
        print_check(False, ".env file not found (optional but recommended)")
        print("\n   Create .env file with:")
        print("   COINGECKO_API_KEY=your_key")
        print("   TWITTER_BEARER_TOKEN=your_token")
        print("   ETHERSCAN_API_KEY=your_key")


def check_api_connectivity():
    """Check connectivity to external APIs."""
    print_header("API CONNECTIVITY")
    
    apis = {
        'https://api.binance.com/api/v3': 'Binance API',
        'https://api.coingecko.com/api/v3': 'CoinGecko API',
    }
    
    import requests
    
    all_connected = True
    for url, name in apis.items():
        try:
            response = requests.get(url, timeout=5)
            print_check(True, f"{name:20} - {response.status_code}")
        except Exception as e:
            print_check(False, f"{name:20} - {str(e)[:40]}")
            all_connected = False
    
    return all_connected


def print_summary(checks: dict):
    """Print verification summary."""
    print_header("VERIFICATION SUMMARY")
    
    total = len(checks)
    passed = sum(1 for v in checks.values() if v)
    
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("\n✅ All checks passed! Ready to run.")
        print("\nNext steps:")
        print("1. Run: python websocket_server.py")
        print("2. In another terminal: streamlit run streamlit_dashboard.py")
        print("3. Open: http://localhost:8501")
    else:
        print(f"\n⚠️ {total - passed} issue(s) found. See above for details.")
        print("\nTo fix:")
        print("1. Install missing dependencies: pip install -r requirements.txt")
        print("2. Free up ports 8000 and 8501")
        print("3. Create .env file with required API keys")


def run_quick_test():
    """Run quick test of WebSocket connection."""
    print_header("QUICK CONNECTION TEST")
    
    try:
        import asyncio
        from crypto_client import CryptoWebSocketClient
        
        async def test():
            client = CryptoWebSocketClient()
            try:
                await asyncio.wait_for(client.connect(), timeout=5)
                print_check(True, "WebSocket connection successful")
                await client.disconnect()
                return True
            except asyncio.TimeoutError:
                print_check(False, "WebSocket server not responding (start websocket_server.py first)")
                return False
            except Exception as e:
                print_check(False, f"Connection error: {str(e)[:50]}")
                return False
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(test())
        loop.close()
        
        return result
    except Exception as e:
        print_check(False, f"Test could not run: {str(e)[:50]}")
        return False


def main():
    """Run all verification checks."""
    print("\n" + "=" * 60)
    print("🔧 CRYPTO INTELLIGENCE SETUP VERIFICATION")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    checks = {
        'Python Version': check_python_version(),
        'Dependencies': check_dependencies(),
        'Port Availability': check_ports(),
        'Project Files': check_files(),
    }
    
    # Optional checks
    check_env_file()
    print()
    
    try:
        checks['API Connectivity'] = check_api_connectivity()
    except Exception as e:
        print(f"⚠️ Could not check API connectivity: {e}")
    
    # Print summary
    print_summary(checks)
    
    # Ask about quick test
    print("\n" + "-" * 60)
    try:
        response = input("Run quick WebSocket test? (y/n): ").lower().strip()
        if response == 'y':
            run_quick_test()
    except:
        pass
    
    print("\n" + "=" * 60)
    print("✅ VERIFICATION COMPLETE\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️ Verification cancelled")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error during verification: {e}")
        sys.exit(1)
