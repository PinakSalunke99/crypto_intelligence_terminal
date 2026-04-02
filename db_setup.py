import psycopg2
import sqlite3
import os
from dotenv import load_dotenv

load_dotenv()

def init_db():
    conn = None
    is_sqlite = False
    
    try:
        # Try connecting to PostgreSQL
        conn = psycopg2.connect(
            dbname=os.getenv('DB_NAME'),
            user=os.getenv('DB_USER'),
            password=os.getenv('DB_PASSWORD'),
            host=os.getenv('DB_HOST', 'localhost'),
            port=os.getenv('DB_PORT', '5432'),
            connect_timeout=3
        )
        print("🗄️  Connected to PostgreSQL")
    except Exception as e:
        print(f"⚠️  PostgreSQL connection failed: {e}")
        print("📁 Falling back to local SQLite database (crypto_terminal.db)")
        conn = sqlite3.connect('crypto_terminal.db')
        is_sqlite = True

    try:
        cur = conn.cursor()
        
        # Adjust SQL syntax for SQLite if needed (SERIAL -> INTEGER PRIMARY KEY AUTOINCREMENT)
        serial_type = "SERIAL PRIMARY KEY" if not is_sqlite else "INTEGER PRIMARY KEY AUTOINCREMENT"
        
        # Price History Table
        cur.execute(f'''
            CREATE TABLE IF NOT EXISTS price_history (
                id {serial_type},
                symbol VARCHAR(10),
                timestamp TIMESTAMP,
                close_price NUMERIC
            )
        ''')
        
        # Sentiment & Signal Logs Table
        timestamp_default = "DEFAULT CURRENT_TIMESTAMP" if not is_sqlite else "DEFAULT (datetime('now','localtime'))"
        
        cur.execute(f'''
            CREATE TABLE IF NOT EXISTS sentiment_logs (
                id {serial_type},
                timestamp TIMESTAMP {timestamp_default},
                asset VARCHAR(10),
                sentiment_label VARCHAR(20),
                sentiment_score NUMERIC,
                reasoning TEXT
            )
        ''')
        
        conn.commit()
        cur.close()
        conn.close()
        print(f"✅ SUCCESS: {'SQLite' if is_sqlite else 'PostgreSQL'} tables initialized.")
    except Exception as e:
        print(f"❌ DATABASE ERROR during initialization: {e}")

if __name__ == "__main__":
    init_db()