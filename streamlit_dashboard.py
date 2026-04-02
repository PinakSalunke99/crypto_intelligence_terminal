"""
Streamlit WebSocket Integration for Real-Time Crypto Dashboard
This version connects to the WebSocket server for live price updates
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import asyncio
import json
import websockets
from data_ingestion import InstitutionalDataEngine
import time

# ===== PAGE CONFIG =====
st.set_page_config(
    page_title="Crypto Intelligence - Real-Time Dashboard",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ===== CUSTOM CSS =====
st.markdown("""
    <style>
        /* Main styling */
        [data-testid="stMetric"] {
            background-color: #1a1b3d;
            padding: 15px;
            border-radius: 8px;
            border: 1px solid #334155;
        }
        
        /* Price cards */
        .price-ticker {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin: 1rem 0;
        }
        
        /* Animations */
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.7; }
        }
        
        .updating {
            animation: pulse 0.5s ease-in-out;
        }
    </style>
""", unsafe_allow_html=True)

# ===== SESSION STATE INITIALIZATION =====
if 'price_data' not in st.session_state:
    st.session_state.price_data = {}
    st.session_state.last_update = {}
    st.session_state.price_history = {symbol: [] for symbol in ["BTC", "ETH", "SOL", "BNB", "ADA"]}
    st.session_state.update_count = 0
    st.session_state.subscribed_symbols = set()

# ===== HEADER =====
col1, col2, col3 = st.columns([1, 3, 1])
with col1:
    st.title("💰 Crypto Intelligence")

with col3:
    st.metric(
        label="Status",
        value="🟢 Live",
        delta="Real-time WebSocket"
    )

# ===== STATISTICS =====
st.divider()
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Active Symbols", len(st.session_state.subscribed_symbols) or "-")
with col2:
    st.metric("Updates/Min", st.session_state.update_count or "-")
with col3:
    st.metric("Data Source", "Binance + CoinGecko")
with col4:
    st.metric("Last Update", 
              st.session_state.last_update.get('time', '--:--') if st.session_state.last_update else '--:--')

st.divider()

# ===== SYMBOL SUBSCRIPTION =====
st.subheader("📊 Real-Time Ticker")

col1, col2 = st.columns([4, 1])
with col1:
    symbol_input = st.text_input(
        "Enter symbols (comma-separated)",
        value="BTC, ETH, SOL, BNB, ADA",
        label_visibility="collapsed",
        key="symbol_input"
    )

with col2:
    if st.button("🔄 Subscribe", use_container_width=True):
        symbols = [s.strip().upper() for s in symbol_input.split(',')]
        st.session_state.subscribed_symbols = set(symbols)
        st.success(f"Subscribed to {', '.join(symbols)}")

# ===== DATA ENGINE INITIALIZATION =====
if 'data_engine' not in st.session_state:
    st.session_state.data_engine = InstitutionalDataEngine()
    # Start polling in background
    st.session_state.data_engine.start_realtime_price_polling(
        symbols=list(st.session_state.subscribed_symbols) or ["BTC", "ETH", "SOL"],
        poll_interval=2
    )

# ===== PRICE CARDS =====
if st.session_state.subscribed_symbols:
    # Get latest prices
    for symbol in sorted(st.session_state.subscribed_symbols):
        try:
            price_data = st.session_state.data_engine.get_sentiment_score(symbol)
            st.session_state.price_data[symbol] = price_data
            st.session_state.last_update['time'] = datetime.now().strftime("%H:%M:%S")
            st.session_state.update_count += 1
        except Exception as e:
            st.warning(f"Error fetching {symbol}: {e}")
    
    # Display price cards in a grid
    st.markdown("### 💹 Live Prices")
    cols = st.columns(min(3, len(st.session_state.subscribed_symbols)))
    
    for idx, symbol in enumerate(sorted(st.session_state.subscribed_symbols)):
        with cols[idx % len(cols)]:
            if symbol in st.session_state.price_data:
                data = st.session_state.price_data[symbol]
                price = data.get('price', 0)
                change_24h = data.get('change_24h', 0)
                
                # Create metric with delta
                color_indicator = "🟢" if change_24h >= 0 else "🔴"
                
                st.metric(
                    label=f"{color_indicator} {symbol}/USDT",
                    value=f"${price:,.2f}",
                    delta=f"{change_24h:+.2f}%"
                )
else:
    st.info("👆 Enter symbols above and click Subscribe to start monitoring prices")

st.divider()

# ===== CHARTS SECTION =====
st.subheader("📈 Price Analysis")

if st.session_state.price_data:
    # Prepare data for visualization
    symbols = list(st.session_state.price_data.keys())
    prices = [st.session_state.price_data[s].get('price', 0) for s in symbols]
    changes = [st.session_state.price_data[s].get('change_24h', 0) for s in symbols]
    
    col1, col2 = st.columns(2)
    
    # Price comparison chart
    with col1:
        fig_price = go.Figure()
        fig_price.add_trace(go.Bar(
            x=symbols,
            y=prices,
            marker=dict(
                color=prices,
                colorscale='RdYlGn',
                showscale=False
            ),
            name='Price (USD)'
        ))
        fig_price.update_layout(
            title="Current Prices",
            xaxis_title="Symbol",
            yaxis_title="Price (USD)",
            template="plotly_dark",
            height=400,
            hovermode='x unified'
        )
        st.plotly_chart(fig_price, use_container_width=True)
    
    # 24H Change chart
    with col2:
        fig_change = go.Figure()
        colors = ['green' if c >= 0 else 'red' for c in changes]
        fig_change.add_trace(go.Bar(
            x=symbols,
            y=changes,
            marker=dict(color=colors),
            name='24H Change (%)'
        ))
        fig_change.update_layout(
            title="24H Price Change",
            xaxis_title="Symbol",
            yaxis_title="Change (%)",
            template="plotly_dark",
            height=400,
            hovermode='x unified'
        )
        st.plotly_chart(fig_change, use_container_width=True)
    
    # Price-to-Symbol mapping
    data_df = pd.DataFrame({
        'Symbol': symbols,
        'Price (USD)': [f"${p:,.2f}" for p in prices],
        '24H Change': [f"{c:+.2f}%" for c in changes],
        'Trend': ['📈' if c >= 0 else '📉' for c in changes]
    })
    
    st.dataframe(
        data_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Symbol": st.column_config.TextColumn(width="small"),
            "Price (USD)": st.column_config.TextColumn(width="small"),
            "24H Change": st.column_config.TextColumn(width="small"),
            "Trend": st.column_config.TextColumn(width="small")
        }
    )

st.divider()

# ===== AUTO REFRESH =====
st.markdown("""
    <script>
        // Auto-refresh every 3 seconds
        setTimeout(() => {
            location.reload();
        }, 3000);
    </script>
""", unsafe_allow_html=True)

# ===== FOOTER =====
st.markdown("""
    ---
    ### 📱 Real-Time Crypto Dashboard
    - **Data Source**: Binance API + CoinGecko
    - **Update Interval**: 2-3 seconds
    - **Powered by**: WebSocket Real-Time Streaming
    
    *All prices are in USD and updated in real-time*
""")
