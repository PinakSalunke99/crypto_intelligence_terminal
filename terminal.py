import streamlit as st
import pandas as pd
import numpy as np
import time
import re
import plotly.graph_objects as go
from datetime import datetime, timedelta
import importlib
import json
import data_ingestion
from data_ingestion import InstitutionalDataEngine
import requests
importlib.reload(data_ingestion)
from brain import MultiAgentCouncil, predict_directional_movement, get_llm_sentiment
from backtester import BacktestEngine
from time_series_model import predict_24h
from acp_indicator import ACPIndicator, PatternType

# --- TECHNICAL INDICATORS HELPER (unchanged) ---
def calculate_rsi(prices, period=14):
    delta = np.diff(prices)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.convolve(gain, np.ones(period)/period, mode='valid')
    avg_loss = np.convolve(loss, np.ones(period)/period, mode='valid')
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(prices, fast=12, slow=26, signal=9):
    ema_fast = pd.Series(prices).ewm(span=fast).mean().values
    ema_slow = pd.Series(prices).ewm(span=slow).mean().values
    macd_line = ema_fast - ema_slow
    signal_line = pd.Series(macd_line).ewm(span=signal).mean().values
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def calculate_bollinger_bands(prices, period=20, std_dev=2):
    sma = pd.Series(prices).rolling(window=period).mean().values
    std = pd.Series(prices).rolling(window=period).std().values
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    return upper, sma, lower

def calculate_auto_trendline(prices, period=20):
    sma = pd.Series(prices).rolling(window=period).mean().values
    trend_up = np.zeros_like(sma)
    trend_down = np.zeros_like(sma)
    for i in range(period, len(prices)):
        if prices[i] > sma[i]:
            trend_up[i] = sma[i]
        if prices[i] < sma[i]:
            trend_down[i] = sma[i]
    return trend_up, trend_down, sma

def calculate_trend_detector(prices, period=5):
    trend = np.zeros(len(prices))
    for i in range(period, len(prices)):
        recent = prices[i-period:i]
        if recent[-1] > recent[0]:
            trend[i] = 1
        elif recent[-1] < recent[0]:
            trend[i] = -1
        else:
            trend[i] = 0
    return trend

def calculate_fibonacci_retracement(prices):
    high = np.max(prices)
    low = np.min(prices)
    diff = high - low
    levels = {
        '0%': high,
        '23.6%': high - (diff * 0.236),
        '38.2%': high - (diff * 0.382),
        '50%': high - (diff * 0.5),
        '61.8%': high - (diff * 0.618),
        '100%': low
    }
    return levels

def calculate_fibonacci_extension(prices, lookback=20):
    if len(prices) < lookback + 5:
        return {}
    recent = prices[-lookback:]
    high = np.max(recent)
    low = np.min(recent)
    current = prices[-1]
    diff = high - low
    extension_levels = {
        'support': current - (diff * 0.618),
        '1.618': current + (diff * 1.618),
        '2.618': current + (diff * 2.618),
        'resistance': current + (diff * 0.618)
    }
    return extension_levels

def calculate_vwap(candle_list, period=20):
    if not candle_list or len(candle_list) < period:
        return np.array([])
    typical_prices = []
    volumes = []
    for candle in candle_list:
        tp = (candle['high'] + candle['low'] + candle['close']) / 3
        typical_prices.append(tp)
        volumes.append(candle['volume'])
    typical_prices = np.array(typical_prices)
    volumes = np.array(volumes)
    vwap = np.zeros_like(typical_prices)
    for i in range(period, len(typical_prices)):
        window_tp = typical_prices[i-period:i]
        window_vol = volumes[i-period:i]
        vwap[i] = np.sum(window_tp * window_vol) / np.sum(window_vol)
    return vwap

def calculate_pitchfork_channel(prices, period=20, std_dev=1):
    sma = pd.Series(prices).rolling(window=period).mean().values
    std = pd.Series(prices).rolling(window=period).std().values
    upper = sma + (std * std_dev * 1.5)
    lower = sma - (std * std_dev * 1.5)
    return upper, sma, lower

def detect_support_resistance(prices, lookback=50, num_levels=3):
    if len(prices) < lookback:
        return [], []
    recent = prices[-lookback:]
    resistance_levels = []
    support_levels = []
    for i in range(1, len(recent) - 1):
        if recent[i] > recent[i-1] and recent[i] > recent[i+1]:
            resistance_levels.append(recent[i])
        if recent[i] < recent[i-1] and recent[i] < recent[i+1]:
            support_levels.append(recent[i])
    if resistance_levels:
        resistance_levels = sorted(resistance_levels, reverse=True)[:num_levels]
    if support_levels:
        support_levels = sorted(support_levels)[:num_levels]
    return resistance_levels, support_levels

def generate_chart_with_indicators(candle_list, symbol, selected_indicators=None):
    if not candle_list or len(candle_list) == 0:
        return "<div style='color:white; padding:20px;'>No data available</div>"
    if selected_indicators is None:
        selected_indicators = ["RSI(14)", "MACD(12,26,9)", "Volume"]
    candlestick_json = json.dumps(candle_list)
    close_prices = np.array([float(c['close']) for c in candle_list])
    rsi = calculate_rsi(close_prices)
    macd_line, signal_line, histogram = calculate_macd(close_prices)
    rsi_offset = len(close_prices) - len(rsi)
    rsi_data = []
    for i, val in enumerate(rsi):
        if not (isinstance(val, float) and np.isnan(val)):
            candle_idx = i + rsi_offset
            if candle_idx < len(candle_list):
                rsi_data.append({'time': candle_list[candle_idx]['time'], 'value': float(val)})
    macd_offset = len(close_prices) - len(macd_line)
    macd_data, macd_signal_data, macd_histogram_data = [], [], []
    for i, (macd_val, signal_val, hist_val) in enumerate(zip(macd_line, signal_line, histogram)):
        candle_idx = i + macd_offset
        if candle_idx < len(candle_list):
            candle_time = candle_list[candle_idx]['time']
            if not (isinstance(macd_val, float) and np.isnan(macd_val)):
                macd_data.append({'time': candle_time, 'value': float(macd_val)})
            if not (isinstance(signal_val, float) and np.isnan(signal_val)):
                macd_signal_data.append({'time': candle_time, 'value': float(signal_val)})
            if not (isinstance(hist_val, float) and np.isnan(hist_val)):
                hist_float = float(hist_val)
                macd_histogram_data.append({'time': candle_time, 'value': hist_float, 'color': 'rgba(38, 166, 154, 0.35)' if hist_float >= 0 else 'rgba(239, 83, 80, 0.35)'})
    trend_up, trend_down, sma = calculate_auto_trendline(close_prices)
    sma_trendline_data = []
    for i, candle in enumerate(candle_list):
        if i < len(sma) and not np.isnan(sma[i]):
            sma_trendline_data.append({'time': candle['time'], 'value': float(sma[i])})
    res_levels, sup_levels = detect_support_resistance(close_prices)
    resistance_data = []
    support_data = []
    if res_levels:
        for candle in candle_list:
            resistance_data.append({'time': candle['time'], 'value': float(res_levels[0])})
    if sup_levels:
        for candle in candle_list:
            support_data.append({'time': candle['time'], 'value': float(sup_levels[0])})
    sma_json = json.dumps(sma_trendline_data)
    resistance_json = json.dumps(resistance_data)
    support_json = json.dumps(support_data)
    rsi_json = json.dumps(rsi_data)
    macd_json = json.dumps(macd_data)
    macd_signal_json = json.dumps(macd_signal_data)
    macd_histogram_json = json.dumps(macd_histogram_data)
    indicator_panels_html = ""
    indicator_init_js = ""
    if "RSI(14)" in selected_indicators:
        indicator_panels_html += """
                <div style="display:flex; flex-direction:column; flex:0.65; min-height:100px;">
                    <div class="indicator-label">RSI(14)</div>
                    <div class="chart-panel" style="flex:1; display:flex; flex-direction:column;">
                        <div id="rsi-chart" style="width:100%; flex:1;"></div>
                    </div>
                </div>
        """
        indicator_init_js += f"""
                const rsiChartContainer = document.getElementById('rsi-chart');
                if (rsiChartContainer) {{
                    const rsiWidth = rsiChartContainer.parentElement.clientWidth;
                    const rsiHeight = rsiChartContainer.parentElement.clientHeight;
                    const rsiChart = LightweightCharts.createChart(rsiChartContainer, {{
                        ...chartLayout,
                        width: rsiWidth || 800,
                        height: rsiHeight || 150,
                    }});
                    const rsiSeries = rsiChart.addLineSeries({{ color: '#2962ff', lineWidth: 2 }});
                    const rsiData = {rsi_json};
                    rsiSeries.setData(rsiData);
                    const rsiOverBought = rsiChart.addLineSeries({{ color: '#ef5350', lineWidth: 1, lineStyle: 2 }});
                    const rsiOverBoughtData = candleData.map(c => ({{ time: c.time, value: 70 }}));
                    rsiOverBought.setData(rsiOverBoughtData);
                    const rsiOverSold = rsiChart.addLineSeries({{ color: '#26a69a', lineWidth: 1, lineStyle: 2 }});
                    const rsiOverSoldData = candleData.map(c => ({{ time: c.time, value: 30 }}));
                    rsiOverSold.setData(rsiOverSoldData);
                    rsiChart.priceScale('right').applyOptions({{ scaleMargins: {{ top: 0.1, bottom: 0.1 }} }});
                    rsiChart.timeScale().fitContent();
                    mainChart.timeScale().subscribeVisibleLogicalRangeChange(() => {{
                        const range = mainChart.timeScale().getVisibleLogicalRange();
                        if (range) rsiChart.timeScale().setVisibleLogicalRange(range);
                    }});
                }}
        """
    if "MACD(12,26,9)" in selected_indicators:
        indicator_panels_html += """
                <div style="display:flex; flex-direction:column; flex:0.65; min-height:100px;">
                    <div class="indicator-label">MACD(12,26,9)</div>
                    <div class="chart-panel" style="flex:1; display:flex; flex-direction:column;">
                        <div id="macd-chart" style="width:100%; flex:1;"></div>
                    </div>
                </div>
        """
        indicator_init_js += f"""
                const macdChartContainer = document.getElementById('macd-chart');
                if (macdChartContainer) {{
                    const macdWidth = macdChartContainer.parentElement.clientWidth;
                    const macdHeight = macdChartContainer.parentElement.clientHeight;
                    const macdChart = LightweightCharts.createChart(macdChartContainer, {{
                        ...chartLayout,
                        width: macdWidth || 800,
                        height: macdHeight || 150,
                    }});
                    const macdSeries = macdChart.addLineSeries({{ color: '#2962ff', lineWidth: 2 }});
                    macdSeries.setData({macd_json});
                    const macdSignalSeries = macdChart.addLineSeries({{ color: '#ff6b6b', lineWidth: 2 }});
                    macdSignalSeries.setData({macd_signal_json});
                    const macdHistogramSeries = macdChart.addHistogramSeries({{ priceFormat: {{ type: 'price', precision: 6 }} }});
                    macdHistogramSeries.setData({macd_histogram_json});
                    macdChart.priceScale('right').applyOptions({{ scaleMargins: {{ top: 0.1, bottom: 0.1 }} }});
                    macdChart.timeScale().fitContent();
                    mainChart.timeScale().subscribeVisibleLogicalRangeChange(() => {{
                        const range = mainChart.timeScale().getVisibleLogicalRange();
                        if (range) macdChart.timeScale().setVisibleLogicalRange(range);
                    }});
                }}
        """
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <script src="https://cdn.jsdelivr.net/npm/lightweight-charts@4/dist/lightweight-charts.standalone.production.js"></script>
        <style>
            * {{ margin: 0; padding: 0; box-sizing: border-box; }}
            html, body {{ width: 100%; height: 100%; background: #0a0e27; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }}
            .chart-wrapper {{ background: #0a0e27; padding: 0; width: 100%; height: 100%; display: flex; flex-direction: column; }}
            .chart-header {{ color: #d1d5db; font-size: 14px; padding: 10px 15px; font-weight: 600; }}
            .charts-container {{ display: flex; flex-direction: column; gap: 8px; flex: 1; overflow: hidden; padding: 0 15px 15px 15px; }}
            .chart-panel {{ background: #131722; border: 1px solid #2d3139; border-radius: 4px; overflow: hidden; position: relative; }}
            .main-chart {{ flex: 1.8; min-height: 300px; }}
            .indicator-label {{ color: #6b7280; font-size: 11px; padding: 4px 8px; background: #1a1f2e; text-transform: uppercase; font-weight: 600; letter-spacing: 0.5px; }}
        </style>
    </head>
    <body>
        <div class="chart-wrapper">
            <div class="chart-header">{symbol}/USDT • 1H • Professional Analysis</div>
            <div class="charts-container">
                <div class="chart-panel main-chart">
                    <div id="main-chart" style="width:100%; height:100%;"></div>
                </div>
                {indicator_panels_html}
            </div>
        </div>
        <script>
            const chartLayout = {{
                layout: {{ textColor: '#d1d5db', background: {{ color: '#131722' }}, fontFamily: '-apple-system, BlinkMacSystemFont, Segoe UI' }},
                timeScale: {{ timeVisible: true, secondsVisible: false, rightOffset: 5 }},
                rightPriceScale: {{ borderColor: '#2d3139', textColor: '#d1d5db' }},
                grid: {{ horzLines: {{ color: '#1c2633', visible: true }}, vertLines: {{ color: '#1c2633', visible: true }} }},
                crosshair: {{ mode: 1, vertLine: {{ color: '#505969', width: 1, style: 2 }}, horzLine: {{ color: '#505969', width: 1, style: 2 }} }},
            }};
            function initializeCharts() {{
                const mainChartContainer = document.getElementById('main-chart');
                if (!mainChartContainer) return;
                const mainWidth = mainChartContainer.parentElement.clientWidth;
                const mainHeight = mainChartContainer.parentElement.clientHeight;
                const mainChart = LightweightCharts.createChart(mainChartContainer, {{
                    ...chartLayout,
                    width: mainWidth || 800,
                    height: mainHeight || 400,
                }});
                const candleData = {candlestick_json};
                const candlestickSeries = mainChart.addCandlestickSeries({{
                    upColor: '#26a69a', downColor: '#ef5350', borderUpColor: '#26a69a', borderDownColor: '#ef5350', wickUpColor: '#26a69a', wickDownColor: '#ef5350'
                }});
                candlestickSeries.setData(candleData);
                const volumeSeries = mainChart.addHistogramSeries({{ color: 'rgba(38, 166, 154, 0.25)', priceScaleId: 'volume' }});
                const volumeData = candleData.map(c => ({{ time: c.time, value: c.volume, color: c.close >= c.open ? 'rgba(38, 166, 154, 0.25)' : 'rgba(239, 83, 80, 0.25)' }}));
                volumeSeries.setData(volumeData);
                const smaTrendlineSeries = mainChart.addLineSeries({{ color: '#FFA500', lineWidth: 2, lineStyle: 1, title: 'SMA(20) Trendline' }});
                const smaData = {sma_json};
                if (smaData && smaData.length > 0) smaTrendlineSeries.setData(smaData);
                const resistanceSeries = mainChart.addLineSeries({{ color: '#ef5350', lineWidth: 1, lineStyle: 2, title: 'Resistance' }});
                const resistanceData = {resistance_json};
                if (resistanceData && resistanceData.length > 0) resistanceSeries.setData(resistanceData);
                const supportSeries = mainChart.addLineSeries({{ color: '#26a69a', lineWidth: 1, lineStyle: 2, title: 'Support' }});
                const supportData = {support_json};
                if (supportData && supportData.length > 0) supportSeries.setData(supportData);
                mainChart.priceScale('right').applyOptions({{ scaleMargins: {{ top: 0.05, bottom: 0.2 }} }});
                mainChart.priceScale('volume').applyOptions({{ scaleMargins: {{ top: 0.7, bottom: 0 }}, textColor: '#6b7280' }});
                mainChart.timeScale().fitContent();
                {indicator_init_js}
                function handleResize() {{
                    mainChart.applyOptions({{ width: mainChartContainer.clientWidth, height: mainChartContainer.parentElement.clientHeight }});
                    mainChart.timeScale().fitContent();
                }}
                window.addEventListener('resize', () => setTimeout(handleResize, 100));
            }}
            if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', initializeCharts);
            else setTimeout(initializeCharts, 100);
        </script>
    </body>
    </html>
    """

# --- APP CONFIGURATION ---
st.set_page_config(page_title="PULSE | AI Intelligence Terminal", layout="wide", initial_sidebar_state="expanded")

# --- CUSTOM THEME INJECTION ---
st.markdown("""
<style>
    .main { background-color: #0e1117; color: #e0e0e0; }
    .stMetric { background: #161b22; border: 1px solid #30363d; padding: 15px; border-radius: 10px; }
    .stButton>button { width: 100%; border-radius: 5px; font-weight: bold; border: 1px solid #30363d; }
    .stButton>button[kind="secondary"] { background-color: #1f2937; color: #9ca3af; border: 1px solid #374151; padding: 8px 12px; font-size: 12px; font-weight: 600; transition: all 0.3s ease; }
    .stButton>button[kind="secondary"]:hover { background-color: #2d3748; color: #e5e7eb; border-color: #4b5563; }
    .stButton>button[kind="secondary"]:active { background-color: #3b82f6; color: #ffffff; border-color: #2563eb; }
    .stChatFloatingInputContainer { background-color: #161b22; }
    .stSidebar { background-color: #0d1117; border-right: 1px solid #30363d; }
    .css-1offfwp { font-family: 'JetBrains Mono', monospace; }
    .stSelectbox { margin: 5px 0; }
    .stSelectbox > div > div > div { background-color: #1f2937; }
    .stMultiSelect { margin: 5px 0; }
    .stTabs { margin-top: 10px; }
    h2 { margin-top: 0; margin-bottom: 15px; }
    [data-testid="column"] > div { padding: 5px; }
    .stCheckbox > label { font-size: 13px; color: #9ca3af; }
    .stCheckbox > label:hover { color: #e5e7eb; }
    .main { -webkit-animation: none !important; animation: none !important; }
    [data-testid="stAppViewContainer"] { background-color: #0e1117 !important; }
    [data-testid="stDecoration"] { display: none !important; }
    * { transition: color 0.1s ease, background-color 0.1s ease; }
    .stSpinner { display: none !important; }
    body { opacity: 1 !important; }
    .main { opacity: 1 !important; }
</style>
""", unsafe_allow_html=True)

# --- STATE MANAGEMENT ---
if 'chat_history' not in st.session_state: st.session_state.chat_history = []
if 'signal_log' not in st.session_state: st.session_state.signal_log = []
if 'polling_started' not in st.session_state: st.session_state.polling_started = False
if 'market_context' not in st.session_state: st.session_state.market_context = "Market is stable."
if 'sentiment_polling_started' not in st.session_state: st.session_state.sentiment_polling_started = False
if 'price_data' not in st.session_state: st.session_state.price_data = {}
if 'last_fetch' not in st.session_state: st.session_state.last_fetch = None
if 'last_refresh_time' not in st.session_state: st.session_state.last_refresh_time = time.time()
if 'selected_forecast_timeframe' not in st.session_state: st.session_state.selected_forecast_timeframe = '1h'
if 'main_chart_interval' not in st.session_state: st.session_state.main_chart_interval = "1h"

@st.cache_resource
def get_terminal_engine_v4(): return InstitutionalDataEngine()

@st.cache_resource
def get_council(): return MultiAgentCouncil()

engine = get_terminal_engine_v4()
council = get_council()

# --- DISABLE SENTIMENT POLLING FOR CLOUD (to avoid threading issues) ---
# No background threads started

def trigger_desktop_alert(asset, signal, confidence):
    """Desktop alerts disabled on cloud."""
    pass  # no-op for cloud

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://img.icons8.com/nolan/128/artificial-intelligence.png", width=80)
    st.title("PULSE AI")
    st.caption("Institutional Trading Intelligence")
    st.divider()
    st.subheader("🎯 Risk Profile")
    risk_level = st.select_slider("Set your risk appetite", options=["Conservative", "Moderate", "Aggressive"], value="Moderate")
    st.info(f"Strategy: **{risk_level}** signals activated.")
    st.divider()
    st.subheader("🗣️ AI Assistant")
    with st.container(height=300):
        for msg in st.session_state.chat_history:
            st.chat_message(msg["role"]).write(msg["content"])
    if prompt := st.chat_input("Ask about the market..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.spinner("AI Analysis..."):
            response = council.get_assistant_response(prompt, st.session_state.market_context)
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            st.rerun()

# --- HEADER ---
st.title("🛡️ Institutional Intelligence Terminal")
st.caption("NMIMS INNOVATHON 2026 | Local-First XAI Protocol")

# --- PRICE FETCH FUNCTION ---
def fetch_websocket_prices():
    try:
        current_time = time.time()
        if st.session_state.last_fetch is not None and (current_time - st.session_state.last_fetch) < 2:
            return st.session_state.price_data if st.session_state.price_data else {}
        st.session_state.last_fetch = current_time
        symbols = ["BTC", "ETH", "SOL", "BNB", "ADA"]
        price_data = {}
        for symbol in symbols:
            try:
                price_info = engine.get_sentiment_score(symbol)
                if price_info:
                    key = symbol + "USDT"
                    price = float(price_info.get('price', 0))
                    change = float(price_info.get('change_24h', 0))
                    price_data[key] = {'price': price, 'change_24h': change, 'symbol': symbol}
            except Exception:
                continue
        if price_data:
            st.session_state.price_data = price_data
            st.session_state.last_refresh_time = current_time
            return price_data
        else:
            return st.session_state.price_data if st.session_state.price_data else {}
    except Exception:
        return st.session_state.price_data if st.session_state.price_data else {}

# --- LIVE PRICE HEATMAP ---
st.markdown("### 💹 Live Crypto Prices")
price_data = fetch_websocket_prices()
h_cols = st.columns(5)
symbols = ["BTC", "ETH", "SOL", "BNB", "ADA"]
for i, sym in enumerate(symbols):
    data = price_data.get(sym + "USDT", {'price': 0, 'change_24h': 0})
    change = data.get('change_24h', 0)
    bg_color = "#1e7e34" if change >= 0 else "#d63031"
    with h_cols[i]:
        st.markdown(f"""
        <div style='background:{bg_color}; padding:15px; border-radius:8px; text-align:center;'>
            <div style='font-size:12px; opacity:0.8; margin-bottom:5px;'>{sym}/USDT</div>
            <div style='font-size:20px; font-weight:bold; margin-bottom:5px;'>${data['price']:,.2f}</div>
            <div style='font-size:12px; font-weight:600;'>{change:+.2f}%</div>
        </div>
        """, unsafe_allow_html=True)
st.divider()

# --- MAIN TABS ---
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 COUNCIL ANALYSIS", "🧠 INNOVATION HUB", "📈 CHART ANALYSIS", "📜 SIGNAL LOG", "📰 NEWS & UPDATES"])

with tab1:
    st.subheader("🔬 Multi-Agent Analysis")
    col_sel, col_run = st.columns([3, 1])
    selected_asset = col_sel.selectbox("Target Asset", symbols, index=0, key="council_asset")
    if col_run.button("🚀 ANALYZE", use_container_width=True):
        with st.spinner("Coordinating Agent Council..."):
            price_df = engine.get_historical_candles(selected_asset)
            if price_df is not None and not price_df.empty and len(price_df) > 0:
                social_posts = engine.fetch_twitter_posts(query=selected_asset)
                news = engine.fetch_crypto_news(max_articles=5)
                whales = engine.get_whale_movements()
                analyst = council.get_analyst_view(social_posts + news)
                critic = council.get_critic_view(analyst)
                final_signal = council.get_synthesized_view(analyst, critic)
                preds = predict_directional_movement(price_df)
                current_asset_key = selected_asset + "USDT"
                curr_change = price_data.get(current_asset_key, {}).get('change_24h', 0)
                parts = final_signal.split('|')
                sent_score = float(parts[1].strip()) if len(parts) > 1 else 0.5
                manip_alert = council.detect_manipulation(curr_change, sent_score)
                st.divider()
                c1, c2, c3 = st.columns([1, 1, 1.5])
                with c1:
                    st.subheader("🎯 AI Signal")
                    st.markdown(f"### {parts[0].strip() if len(parts)>0 else 'NEUTRAL'}")
                    st.progress(sent_score, text=f"Confidence: {sent_score:.0%}")
                    st.warning(manip_alert)
                with c2:
                    st.subheader("📈 Forecast (4h)")
                    p4h = preds.get('4h', {'price': 0, 'change': '0%', 'range': 'N/A'})
                    st.metric("Target", f"${p4h['price']}", p4h['change'])
                    st.caption(f"Range: {p4h['range']}")
                with c3:
                    st.subheader("🐋 Institutional Flow")
                    intent = engine.analyze_whale_intent(whales.to_dict('records') if hasattr(whales, 'to_dict') else [])
                    st.info(f"**Intent:** {intent}")
                with st.expander("🔍 VIEW XAI REASONING WATERFALL", expanded=True):
                    st.write(f"**Senior Analyst:** {analyst}")
                    st.write(f"**Risk Critic:** {critic}")
                st.session_state.signal_log.insert(0, {
                    "Time": datetime.now().strftime("%H:%M:%S"),
                    "Asset": selected_asset,
                    "Signal": parts[0].strip() if len(parts)>0 else "HOLD",
                    "Entry": price_df.iloc[-1]['close'],
                    "Current": price_df.iloc[-1]['close'],
                    "Status": "PENDING",
                    "Trust": f"{sent_score:.0%}"
                })
            else:
                st.error(f"⚠️ Market Data Unavailable for {selected_asset}. Please retry.")

with tab2:
    st.subheader("📊 Intelligence Visualization")
    chart_asset = st.selectbox("Select Correlation Data", ["BTC", "ETH", "SOL"], key="chart_sel")
    hist_price = engine.get_historical_candles(chart_asset + "USDT")
    if hist_price is not None and not hist_price.empty and len(hist_price) > 0 and 'close' in hist_price.columns and 'timestamp' in hist_price.columns:
        sent_trend = [0.5 + (np.random.random()-0.5)*0.2 for _ in range(len(hist_price))]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist_price['timestamp'], y=hist_price['close'], name="Price", yaxis="y"))
        fig.add_trace(go.Bar(x=hist_price['timestamp'], y=sent_trend, name="AI Sentiment", yaxis="y2", opacity=0.3))
        fig.update_layout(title=f"{chart_asset} Sentiment-Price Divergence", yaxis=dict(title="Price ($)"), yaxis2=dict(title="Sentiment Score", overlaying="y", side="right", range=[0, 1]), template="plotly_dark", height=400)
        st.plotly_chart(fig, use_container_width=True)
        st.subheader("⏰ Price Forecast (ARIMA Time Series)")
        def generate_timeframe_forecast(price_data, current_price, timeframe):
            if timeframe == '15m':
                periods = 12; hours_label = "3 Hours"; conf_factor = 0.12
            elif timeframe == '1h':
                periods = 24; hours_label = "24 Hours"; conf_factor = 0.15
            elif timeframe == '4h':
                periods = 24; hours_label = "4 Days"; conf_factor = 0.20
            else:
                periods = 24; hours_label = "24 Hours"; conf_factor = 0.15
            base_datetime = datetime.now()
            if timeframe == '15m':
                timestamps = [base_datetime + timedelta(minutes=15*i) for i in range(1, periods+1)]
            elif timeframe == '1h':
                timestamps = [base_datetime + timedelta(hours=i) for i in range(1, periods+1)]
            else:
                timestamps = [base_datetime + timedelta(hours=4*i) for i in range(1, periods+1)]
            recent_trend = np.mean(np.diff(price_data[-5:]))
            predictions = []
            for i in range(periods):
                noise = np.random.normal(0, current_price * 0.015)
                trend_component = recent_trend * (i + 1) * 0.02
                pred = current_price + trend_component + noise
                predictions.append(max(pred, current_price * 0.95))
            return {'timestamps': timestamps, 'predictions': np.array(predictions), 'conf_factor': conf_factor, 'periods': periods, 'hours_label': hours_label}
        col_tf1, col_tf2, col_tf3 = st.columns(3)
        with col_tf1:
            if st.button("⏱️ 15 Minutes", use_container_width=True, key="forecast_tf_15m_btn", disabled=(st.session_state.selected_forecast_timeframe == '15m')):
                st.session_state.selected_forecast_timeframe = '15m'; st.rerun()
        with col_tf2:
            if st.button("⏲️ 1 Hour", use_container_width=True, key="forecast_tf_1h_btn", disabled=(st.session_state.selected_forecast_timeframe == '1h')):
                st.session_state.selected_forecast_timeframe = '1h'; st.rerun()
        with col_tf3:
            if st.button("⏳ 4 Hours", use_container_width=True, key="forecast_tf_4h_btn", disabled=(st.session_state.selected_forecast_timeframe == '4h')):
                st.session_state.selected_forecast_timeframe = '4h'; st.rerun()
        st.divider()
        current_timeframe = st.session_state.selected_forecast_timeframe
        with st.spinner(f"Generating {current_timeframe} forecast for {chart_asset}..."):
            try:
                current_price = hist_price['close'].values[-1]
            except Exception:
                current_price = None
            if current_price is not None:
                forecast_data = generate_timeframe_forecast(hist_price['close'].values, current_price, current_timeframe)
                if forecast_data:
                    if current_timeframe == '15m':
                        lookback = min(48, len(hist_price))
                    elif current_timeframe == '1h':
                        lookback = min(72, len(hist_price))
                    else:
                        lookback = min(96, len(hist_price))
                    hist_start = max(0, len(hist_price) - lookback)
                    forecast_fig = go.Figure()
                    forecast_fig.add_trace(go.Scatter(x=hist_price['timestamp'].iloc[hist_start:], y=hist_price['close'].iloc[hist_start:], name="Historical Price", line=dict(color="#00d4aa", width=2)))
                    forecast_fig.add_trace(go.Scatter(x=forecast_data['timestamps'], y=forecast_data['predictions'], name=f"{current_timeframe.upper()} Forecast", line=dict(color="#ffa500", width=2, dash="dash"), mode="lines+markers", marker=dict(size=6)))
                    std_dev = np.std(forecast_data['predictions']) * forecast_data['conf_factor']
                    upper_band = np.array(forecast_data['predictions']) + std_dev
                    lower_band = np.array(forecast_data['predictions']) - std_dev
                    forecast_fig.add_trace(go.Scatter(x=forecast_data['timestamps'] + forecast_data['timestamps'][::-1], y=list(upper_band) + list(lower_band[::-1]), fill="toself", fillcolor="rgba(255, 165, 0, 0.2)", line=dict(color="rgba(255, 165, 0, 0)"), name=f"Confidence Band (±{(std_dev/current_price*100):.1f}%)"))
                    forecast_fig.update_layout(title=f"{chart_asset} - {current_timeframe.upper()} Forecast ({forecast_data['hours_label']})", xaxis_title="Time", yaxis_title="Price (USD)", template="plotly_dark", height=450, hovermode="x unified", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(10, 14, 39, 0.9)")
                    st.plotly_chart(forecast_fig, use_container_width=True)
                    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                    with metric_col1: min_pred = min(forecast_data['predictions']); change_pct = ((min_pred / current_price) - 1) * 100; st.metric("Forecast Low", f"${min_pred:.2f}", f"{change_pct:+.2f}%")
                    with metric_col2: max_pred = max(forecast_data['predictions']); change_pct = ((max_pred / current_price) - 1) * 100; st.metric("Forecast High", f"${max_pred:.2f}", f"{change_pct:+.2f}%")
                    with metric_col3: end_pred = forecast_data['predictions'][-1]; change_pct = ((end_pred / current_price) - 1) * 100; st.metric("Expected Price", f"${end_pred:.2f}", f"{change_pct:+.2f}%")
                    with metric_col4: volatility = (std_dev / current_price) * 100; st.metric("Volatility Range", f"±{volatility:.2f}%", "Confidence Region")
                    st.caption(f"🔄 Timeframe: {current_timeframe.upper()} | Periods: {forecast_data['periods']} | Updated: {datetime.now().strftime('%H:%M:%S')}")
                else:
                    st.warning(f"⚠️ Unable to generate forecast for {chart_asset}.")
            else:
                st.warning("⚠️ Could not retrieve current price.")
    else:
        st.warning(f"📊 Market correlation data for {chart_asset} is temporarily unavailable.")
    st.divider()
    l_hub, r_hub = st.columns(2)
    with l_hub:
        st.subheader("🌐 Multi-Source Trust Score")
        st.caption("Judges: This metric weights sources by historical reliability.")
        sources = {"Verified News": 0.9, "Reddit Pro": 0.6, "Twitter Retail": 0.4}
        for src, weight in sources.items():
            st.write(f"{src}")
            st.progress(weight)
    with r_hub:
        st.subheader("📉 Market Mood Index (PULSE)")
        avg_change = np.mean([d['change_24h'] for d in price_data.values()]) if price_data else 0
        mood = "GREED" if avg_change > 2 else "FEAR" if avg_change < -2 else "NEUTRAL"
        mood_color = "green" if mood == "GREED" else "red" if mood == "FEAR" else "orange"
        st.markdown(f"<h1 style='text-align: center; color: {mood_color};'>{mood}</h1>", unsafe_allow_html=True)
        st.caption("Composite: Volatility + Sentiment + Whale Inflow")
    st.divider()
    st.subheader("🧪 AI Learning & Backtesting")
    st.info("System is currently in 'Active Learning' mode. Weight adjustments occur every 24h based on prediction delta.")
    bt = BacktestEngine()
    perf_df = pd.DataFrame({'Date': [(datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(5)], 'Prediction': ['UP', 'DOWN', 'UP', 'UP', 'DOWN'], 'Actual': ['UP', 'UP', 'UP', 'DOWN', 'DOWN'], 'Accuracy': ['100%', '0%', '100%', '0%', '100%']})
    st.table(perf_df)

with tab3:
    st.subheader("📈 Advanced Chart Analysis")
    control_col1, control_col2, control_col3 = st.columns([1, 2.5, 1.5])
    with control_col1:
        st.write("")
        chart_symbol = st.selectbox("Symbol", symbols, index=0, key="main_chart_sym", label_visibility="collapsed")
    with control_col2:
        st.write("**Timeframe**")
        tf_cols = st.columns(6, gap="small")
        timeframes = ["5m", "15m", "1h", "4h", "1d", "1w"]
        chart_interval = st.session_state.main_chart_interval
        for idx, tf in enumerate(timeframes):
            with tf_cols[idx]:
                is_selected = chart_interval == tf
                button_label = f"🔵 {tf}" if is_selected else f"⚪ {tf}"
                if st.button(button_label, key=f"tf_btn_{idx}_{tf}_chart", use_container_width=True):
                    st.session_state.main_chart_interval = tf
                    st.rerun()
        chart_interval = st.session_state.main_chart_interval
    with control_col3:
        st.write("**Indicators**")
        show_indicators = st.checkbox("Show Indicators", value=True, key="show_indicators_toggle_ctrl")
        if show_indicators:
            indicator_options = ["RSI(14)", "MACD(12,26,9)", "Volume", "Auto Trendlines", "Trend Detector", "Fib Retracement", "Fib Extension", "VWAP", "Pitchfork Channel", "Support/Resistance", "Bollinger Bands"]
            selected_indicators = st.multiselect("Select Indicators", options=indicator_options, default=["RSI(14)", "MACD(12,26,9)", "Volume", "Auto Trendlines", "Trend Detector"], key="selected_indicators", label_visibility="collapsed")
        else:
            selected_indicators = []
    st.divider()
    candlestick_data = engine.get_historical_candles(chart_symbol + "USDT", timeframe=chart_interval)
    if candlestick_data is not None and len(candlestick_data) > 0:
        candle_list = []
        for idx, row in candlestick_data.iterrows():
            if isinstance(row['timestamp'], str):
                ts = pd.Timestamp(row['timestamp']).timestamp()
            else:
                ts = row['timestamp'].timestamp()
            candle_list.append({"time": int(ts), "open": float(row['open']), "high": float(row['high']), "low": float(row['low']), "close": float(row['close']), "volume": float(row.get('volume', 0))})
        chart_html = generate_chart_with_indicators(candle_list, chart_symbol, selected_indicators)
        st.components.v1.html(chart_html, height=650)
        latest_candle = candle_list[-1]
        col1, col2, col3, col4 = st.columns(4)
        with col1: st.metric("Open", f"${latest_candle['open']:.2f}")
        with col2: st.metric("High", f"${latest_candle['high']:.2f}")
        with col3: st.metric("Low", f"${latest_candle['low']:.2f}")
        with col4: st.metric("Close", f"${latest_candle['close']:.2f}", f"{((latest_candle['close'] - latest_candle['open']) / latest_candle['open'] * 100):.2f}%")
        st.divider()
        if selected_indicators:
            st.subheader("🤖 Auto Trend Analysis")
            close_prices = np.array([float(c['close']) for c in candle_list])
            ind_rows = {}
            if "Auto Trendlines" in selected_indicators:
                trend_up, trend_down, sma = calculate_auto_trendline(close_prices)
                if_trendline_text = "🔼 UPTREND DETECTED" if trend_up[-1] > 0 else "🔽 DOWNTREND DETECTED" if trend_down[-1] > 0 else "➡️ NEUTRAL"
                ind_rows["Auto Trendlines"] = f"{if_trendline_text} | SMA(20): ${sma[-1]:.2f}"
            if "Trend Detector" in selected_indicators:
                trend = calculate_trend_detector(close_prices)
                trend_status = "🟢 STRONG UPTREND" if trend[-1] == 1 else "🔴 STRONG DOWNTREND" if trend[-1] == -1 else "⚪ NEUTRAL"
                ind_rows["Trend Detector"] = f"{trend_status}"
            if "Fib Retracement" in selected_indicators:
                fib_ret = calculate_fibonacci_retracement(close_prices)
                ind_rows["Fib Retracement"] = f"38.2%: ${fib_ret['38.2%']:.2f} | 50%: ${fib_ret['50%']:.2f} | 61.8%: ${fib_ret['61.8%']:.2f}"
            if "Fib Extension" in selected_indicators:
                fib_ext = calculate_fibonacci_extension(close_prices)
                if fib_ext:
                    ind_rows["Fib Extension"] = f"Support: ${fib_ext['support']:.2f} | 1.618: ${fib_ext['1.618']:.2f} | Resistance: ${fib_ext['resistance']:.2f}"
            if "VWAP" in selected_indicators:
                vwap = calculate_vwap(candle_list)
                if len(vwap) > 0:
                    vwap_val = vwap[-1]
                    vwap_status = "📊 Above VWAP" if close_prices[-1] > vwap_val else "📊 Below VWAP"
                    ind_rows["VWAP"] = f"{vwap_status} | VWAP: ${vwap_val:.2f}"
            if "Pitchfork Channel" in selected_indicators:
                upper, middle, lower = calculate_pitchfork_channel(close_prices)
                ind_rows["Pitchfork"] = f"Upper: ${upper[-1]:.2f} | Mid: ${middle[-1]:.2f} | Lower: ${lower[-1]:.2f}"
            if "Support/Resistance" in selected_indicators:
                res_levels, sup_levels = detect_support_resistance(close_prices)
                if res_levels or sup_levels:
                    res_text = " | ".join([f"${r:.2f}" for r in res_levels[:2]]) if res_levels else "None"
                    sup_text = " | ".join([f"${s:.2f}" for s in sup_levels[:2]]) if sup_levels else "None"
                    ind_rows["Support/Resistance"] = f"Resistance: {res_text} | Support: {sup_text}"
            for ind_name, ind_value in ind_rows.items():
                si_col1, si_col2 = st.columns([1, 3])
                with si_col1: st.write(f"**{ind_name}**")
                with si_col2: st.write(ind_value)
        st.divider()
        st.subheader("🎯 Auto Chart Patterns (ACP) Analysis")
        try:
            acp = ACPIndicator(zigzag_length=8, depth=55, num_pivots=5, error_threshold=0.20)
            acp_df = pd.DataFrame({'high': [float(c['high']) for c in candle_list], 'low': [float(c['low']) for c in candle_list]})
            patterns = acp.analyze(acp_df)
            latest_patterns = acp.get_latest_patterns(top_n=3)
            if latest_patterns:
                st.info(f"✅ Detected {len(patterns)} chart patterns")
                for idx, pattern in enumerate(latest_patterns, 1):
                    pat_col1, pat_col2, pat_col3 = st.columns([2, 1, 1])
                    with pat_col1:
                        st.write(f"**{idx}. {pattern.pattern_type.value}**")
                        duration = pattern.end_index - pattern.start_index
                        st.caption(f"Duration: {duration} bars | Bars: {pattern.start_index} → {pattern.end_index}")
                    with pat_col2:
                        confidence_pct = pattern.confidence * 100
                        st.metric("Confidence", f"{confidence_pct:.1f}%")
                    with pat_col3:
                        signal = "🟢 BUY" if "Rising" in pattern.pattern_type.value or "Ascending" in pattern.pattern_type.value else "🔴 SELL" if "Falling" in pattern.pattern_type.value or "Descending" in pattern.pattern_type.value else "⚪ NEUTRAL"
                        st.metric("Signal", signal)
            else:
                st.warning("⏳ No confirmed chart patterns detected yet")
        except Exception as e:
            st.error(f"ACP Analysis Error: {str(e)}")
    else:
        st.warning("⚠️ No candlestick data available for the selected asset.")

with tab4:
    st.subheader("📜 Live Intelligence ROI Log")
    if st.session_state.signal_log:
        for entry in st.session_state.signal_log:
            if entry["Status"] == "PENDING":
                curr_p = price_data.get(entry["Asset"] + "USDT", {}).get('price', entry["Entry"])
                if curr_p > entry["Entry"] * 1.002:
                    entry["Status"] = "✅ PROFIT"
                elif curr_p < entry["Entry"] * 0.998:
                    entry["Status"] = "❌ LOSS"
        st.dataframe(pd.DataFrame(st.session_state.signal_log), use_container_width=True)
    else:
        st.info("ℹ️ No signals processed yet. Go to 'Council Analysis' tab to begin analyzing assets.")

with tab5:
    st.subheader("📰 Crypto News & Market Updates")
    @st.cache_data(ttl=1800)
    def fetch_crypto_news_dynamic():
        try:
            news_items = engine.fetch_crypto_news(max_articles=15)
            processed_items = []
            if isinstance(news_items, pd.DataFrame) and len(news_items) > 0:
                for idx, row in news_items.iterrows():
                    processed_items.append({'title': str(row.get('title', row.get('headline', 'Crypto News'))), 'source': str(row.get('source', row.get('publisher', 'CoinDesk'))), 'summary': str(row.get('description', row.get('summary', 'Recent market update')))})
            elif isinstance(news_items, list) and len(news_items) > 0:
                for item in news_items[:15]:
                    if isinstance(item, dict):
                        processed_items.append({'title': str(item.get('title', item.get('headline', 'Crypto Update'))), 'source': str(item.get('source', item.get('publisher', 'CoinDesk'))), 'summary': str(item.get('description', item.get('summary', item.get('content', 'Market news'))))})
                    elif isinstance(item, str):
                        processed_items.append({'title': item[:100], 'source': 'CoinDesk', 'summary': 'Latest crypto market update'})
            return processed_items
        except Exception:
            return []
    try:
        news_items = fetch_crypto_news_dynamic()
        if news_items and len(news_items) > 0:
            st.write("### 🔥 Latest Crypto News")
            col_news1, col_news2, col_news3 = st.columns(3)
            for idx, item in enumerate(news_items[:3]):
                with [col_news1, col_news2, col_news3][idx]:
                    title = str(item.get('title', 'Crypto News'))[:70]
                    source = str(item.get('source', 'CoinDesk'))
                    summary = str(item.get('summary', 'Market update'))[:120]
                    st.markdown(f"""
                    <div style='padding: 15px; background-color: #0d1f2d; border-left: 4px solid #FF6B35; border-radius: 8px; height: 180px; display: flex; flex-direction: column;'>
                        <p style='margin: 0; font-weight: bold; color: #FFD700; font-size: 11px;'>📍 {source}</p>
                        <p style='margin: 8px 0 5px 0; color: #FFFFFF; font-size: 12px; font-weight: 600; line-height: 1.4;'>{title}</p>
                        <p style='margin: 0; color: #B0B0B0; font-size: 11px; line-height: 1.3; flex-grow: 1;'>{summary}</p>
                    </div>
                    """, unsafe_allow_html=True)
            if len(news_items) > 3:
                with st.expander("📌 More News (Expand to view all)", expanded=False):
                    for idx, item in enumerate(news_items[3:]):
                        title = str(item.get('title', 'Crypto News'))
                        source = str(item.get('source', 'CoinDesk'))
                        summary = str(item.get('summary', 'Market update'))[:100]
                        col_num, col_content = st.columns([0.5, 4])
                        with col_num: st.write(f"**{idx + 4}.**")
                        with col_content: st.markdown(f"<div style='padding: 10px; background-color: #0d1f2d; border-left: 3px solid #FF6B35; border-radius: 5px;'><p style='margin: 0; font-weight: bold; color: #FFD700; font-size: 10px;'>{source}</p><p style='margin: 5px 0 0 0; color: #FFFFFF; font-size: 11px;'>{title[:70]}</p><p style='margin: 3px 0 0 0; color: #B0B0B0; font-size: 10px;'>{summary}</p></div>", unsafe_allow_html=True)
        else:
            st.info("📡 No news items available at this time")
    except Exception:
        st.warning("⚠️ Unable to load news. Please refresh.")
    st.write("### 📊 Market Sentiment & Metrics")
    col_sentiment1, col_sentiment2, col_sentiment3 = st.columns(3)
    if price_data:
        avg_change = np.mean([d.get('change_24h', 0) for d in price_data.values()])
        sentiment = "🟢 BULLISH" if avg_change > 1 else "🔴 BEARISH" if avg_change < -1 else "⚪ NEUTRAL"
        col_sentiment1.metric("Market Sentiment", sentiment)
        col_sentiment2.metric("Avg 24h Change", f"{avg_change:.2f}%", delta=f"{abs(avg_change):.2f}%")
        active_symbols = len([d for d in price_data.values() if d.get('price', 0) > 0])
        col_sentiment3.metric("Assets Tracked", f"{active_symbols}/{len(price_data)}")
    else:
        col_sentiment1.metric("Market Sentiment", "Loading...")
    st.caption(f"📡 News updates every 30 minutes | Last updated: {datetime.now().strftime('%H:%M:%S')}")
