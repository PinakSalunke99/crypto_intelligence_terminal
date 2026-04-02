# 🚀 Self-Hosted Crypto Intelligence Terminal
**NMIMS INNOVATHON 2026 | Challenge 2: AI & Data Intelligence**

## 1. Executive Summary
A full-stack trading intelligence terminal using **locally-hosted LLMs (Mistral 7B)** to analyze social sentiment and **Prophet** for time-series forecasting. [cite_start]The system provides real-time, explainable trading signals without relying on expensive cloud APIs[cite: 19, 20].

## 2. Technical Architecture

- [cite_start]**Data Ingestion**: Real-time pipelines for Binance (Price), Reddit (Sentiment), and Etherscan (Whale Alerts)[cite: 44].
- [cite_start]**Sentiment Engine**: Ollama-powered Mistral 7B using few-shot prompting for 75% accuracy[cite: 50, 51].
- [cite_start]**Price Forecasting**: Facebook Prophet model for 1h, 4h, and 24h directional prediction[cite: 60].
- [cite_start]**UI**: High-performance Streamlit dashboard for real-time visualization[cite: 83].

## 3. Performance Metrics
- [cite_start]**Target Directional Accuracy**: >55%.
- **Backtest Win Rate**: Tracked via 30-day historical simulation.
- [cite_start]**Latency**: Local LLM inference < 2 seconds on 16GB RAM[cite: 93].

## 4. Setup Instructions
1. Install [Ollama](https://ollama.com) and run `ollama pull mistral`.
2. Clone this repo and install dependencies: `pip install -r requirements.txt`.
3. Configure your API keys in the `.env` file.
4. Launch the terminal: `streamlit run terminal.py`.