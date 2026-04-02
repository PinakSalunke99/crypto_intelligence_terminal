import os
import ollama
import pandas as pd
import numpy as np
from prophet import Prophet
from dotenv import load_dotenv
import random

class MultiAgentCouncil:
    def __init__(self, model_name='mistral'):
        self.model = model_name
        # Internal context store for the chat assistant
        self.market_context = ""
        self.ollama_available = False
        self._check_ollama_availability()
    
    def _check_ollama_availability(self):
        """Check if Ollama is available and accessible."""
        try:
            load_dotenv(override=True)
            host = os.getenv('OLLAMA_HOST', 'http://127.0.0.1:11434')
            client = ollama.Client(host=host, timeout=5.0)
            # Try a simple tags call to verify connectivity
            client.list()
            self.ollama_available = True
            print("✅ Ollama AI Engine Connected")
        except Exception as e:
            self.ollama_available = False
            print(f"⚠️ Ollama Not Available - Using Synthetic Intelligence: {str(e)[:50]}")

    def get_analyst_view(self, text_batch):
        """Phase 1: Analyst identifies sentiment and trends."""
        combined_text = "\n".join(text_batch[:5])
        prompt = f"Analyze the following crypto news/social data: {combined_text}. Identify the current market sentiment (Bullish/Bearish/Neutral) and the primary drivers."
        
        if self.ollama_available:
            return self._chat(prompt, "You are a Senior Crypto Analyst.")
        else:
            return self._synthetic_analyst_view(combined_text)

    def get_critic_view(self, analyst_view):
        """Phase 2: Critic plays Devil's Advocate to identify risks."""
        prompt = f"The primary analyst says: '{analyst_view}'. Find the flaws in this logic. What are the 'Bearish' risks or counter-indicators even if the sentiment seems Bullish? Be critical."
        
        if self.ollama_available:
            return self._chat(prompt, "You are a Professional Risk Manager (Critic).")
        else:
            return self._synthetic_critic_view(analyst_view)

    def get_synthesized_view(self, analyst_view, critic_view):
        """Phase 3: Synthesizer provides the final consensus and signal."""
        prompt = (
            f"Analyst: {analyst_view}\nCritic: {critic_view}\n"
            f"Generate a final signal in this EXACT format: LABEL | SCORE | REASONING. "
            f"(LABEL: BULLISH/BEARISH/NEUTRAL, SCORE: 0.0-1.0)"
        )
        
        if self.ollama_available:
            return self._chat(prompt, "You are a Chief Investment Officer (Synthesizer).")
        else:
            return self._synthetic_synthesized_view(analyst_view, critic_view)

    def detect_manipulation(self, price_change, sentiment_score):
        """
        Hackathon Feature: Manipulation Detector.
        Identifies Pump & Dump patterns where price moves without organic news support.
        """
        # Logic: High price volatility with low sentiment confirmation = Potential Manipulation
        if abs(price_change) > 5.0 and abs(sentiment_score - 0.5) < 0.15:
            return "🚨 ALERT: Potential Price Manipulation Detected (Volatility without organic news support)."
        return "✅ Market Movement appears organic."
    
    def _synthetic_analyst_view(self, text_data):
        """Generate synthetic analyst view when Ollama is unavailable."""
        sentiments = ["BULLISH", "BEARISH", "NEUTRAL"]
        drivers = [
            "institutional buying pressure",
            "regulatory clarity improving",
            "technical resistance breached",
            "whale accumulation detected",
            "network activity declining",
            "macroeconomic uncertainty"
        ]
        
        sentiment = random.choice(sentiments)
        driver = random.choice(drivers)
        confidence = round(0.55 + random.random() * 0.35, 2)  # 0.55-0.90
        
        return f"{sentiment} | {confidence} | Primary driver: {driver}. Market showing mixed signals."
    
    def _synthetic_critic_view(self, analyst_view):
        """Generate synthetic critic view when Ollama is unavailable."""
        risks = [
            "High leverage exposure could trigger liquidations",
            "Regulatory crackdown could reverse gains quickly",
            "Macro headwinds may contradict technical strength",
            "Overbought conditions suggest pullback risks",
            "Sentiment extremes often precede reversals"
        ]
        
        risk = random.choice(risks)
        counter_conf = round(0.45 + random.random() * 0.35, 2)
        
        return f"COUNTER VIEW | {counter_conf} | {risk}. Recommend caution until confirmation."
    
    def _synthetic_synthesized_view(self, analyst_view, critic_view):
        """Generate synthetic final signal when Ollama is unavailable."""
        # Parse analyst sentiment if possible, otherwise randomize
        if "BULLISH" in analyst_view:
            base_signal = "BULLISH"
            base_score = 0.62
        elif "BEARISH" in analyst_view:
            base_signal = "BEARISH"
            base_score = 0.38
        else:
            base_signal = "NEUTRAL"
            base_score = 0.50
        
        # Adjust based on balance between analyst and critic
        final_score = base_score + (random.random() - 0.5) * 0.1
        final_score = max(0.0, min(1.0, final_score))  # Clamp 0-1
        
        reasoning = f"Balanced view accounting for both bullish technicals and bearish macro risks. Current market conditions favor cautious positioning."
        
        return f"{base_signal} | {final_score:.2f} | {reasoning}"

    def get_assistant_response(self, user_query, current_market_summary):
        """Phase 4: Provides ChatGPT-like interaction using live terminal context."""
        prompt = f"User Question: {user_query}\n\nLive Market State: {current_market_summary}"
        return self._chat(prompt, "You are a Professional Crypto Assistant. Answer based on the provided live market data.")

    def _chat(self, user_content, system_role):
        try:
            load_dotenv(override=True)
            host = os.getenv('OLLAMA_HOST', 'http://127.0.0.1:11434')
            client = ollama.Client(host=host, timeout=180.0)
            
            response = client.chat(model='mistral', messages=[
                {'role': 'system', 'content': system_role},
                {'role': 'user', 'content': user_content or "Provide current market insight."}
            ])
            return response['message']['content'].strip()
        except Exception as e:
            return f"NEUTRAL | 0.50 | AI Link Interrupted (Reason: {str(e)[:40]}...)"

def get_llm_sentiment(text_batch):
    """Main entry point for Council-based intelligence."""
    council = MultiAgentCouncil()
    analyst = council.get_analyst_view(text_batch)
    critic = council.get_critic_view(analyst)
    final = council.get_synthesized_view(analyst, critic)
    
    # Pack reasoning for visual 'Waterfall'
    reasoning_waterfall = f"**Analyst:** {analyst}\n\n**Risk Critic:** {critic}"
    return final, reasoning_waterfall

def predict_directional_movement(df):
    """Time-series forecasting with specific Lag Range calculation."""
    try:
        df_prophet = df.rename(columns={'timestamp': 'ds', 'close': 'y'})
        m = Prophet(changepoint_prior_scale=0.05, interval_width=0.95)
        m.fit(df_prophet)
        
        future = m.make_future_dataframe(periods=96, freq='15min')
        forecast = m.predict(future)
        
        curr = df['close'].iloc[-1]
        results = {}
        for label, step in [('1h', 4), ('4h', 16), ('24h', 96)]:
            predicted_price = float(forecast['yhat'].iloc[-(97-step)])
            
            # LAG RANGE Calculation (±0.5%)
            range_lower = predicted_price * 0.995
            range_upper = predicted_price * 1.005
            
            diff = ((predicted_price - curr) / curr) * 100
            conf = 1 - ((forecast['yhat_upper'].iloc[-(97-step)] - forecast['yhat_lower'].iloc[-(97-step)]) / predicted_price)
            
            results[label] = {
                "direction": "UP" if diff > 0.3 else "DOWN" if diff < -0.3 else "SIDEWAYS",
                "price": round(predicted_price, 2),
                "range": f"${round(range_lower, 2)} - ${round(range_upper, 2)}",
                "change": f"{diff:.2f}%",
                "confidence": f"{max(0, conf):.2%}"
            }
        return results
    except:
        return {h: {"direction": "SIDEWAYS", "price": 0, "range": "N/A", "change": "0.00%", "confidence": "0%"} for h in ['1h', '4h', '24h']}
        return {h: {"direction": "SIDEWAYS", "price": 0, "range": "N/A", "change": "0.00%", "confidence": "0%"} for h in ['1h', '4h', '24h']}, None