
import requests
import numpy as np
import pandas as pd
from scipy.stats import norm
import json
import os # Added for environment variables
from datetime import datetime, timedelta

# API Keys will be read from environment variables
# TWELVE_DATA_API_KEY = os.environ.get("TWELVE_DATA_API_KEY")
# GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

def get_forex_data_twelvedata(symbol="EUR/USD", interval="1day", outputsize=252):
    api_key = os.environ.get("TWELVE_DATA_API_KEY")
    if not api_key:
        print("Error: TWELVE_DATA_API_KEY environment variable not set.")
        return None
        
    base_url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": symbol,
        "interval": interval,
        "apikey": api_key,
        "outputsize": outputsize,
        "timezone": "UTC"
    }
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        if data.get("status") == "error":
            print(f"Error from Twelve Data API: {data.get('message')}")
            return None
        
        df = pd.DataFrame(data["values"])
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.sort_values("datetime", ascending=True)
        df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
        return df
    except requests.exceptions.RequestException as e:
        print(f"RequestException fetching data from Twelve Data: {e}")
        return None
    except KeyError as e:
        print(f"KeyError parsing Twelve Data response (likely unexpected format or error): {e}, Response: {data}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred in get_forex_data_twelvedata: {e}")
        return None

def estimate_historical_parameters(prices_df):
    if prices_df is None or prices_df.empty or 'close' not in prices_df.columns:
        return 0.0, 0.05, 0.0 
    prices = prices_df['close']
    log_returns = np.log(prices / prices.shift(1)).dropna()
    if log_returns.empty:
        return 0.0, 0.05, 0.0

    mu = log_returns.mean() * 252
    sigma = log_returns.std() * np.sqrt(252)
    current_price = prices.iloc[-1]
    
    sigma = min(sigma, 1.0) 
    sigma = max(sigma, 0.01)
    mu = min(mu, 0.5) 
    mu = max(mu, -0.5)

    return mu, sigma, current_price

def analyze_liquidity(prices_df, window=20):
    if prices_df is None or 'volume' not in prices_df.columns or len(prices_df) < window:
        return "unknown", 0.0
    
    current_volume = prices_df['volume'].iloc[-1]
    avg_volume = prices_df['volume'].rolling(window=window).mean().iloc[-1]
    
    if avg_volume == 0: 
        liquidity_status = "unknown"
    elif current_volume < avg_volume * 0.6:
        liquidity_status = "low"
    elif current_volume > avg_volume * 1.4:
        liquidity_status = "high"
    else:
        liquidity_status = "normal"
    return liquidity_status, current_volume

def call_gemini_api(prompt_text):
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return {"error": "Gemini API key not configured (GEMINI_API_KEY environment variable not set)."}

    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
    
    payload = {
        "contents": [{"parts": [{"text": prompt_text}]}],
        "generationConfig": {
            "temperature": 0.7,
            "topK": 1,
            "topP": 1,
            "maxOutputTokens": 2048,
        }
    }
    try:
        response = requests.post(api_url, json=payload, headers={'Content-Type': 'application/json'})
        response.raise_for_status()
        result = response.json()

        if "candidates" in result and len(result["candidates"]) > 0:
            if "content" in result["candidates"][0] and "parts" in result["candidates"][0]["content"]:
                if len(result["candidates"][0]["content"]["parts"]) > 0:
                    return {"text": result["candidates"][0]["content"]["parts"][0]["text"]}
        return {"error": "Unexpected response structure from Gemini.", "details": result}

    except requests.exceptions.RequestException as e:
        return {"error": f"Gemini API request failed: {e}"}
    except Exception as e:
        return {"error": f"An unexpected error occurred calling Gemini: {e}"}


def generate_gemini_prompt(symbol, current_price, mu, sigma, liquidity_status, recent_volume, prices_df):
    recent_prices = prices_df['close'].tail(5).tolist() if prices_df is not None else "N/A"
    prompt = f"""
    Analyze the current market conditions for {symbol} and provide a brief outlook.
    Current Price: {current_price:.4f}
    Estimated Annualized Drift (from historical data): {mu:.4f}
    Estimated Annualized Volatility (from historical data): {sigma:.4f}
    Recent Closing Prices (last 5 periods): {recent_prices}
    Current Liquidity Assessment (based on volume): {liquidity_status}
    Most Recent Volume: {recent_volume:,.0f}

    Based on this data:
    1. What is your overall short-term sentiment (e.g., Bullish, Bearish, Neutral) for {symbol}?
    2. Briefly list 2-3 key potential bullish factors for {symbol} right now.
    3. Briefly list 2-3 key potential bearish factors for {symbol} right now.
    4. Provide a concise summary paragraph of your analysis and outlook.
    5. Considering the volatility and liquidity, suggest a general level of caution for traders (e.g., Low, Medium, High).

    Format your response clearly.
    Example for factors:
    - Bullish Factor 1: [Description]
    - Bearish Factor 1: [Description]
    """
    return prompt

def parse_gemini_response(gemini_text_response):
    parsed = {
        "sentiment": "unknown",
        "summary": "Could not parse Gemini summary.",
        "key_bullish_factors": [],
        "key_bearish_factors": [],
        "caution_level": "medium" 
    }
    if not gemini_text_response or "error" in gemini_text_response:
        parsed["summary"] = gemini_text_response.get("error", "Gemini API error")
        return parsed

    text = gemini_text_response.get("text", "")
    lines = text.split('\n')
    for i, line in enumerate(lines):
        line_lower = line.lower()
        if "overall short-term sentiment" in line_lower or "sentiment:" in line_lower :
            if "bullish" in line_lower: parsed["sentiment"] = "bullish"
            elif "bearish" in line_lower: parsed["sentiment"] = "bearish"
            elif "neutral" in line_lower: parsed["sentiment"] = "neutral"
        elif "bullish factor" in line_lower or (line.strip().startswith("-") and i > 0 and "bullish" in lines[i-1].lower()):
            parsed["key_bullish_factors"].append(line.strip().replace("-","").strip())
        elif "bearish factor" in line_lower or (line.strip().startswith("-") and i > 0 and "bearish" in lines[i-1].lower()):
            parsed["key_bearish_factors"].append(line.strip().replace("-","").strip())
        elif "summary paragraph" in line_lower or "summary:" in line_lower:
            summary_parts = []
            for j in range(i + 1, len(lines)):
                if lines[j].strip() == "" or "factor" in lines[j].lower() or "caution" in lines[j].lower():
                    break
                summary_parts.append(lines[j].strip())
            if summary_parts:
                parsed["summary"] = " ".join(summary_parts)
            elif "summary:" in line_lower:
                 parsed["summary"] = line.split("summary:",1)[-1].strip()
        elif "level of caution" in line_lower or "caution:" in line_lower:
            if "low" in line_lower: parsed["caution_level"] = "low"
            elif "high" in line_lower: parsed["caution_level"] = "high"
            elif "medium" in line_lower: parsed["caution_level"] = "medium"
            
    if parsed["summary"] == "Could not parse Gemini summary." and text:
        first_few_lines = lines[:5] 
        parsed["summary"] = " ".join([l.strip() for l in first_few_lines if l.strip() and "factor" not in l.lower()])
    return parsed


def monte_carlo_gbm(S0, mu, sigma, T_days, dt_days, n_simulations):
    if S0 <= 0 or sigma <= 0: 
        print(f"Warning: Invalid inputs for GBM (S0={S0}, sigma={sigma}). Returning current price.")
        paths = np.full((int(T_days / dt_days) + 1, n_simulations), S0)
        return paths

    T_years = T_days / 252.0
    dt_years = dt_days / 252.0
    n_steps = int(T_years / dt_years)
    
    paths = np.zeros((n_steps + 1, n_simulations))
    paths[0] = S0
    
    for i_sim in range(n_simulations):
        for i_step in range(1, n_steps + 1):
            Z = norm.ppf(np.random.rand()) 
            paths[i_step, i_sim] = paths[i_step - 1, i_sim] * np.exp(
                (mu - 0.5 * sigma**2) * dt_years + sigma * np.sqrt(dt_years) * Z
            )
    return paths

def get_portfolio_suggestion(prob_up, prob_down, volatility, liquidity_status, gemini_sentiment, gemini_caution):
    base_allocation_pct = 2.0 
    
    if volatility > 0.15: 
        base_allocation_pct *= 0.75
    if volatility > 0.25: 
        base_allocation_pct *= 0.5

    if liquidity_status == "low":
        base_allocation_pct *= 0.6

    if gemini_caution == "high":
        base_allocation_pct *= 0.5
    elif gemini_caution == "low" and gemini_sentiment != "unknown":
         base_allocation_pct *= 1.25
    
    if gemini_sentiment == "bullish" and prob_up > 0.55:
        base_allocation_pct *= 1.1
    elif gemini_sentiment == "bearish" and prob_down > 0.55:
        base_allocation_pct *= 1.1 

    final_allocation_pct = max(0.5, min(base_allocation_pct, 5.0)) 

    reasoning_parts = [
        f"Volatility is {'{:.2%}'.format(volatility)}.",
        f"Liquidity is {liquidity_status}.",
        f"Gemini sentiment: {gemini_sentiment} (Caution: {gemini_caution})."
    ]
    if prob_up > 0.5:
        reasoning_parts.append(f"Model suggests {'{:.1%}'.format(prob_up)} chance of upward movement.")
    else:
        reasoning_parts.append(f"Model suggests {'{:.1%}'.format(prob_down)} chance of downward movement.")

    advice = f"Consider allocating no more than {final_allocation_pct:.1f}% of trading capital."
    reasoning = " ".join(reasoning_parts)
    
    return {"advice": advice, "reasoning": reasoning}

def analyze_eur_usd(prediction_horizon_days=30, num_simulations=1000):
    symbol = "EUR/USD"
    output = {
        "asset": symbol,
        "current_price": None,
        "analysis_timestamp": datetime.utcnow().isoformat() + "Z",
        "data_source": "Twelve Data",
        "liquidity_assessment": "unknown",
        "current_volume": None,
        "volatility_annualized": None,
        "drift_annualized_historical": None,
        "gemini_analysis": {"error": "Not run"},
        "prediction_horizon_days": prediction_horizon_days,
        "simulation_parameters": {"num_paths": num_simulations, "time_step_days": 1},
        "simulated_price_info": None,
        "signal_trust_rate": "low", 
        "trade_reliability_probability": 0.0,
        "portfolio_management_suggestion": None,
        "errors": []
    }

    prices_df = get_forex_data_twelvedata(symbol=symbol, interval="1day", outputsize=300)

    if prices_df is None or prices_df.empty:
        output["errors"].append("Failed to fetch or process market data from Twelve Data.")
        # Return early if core data is missing
        output["gemini_analysis"]["error"] = "Market data unavailable for Gemini analysis."
        output["portfolio_management_suggestion"] = {
            "advice": "Cannot provide suggestion due to missing market data.",
            "reasoning": "Market data fetch failed."
        }
        return output # Critical error, return immediately
    
    mu, sigma, S0 = estimate_historical_parameters(prices_df)
    output["current_price"] = S0
    output["volatility_annualized"] = sigma
    output["drift_annualized_historical"] = mu

    liquidity_status, current_volume = analyze_liquidity(prices_df)
    output["liquidity_assessment"] = liquidity_status
    output["current_volume"] = current_volume
    
    # Only call Gemini if we have some basic data
    if S0 > 0:
        gemini_prompt = generate_gemini_prompt(symbol, S0, mu, sigma, liquidity_status, current_volume, prices_df.tail(60))
        gemini_response_raw = call_gemini_api(gemini_prompt)
        
        if "error" in gemini_response_raw:
            output["errors"].append(f"Gemini API call failed: {gemini_response_raw['error']}")
            output["gemini_analysis"] = {"error": gemini_response_raw['error'], "summary": gemini_response_raw['error']}
        else:
            gemini_parsed_response = parse_gemini_response(gemini_response_raw)
            output["gemini_analysis"] = gemini_parsed_response
    else:
        output["errors"].append("Skipped Gemini API call due to invalid current price (S0).")
        output["gemini_analysis"]["error"] = "Skipped due to invalid current price."


    dt_days = 1 
    if S0 > 0 and sigma > 0: # Only run simulation if S0 and sigma are valid
        simulated_paths = monte_carlo_gbm(S0, mu, sigma, prediction_horizon_days, dt_days, num_simulations)
        
        if simulated_paths is not None:
            final_prices = simulated_paths[-1, :]
            median_final_price = np.median(final_prices)
            percentile_5 = np.percentile(final_prices, 5)
            percentile_95 = np.percentile(final_prices, 95)
            
            prob_up = np.mean(final_prices > S0)
            prob_down = np.mean(final_prices < S0)

            output["simulated_price_info"] = {
                "median_expected_price": median_final_price,
                "5th_percentile_price": percentile_5,
                "95th_percentile_price": percentile_95,
                "probability_price_up": prob_up,
                "probability_price_down": prob_down
            }
            
            price_range_normalized = (percentile_95 - percentile_5) / S0 if S0 > 0 else float('inf')
            
            if price_range_normalized < 0.05 and abs(prob_up - 0.5) > 0.1: 
                output["signal_trust_rate"] = "medium"
            if price_range_normalized < 0.03 and abs(prob_up - 0.5) > 0.15: 
                 output["signal_trust_rate"] = "high"
            
            output["trade_reliability_probability"] = abs(prob_up - 0.5) * 2.0 
            if output["gemini_analysis"].get("caution_level") == "high":
                output["trade_reliability_probability"] *= 0.7
                output["signal_trust_rate"] = "low" if output["signal_trust_rate"] == "medium" else output["signal_trust_rate"]
    else:
        output["errors"].append("Skipped GBM simulation due to invalid S0 or sigma.")
        output["simulated_price_info"] = None


    gemini_sentiment = output["gemini_analysis"].get("sentiment", "unknown")
    gemini_caution = output["gemini_analysis"].get("caution_level", "medium")

    if output["simulated_price_info"]:
        output["portfolio_management_suggestion"] = get_portfolio_suggestion(
            output["simulated_price_info"]["probability_price_up"],
            output["simulated_price_info"]["probability_price_down"],
            sigma,
            liquidity_status,
            gemini_sentiment,
            gemini_caution
        )
    else:
        # Ensure portfolio suggestion is set even if simulation fails
        output["portfolio_management_suggestion"] = {
            "advice": "Cannot provide portfolio suggestion due to issues in prior analysis steps.",
            "reasoning": "Simulation did not run or market data was incomplete."
        }
        if not ("Market data fetch failed." in output["portfolio_management_suggestion"]["reasoning"] or \
                "Market data unavailable for Gemini analysis." in output["gemini_analysis"].get("error", "")):
             output["errors"].append("GBM simulation did not produce results needed for portfolio suggestion.")


    return output

# --- Netlify Handler Function ---
def handler(event, context):
    prediction_horizon = 30
    num_simulations = 500 # Reduced for potentially faster serverless execution

    if event.get('queryStringParameters'):
        params = event['queryStringParameters']
        try:
            prediction_horizon = int(params.get('horizon', prediction_horizon))
            num_simulations = int(params.get('sims', num_simulations))
        except ValueError:
            # Handle cases where conversion might fail, keep defaults
            print("Warning: Could not parse query string parameters for horizon/sims. Using defaults.")
            pass


    analysis_results = analyze_eur_usd(
        prediction_horizon_days=prediction_horizon,
        num_simulations=num_simulations
    )

    return {
        "statusCode": 200,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*", 
            "Access-Control-Allow-Headers": "Content-Type",
            "Access-Control-Allow-Methods": "GET, OPTIONS" 
        },
        "body": json.dumps(analysis_results)
    }

# Removed the if __name__ == '__main__': block
