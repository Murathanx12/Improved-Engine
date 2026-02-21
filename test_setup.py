import os
from dotenv import load_dotenv
load_dotenv()

print("Testing packages...")
import numpy, pandas, yfinance, fredapi, finnhub
print("  [OK] All packages imported")

print("\nTesting FRED API...")
fred = fredapi.Fred(api_key=os.getenv('FRED_API_KEY'))
vix = fred.get_series('VIXCLS', limit=5)
print(f"  [OK] VIX: {vix.iloc[-1]:.2f}")
spread = fred.get_series('T10Y3M', limit=5)
print(f"  [OK] Yield Spread: {spread.iloc[-1]:.2f}%")
sahm = fred.get_series('SAHMREALTIME', limit=5)
print(f"  [OK] Sahm Rule: {sahm.iloc[-1]:.2f}")

print("\nTesting Finnhub...")
client = finnhub.Client(api_key=os.getenv('FINNHUB_API_KEY'))
quote = client.quote('SPY')
print(f"  [OK] SPY: ${quote['c']:.2f}")

print("\nTesting yfinance...")
import yfinance as yf
spy = yf.download('SPY', period='5d', progress=False)
close = spy['Close'].iloc[-1]
if hasattr(close, 'iloc'):
    close = close.iloc[0]
print(f"  [OK] SPY last close: ${float(close):.2f}")

print("\n✅ Setup complete. All systems go.")