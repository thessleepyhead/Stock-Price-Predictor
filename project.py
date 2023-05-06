import yfinance as yf
import pandas as pd
from pandas_datareader import data as web
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
from textblob import TextBlob
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Define the stock symbol and time range
stock_symbol = "GOOG"
start_date = "2015-01-01"
end_date = "2020-12-31"

# Retrieve historical stock prices
stock_data = web.DataReader(stock_symbol, "yahoo", start=start_date, end=end_date)

# Retrieve financial statements
income_statement = yf.Ticker(stock_symbol).financials.loc["Total Revenue"]
balance_sheet = yf.Ticker(stock_symbol).balance_sheet.loc["Total Current Assets":"Total Liabilities"]
cash_flow = yf.Ticker(stock_symbol).cashflow.loc["Total Cash From Operating Activities":"Total Cashflows From Investing Activities"]

# Retrieve analyst recommendations
recommendations = yf.Ticker(stock_symbol).recommendations

# Retrieve news articles
url = f"https://finance.yahoo.com/quote/{stock_symbol}/news?p={stock_symbol}"
response = requests.get(url)
soup = BeautifulSoup(response.text, "html.parser")
news_headlines = []
for headline in soup.select(".js-stream-content h3"):
    news_headlines.append(headline.get_text())
news_sentiments = []
for headline in news_headlines:
    blob = TextBlob(headline)
    news_sentiments.append(blob.sentiment.polarity)

# Retrieve market trends data
market_data = yf.Ticker(stock_symbol).sustainability

# Combine all data into a single dataframe
df = pd.DataFrame()
df["Date"] = stock_data.index
df["Close"] = stock_data["Close"]
df["Total Revenue"] = income_statement
df["Total Current Assets"] = balance_sheet[:1].values.flatten()
df["Total Liabilities"] = balance_sheet[-1:].values.flatten()
df["Total Cash From Operating Activities"] = cash_flow[:1].values.flatten()
df["Total Cashflows From Investing Activities"] = cash_flow[-1:].values.flatten()
df["Recommendations"] = recommendations["To Grade"]
df["News Headlines"] = news_headlines
df["News Sentiments"] = news_sentiments
df["Sustainability Score"] = market_data["Value"].values.flatten()

# Plot the historical stock prices
plt.plot(df["Date"], df["Close"])
plt.title(f"{stock_symbol} Stock Price")
plt.xlabel("Date")
plt.ylabel("Price ($)")
plt.show()

# Perform linear regression to predict future stock prices
X = df[["Total Revenue", "Total Current Assets", "Total Liabilities", "Total Cash From Operating Activities", "Total Cashflows From Investing Activities", "News Sentiments", "Sustainability Score"]]
y = df["Close"]
model = LinearRegression()
model.fit(X, y)
future_dates = pd.date_range(end_date, end_date + timedelta(days=365))
future_df = pd.DataFrame({"Date": future_dates})
future_X = df[["Total Revenue", "Total Current Assets", "Total Liabilities", "Total Cash From Operating Activities", "Total Cashflows From Investing Activities", "News Sentiments", "Sustainability Score"]]
future_y = model.predict(future_X)
future_df["Close"] = future_y
plt.plot(df["Date"], df["Close"])
plt.plot(future_df["Date"], future_df["Close"])
plt.title(f"{stock_symbol} Stock Price Prediction")
