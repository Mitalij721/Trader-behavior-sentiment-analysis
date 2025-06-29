import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Loading data
sentiment_df = pd.read_csv(r'C:\Mitali\placement prep\Task\Project\fear_greed_index.csv')
trader_df = pd.read_csv(r'C:\Mitali\placement prep\Task\Project\historical_data.csv')

# Check the first few rows
print(sentiment_df.head())
print(trader_df.head())

# Check data types and missing values
print(sentiment_df.info())
print(trader_df.info())

# 1. Clean Sentiment Data
sentiment_df = sentiment_df.rename(columns={
    'timestamp': 'unix_timestamp',
    'value': 'fear_greed_value',
    'classification': 'sentiment_class'
})
sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
sentiment_df = sentiment_df[['date', 'fear_greed_value', 'sentiment_class']]

# 2. Clean Trader Data
trader_df['Trade_Date'] = pd.to_datetime(trader_df['Timestamp'], unit='ms').dt.date
trader_df['Trade_DateTime'] = pd.to_datetime(trader_df['Timestamp'], unit='ms')
trader_df = trader_df.rename(columns={'Closed PnL': 'closed_pnl'})

# FIXED: Keep 'Size USD' column for later calculations
trader_df = trader_df[[
    'Account', 'Coin', 'Execution Price', 'Size Tokens', 'Size USD',
    'Side', 'Trade_Date', 'closed_pnl', 'Fee'
]]

print("Sentiment Data:\n", sentiment_df.head())
print("\nTrader Data:\n", trader_df.head())

# Convert trader_df['Trade_Date'] to datetime64[ns]
trader_df['Trade_Date'] = pd.to_datetime(trader_df['Trade_Date'])

# Now merge
merged_df = pd.merge(
    trader_df,
    sentiment_df,
    left_on='Trade_Date',
    right_on='date',
    how='left'
)

# Check the first few rows and shape
print("Merged Data Shape:", merged_df.shape)
print("\nMerged Data Sample:\n", merged_df.head())

# Check for missing sentiment 
print("\nNumber of missing sentiment_class values:", 
      merged_df['sentiment_class'].isnull().sum())

plt.figure(figsize=(10, 6))
sns.countplot(x='sentiment_class', data=merged_df)
plt.title('Number of Trades by Market Sentiment')
plt.show()

plt.figure(figsize=(12, 7))
sns.boxplot(x='sentiment_class', y='closed_pnl', data=merged_df, showfliers=False)
plt.title('Profit/Loss Distribution by Market Sentiment')
plt.ylabel('Closed PnL (USD)')
plt.show()

print("\nPerformance by Sentiment:\n", 
      merged_df.groupby('sentiment_class')['closed_pnl'].agg(['mean', 'median', 'count']))

# Filter Out Trades Without Sentiment Data
merged_with_sentiment = merged_df[merged_df['sentiment_class'].notna()]
print("Trades with known sentiment:", merged_with_sentiment.shape[0])

# Statistical Testing
groups = merged_with_sentiment.groupby('sentiment_class')['closed_pnl'].apply(list)
f_value, p_value = stats.f_oneway(groups['Greed'], groups['Fear'])
print(f"ANOVA result for Greed vs Fear: p-value = {p_value:.4f}")

# Top Performers Analysis
top_traders = merged_with_sentiment.groupby(['Account', 'sentiment_class'])['closed_pnl'].sum().unstack().sort_values(by='Greed', ascending=False)
print(top_traders.head())

# Time Series Analysis
time_series = merged_with_sentiment.groupby(['Trade_Date', 'sentiment_class'])['closed_pnl'].mean().unstack()
time_series.plot(figsize=(15,6))
plt.title('Average Profit/Loss by Sentiment Over Time')
plt.ylabel('Average Closed PnL (USD)')
plt.show()

# Machine Learning
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Dummy encode sentiment
X = pd.get_dummies(merged_with_sentiment[['sentiment_class', 'Size Tokens', 'Fee']])
y = merged_with_sentiment['closed_pnl']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, y_train)
print("Model score:", model.score(X_test, y_test))

# FIXED: Additional Analysis with proper error handling
print("\n=== ADDITIONAL ANALYSIS ===")

# Risk-adjusted returns (now with Size USD available)
# Handle division by zero by replacing 0 with NaN
merged_with_sentiment['risk_adj_return'] = merged_with_sentiment['closed_pnl'] / merged_with_sentiment['Size USD'].replace(0, np.nan)

print("\nRisk-Adjusted Returns by Sentiment:")
risk_adj_by_sentiment = merged_with_sentiment.groupby('sentiment_class')['risk_adj_return'].agg(['mean', 'median', 'count'])
print(risk_adj_by_sentiment)

# Win rate by sentiment
print("\nWin Rates by Sentiment:")
win_rates = merged_with_sentiment.groupby('sentiment_class').apply(
    lambda x: (x['closed_pnl'] > 0).sum() / len(x)
)
print(win_rates)

# Volatility by sentiment
print("\nVolatility (Standard Deviation) by Sentiment:")
volatility = merged_with_sentiment.groupby('sentiment_class')['closed_pnl'].std()
print(volatility)

# Average trade size by sentiment
print("\nAverage Trade Size by Sentiment:")
avg_trade_size = merged_with_sentiment.groupby('sentiment_class')['Size USD'].mean()
print(avg_trade_size)

# Sharpe ratio approximation (return/volatility)
print("\nReturn-to-Risk Ratio by Sentiment:")
sharpe_approx = (merged_with_sentiment.groupby('sentiment_class')['closed_pnl'].mean() / 
                merged_with_sentiment.groupby('sentiment_class')['closed_pnl'].std())
print(sharpe_approx)

# Create a comprehensive summary
print("\n=== COMPREHENSIVE SUMMARY ===")
summary = pd.DataFrame({
    'Avg_PnL': merged_with_sentiment.groupby('sentiment_class')['closed_pnl'].mean(),
    'Win_Rate': win_rates,
    'Volatility': volatility,
    'Trade_Count': merged_with_sentiment.groupby('sentiment_class')['closed_pnl'].count(),
    'Avg_Trade_Size': avg_trade_size,
    'Return_Risk_Ratio': sharpe_approx
})
print(summary.round(4))
summary.to_csv('summary_insights.csv', index=True)