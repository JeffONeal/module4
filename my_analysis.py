# Initial imports -- pulling items to ensure things run smoothly throughout. 
import pandas as pd
import numpy as np
import datetime as dt
from pathlib import Path
import seaborn as sns
%matplotlib inline

# Reading whale returns -- getting data prepared to clean/run
wr_data = Path("Resources/whale_returns.csv")
wr_data = pd.read_csv(wr_data, index_col="Date", parse_dates=True, infer_datetime_format=True)
wr_data

# Count nulls for Whale Return Data
wr_data.isna().sum().sum()

# Drop nulls for Whale Return Data
wr_data =wr_data.dropna().copy()
wr_data

# Reading algorithmic returns -- getting data prepared to clean/run
algo_data = Path("Resources/algo_returns.csv")
algo_data = pd.read_csv(algo_data, index_col="Date", parse_dates=True, infer_datetime_format=True)
algo_data

# Count nulls
algo_data.isna().sum().sum()

# Drop nulls
algo_data =algo_data.dropna()
algo_data

# Reading S&P 500 Closing Prices
sp500_data = Path("Resources/sp500_history.csv")
sp500_data = pd.read_csv(sp500_data, index_col="Date", parse_dates=True, infer_datetime_format=True)
sp500_data

# Check Data Types
sp500_data.dtypes

# Cleaning Data
sp500_data["Close"] = sp500_data["Close"].str.replace("$","")
sp500_data

# Fix Data Types
sp500_data = sp500_data.astype(float)
sp500_data

# Calculate Daily Returns for S&P 500
sp500_data = sp500_data.pct_change()
sp500_data

# Drop nulls
sp500_data = sp500_data.dropna().copy()
sp500_data

# Rename `Close` Column to be specific to this portfolio.
sp500_data = sp500_data.rename(columns={"Close": "S&P500"})

sp500_data

# Join Whale Returns, Algorithmic Returns, and the S&P 500 Returns into a single DataFrame with columns for each portfolio's returns.
combined_port = pd.concat([wr_data, algo_data, sp500_data], axis='columns', join='inner')
combined_port

# Plot daily returns of all portfolios
combined_port.plot(figsize=(20,10))

# Calculate cumulative returns of all portfolios
cumulative_returns = (1 + combined_port).cumprod()
# Plot cumulative returns
cumulative_returns.plot(figsize=(20,10))

# Box plot to visually show risk
combined_port.plot.box(figsize=(20,10))

# Calculate the daily standard deviations of all portfolios
daily_std = combined_port.std()
daily_std

# Calculate  the daily standard deviation of S&P 500 (done above with combined port)

# Determine which portfolios are riskier than the S&P 500
daily_std = daily_std.sort_values(ascending=False)
daily_std

# Calculate the annualized standard deviation (252 trading days)
annualized_std = daily_std*np.sqrt(252)
annualized_std

# Calculate the rolling standard deviation for all portfolios using a 21-day window & plotting the data
combined_port.rolling(window=21).std().plot(figsize=(20,10))

# Calculate the correlation
correlation = combined_port.corr()
# Display de correlation matrix
correlation

# Calculate covariance of a single portfolio & plot it
algo1_covariance = combined_port['Algo 1'].cov(combined_port['S&P500'])
algo1_covariance
# Calculate variance of S&P 500
variance = combined_port['S&P500'].var()
variance
# Computing beta
algo1_beta = algo1_covariance / variance
algo1_beta
# Plot beta trend
rolling_algo1_cov = combined_port['Algo 1'].rolling(window=30).cov(combined_port['S&P500'])
rolling_variance = combined_port['S&P500'].rolling(window=30).var()
                                                                   
rolling_algo1_beta = rolling_algo1_cov / rolling_variance
rolling_algo1_beta.plot(figsize=(20,10))       


# Use `ewm` to calculate the rolling window
combined_port_ewm = combined_port.ewm(halflife=21).std()
combined_port_ewm.plot(figsize=(20,10))


# Annualized Sharpe Ratios

sharpe_ratios = (combined_port.mean() * 252) / (annualized_std * np.sqrt(252))
sharpe_ratios

# Visualize the sharpe ratios as a bar plot
sharpe_ratios.plot.bar(figsize=(20,10))

#According to the bar plot above both algo strategies are beating the market. 

# Reading data from 1st stock
aapl_data = Path("Resources/aapl_historical.csv")
aapl_data = pd.read_csv(aapl_data, index_col="Trade DATE", parse_dates=True, infer_datetime_format=True)
aapl_data
aapl_data = aapl_data.sort_values(by = "Trade DATE", ascending=True)
aapl_data

aapl_data = aapl_data.drop(columns=["Symbol"])
aapl_data

aapl_data = aapl_data.rename(columns={"NOCP": "AAPL"})
aapl_data

# Reading data from 2nd stock
cost_data = Path("Resources/cost_historical.csv")
cost_data = pd.read_csv(cost_data, index_col="Trade DATE", parse_dates=True, infer_datetime_format=True)
cost_data = cost_data.sort_values(by = "Trade DATE", ascending=True)
cost_data = cost_data.drop(columns=["Symbol"])
cost_data

cost_data = cost_data.rename(columns={"NOCP": "COST"})
cost_data

# Reading data from 3rd stock
goog_data = Path("Resources/goog_historical.csv")
goog_data = pd.read_csv(goog_data, index_col="Trade DATE", parse_dates=True, infer_datetime_format=True)
goog_data = goog_data.sort_values(by = "Trade DATE", ascending=True)
goog_data

goog_data = goog_data.drop(columns=["Symbol"])
goog_data

goog_data = goog_data.rename(columns={"NOCP": "GOOG"})
goog_data

# Combine all stocks in a single DataFrame
user_port = pd.concat([aapl_data, cost_data, goog_data], axis='columns', join='inner')
user_port

# Calculate daily returns
user_port1 = user_port.pct_change()
user_port1
# Drop NAs
user_port = user_port1.dropna().copy()
# Display sample data
user_port

# Set weights
weights = [.1, .3, .6]

# Calculate portfolio return
port_returns = user_port.dot(weights)
# Display sample data
port_returns

#combining my port returns to the other data
combined_port["my_port"] = port_returns
combined_port

#cleaning any NA's
combined_port.dropna(inplace=True)

# Calculate the annualized `std`
combined_port.std()

# Calculate rolling standard deviation & plot
combined_port.rolling(window=21).std().plot(figsize=(20,10))

# Calculate and plot the correlation
correlation = combined_port.corr()
correlation.plot(figsize=(20, 10))


# Calculate and plot Beta
# Calculate covariance of my portfolio
my_covariance = combined_port['my_port'].cov(combined_port['S&P500'])
my_covariance
# Calculate variance of S&P 500
variance = combined_port['S&P500'].var()
variance
# Computing beta
my_beta = my_covariance / variance
my_beta
# Plot beta trend
rolling_my_cov = combined_port['my_port'].rolling(window=30).cov(combined_port['S&P500'])
rolling_variance = combined_port['S&P500'].rolling(window=30).var()
                                                                   
rolling_my_beta = rolling_my_cov / rolling_variance
rolling_my_beta.plot(figsize=(20,10)) 

# Calculate Annualized Sharpe Ratios
daily_std1 = final_port.std()
#daily_std1
annualized_std1 = daily_std1*np.sqrt(252)
#annualized_std1
final_sharpe = (final_port.mean() * 252) / (annualized_std1 * np.sqrt(252))
final_sharpe

# Visualize the sharpe ratios as a bar plot
final_sharpe.plot.bar(figsize=(20,10))

#My portfolio outperformed the market and most of the other portfolios.