# Import the yfinance. If you get module not found error the run !pip install yfinance from your Jupyter notebook
import yfinance as yf

"""
# Get the data for the stock AAPL
data = yf.download('AAPL','2016-01-01','2019-08-01')



# Import the plotting library
import matplotlib.pyplot as plt

# Plot the close price of the AAPL
data['Adj Close'].plot()
plt.savefig('aapl_data.png')
plt.show()
"""
import numpy as np
import random
import pandas as pd

def read_data ():

    #tickers_list = ['AAPL', 'LCID', 'F', 'RIVN', 'BABA', 'AMD', 'NIO', 'UBER', 'LYFT','META','GOOGL', 'NVDA', 'TSLA']
    tickers_list = ['IBM']

    data_list = yf.download(tickers_list,'2024-06-01')['Adj Close']


    dates = pd.date_range('2024-06-01','2024-06-22',freq='B')
    df1=pd.DataFrame(index=dates)



    # Print first 5 rows of the data
    print(data_list.head())

if __name__ == "__main__":
    #stocks_data()
    read_data()