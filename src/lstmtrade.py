import numpy as np
import random
import pandas as pd
#from matplotlib import plt
from pylab import mpl, plt
plt.style.use('seaborn-v0_8-whitegrid')
mpl.rcParams['font.family'] = 'serif' 


import datetime
import math, time
import itertools
import datetime
from operator import itemgetter
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from math import sqrt
import torch
import torch.nn as nn
from torch.autograd import Variable

import os

def load_files():
    for dirname, _, filenames in os.walk('/u/djy8hg/Project/arupcsedu/pytrade/content'):
        for i, filename in enumerate(filenames):
            if i<5:
                print(os.path.join(dirname,filename))

def stocks_data(symbols, dates):
    df = pd.DataFrame(index=dates)
    for symbol in symbols:
        df_temp = pd.read_csv("/u/djy8hg/Project/arupcsedu/pytrade/content/Data/Stocks/{}.us.txt".format(symbol), index_col='Date',
                parse_dates=True, usecols=['Date', 'Close'], na_values=['nan'])
        df_temp = df_temp.rename(columns={'Close': symbol})
        df = df.join(df_temp)
    return df

def read_data():
    dates = pd.date_range('2015-01-02','2016-12-31',freq='B')
    symbols = ['goog','ibm','aapl']
    df = stocks_data(symbols, dates)
    df.fillna(method='pad')
    #df.plot(figsize=(10, 6), subplots=True)

def plot_data():
    dates = pd.date_range('2010-01-02','2017-10-11',freq='B')
    df1=pd.DataFrame(index=dates)
    df_ibm=pd.read_csv("/u/djy8hg/Project/arupcsedu/pytrade/content/Data/Stocks/ibm.us.txt", parse_dates=True, index_col=0)
    df_ibm=df1.join(df_ibm)
    df_ibm[['Close']].plot(figsize=(15, 6))
    plt.ylabel("stock_price")
    plt.title("IBM Stock")
    plt.savefig('ibm_data.png')
    plt.show()
    

    df_ibm=df_ibm[['Close']]
    df_ibm.info()



# function to create train, test data given stock data and sequence length
def load_data(stock, look_back):
    data_raw = stock.values # convert to numpy array
    data = []

    # create all possible sequences of length look_back
    for index in range(len(data_raw) - look_back):
        data.append(data_raw[index: index + look_back])

    data = np.array(data);
    test_set_size = int(np.round(0.2*data.shape[0]));
    train_set_size = data.shape[0] - (test_set_size);

    x_train = data[:train_set_size,:-1,:]
    y_train = data[:train_set_size,-1,:]

    x_test = data[train_set_size:,:-1]
    y_test = data[train_set_size:,-1,:]

    return [x_train, y_train, x_test, y_test]

def scale_data():
    dates = pd.date_range('2010-01-02','2017-10-11',freq='B')
    df1=pd.DataFrame(index=dates)
    df_ibm=pd.read_csv("/u/djy8hg/Project/arupcsedu/pytrade/content/Data/Stocks/ibm.us.txt", parse_dates=True, index_col=0)
    df_ibm=df1.join(df_ibm)
    df_ibm=df_ibm[['Close']]


    
    df_ibm=df_ibm.fillna(method='ffill')

    scaler = MinMaxScaler(feature_range=(-1, 1))
    df_ibm['Close'] = scaler.fit_transform(df_ibm['Close'].values.reshape(-1,1))

    print(df_ibm.head())
    return [df_ibm, scaler] 

def load_data_call():
 
    df_ibm, scaler = scale_data()
    look_back = 60 # choose sequence length
    x_train, y_train, x_test, y_test = load_data(df_ibm, look_back)
    print('x_train.shape = ',x_train.shape)
    print('y_train.shape = ',y_train.shape)
    print('x_test.shape = ',x_test.shape)
    print('y_test.shape = ',y_test.shape)

    # make training and test sets in torch
    x_train = torch.from_numpy(x_train).type(torch.Tensor)
    x_test = torch.from_numpy(x_test).type(torch.Tensor)
    y_train = torch.from_numpy(y_train).type(torch.Tensor)
    y_test = torch.from_numpy(y_test).type(torch.Tensor)
    y_train.size(),x_train.size()

    return [x_train, y_train, x_test, y_test, df_ibm, scaler]

# Build model
#####################
input_dim = 1
hidden_dim = 32
num_layers = 2
output_dim = 1


# Here we define our model as a class
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.num_layers = num_layers
  # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()

        # Initialize cell state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()

        # We need to detach as we are doing truncated backpropagation through time (BPTT)
               # If we don't, we'll backprop all the way to the start even after going through another batch
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # Index hidden state of last time step
        # out.size() --> 100, 32, 100
        # out[:, -1, :] --> 100, 100 --> just want last time step hidden states!
        out = self.fc(out[:, -1, :])
        # out.size() --> 100, 10
        return out



def train_load():

    model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)

    loss_fn = torch.nn.MSELoss()

    optimiser = torch.optim.Adam(model.parameters(), lr=0.01)
    print(model)
    print(len(list(model.parameters())))
    for i in range(len(list(model.parameters()))):
        print(list(model.parameters())[i].size())

    # Train model
    #####################
    look_back = 60 # choose sequence length
    num_epochs = 100
    hist = np.zeros(num_epochs)

    # Number of steps to unroll
    seq_dim =look_back-1

    x_train, y_train, x_test, y_test, df_ibm, scaler = load_data_call()

    for t in range(num_epochs):
        # Initialise hidden state
        # Don't do this if you want your LSTM to be stateful
        #model.hidden = model.init_hidden()

        # Forward pass
        y_train_pred = model(x_train)

        loss = loss_fn(y_train_pred, y_train)
        if t % 10 == 0 and t !=0:
            print("Epoch ", t, "MSE: ", loss.item())
        hist[t] = loss.item()

        # Zero out gradient, else they will accumulate between epochs
        optimiser.zero_grad()

        # Backward pass
        loss.backward()

        # Update parameters
        optimiser.step()
    
    np.shape(y_train_pred)
    # make predictions
    y_test_pred = model(x_test)

    # invert predictions
    y_train_pred = scaler.inverse_transform(y_train_pred.detach().numpy())
    y_train = scaler.inverse_transform(y_train.detach().numpy())
    y_test_pred = scaler.inverse_transform(y_test_pred.detach().numpy())
    y_test = scaler.inverse_transform(y_test.detach().numpy())

    # calculate root mean squared error
    trainScore = math.sqrt(mean_squared_error(y_train[:,0], y_train_pred[:,0]))
    print('Train Score: %.2f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(y_test[:,0], y_test_pred[:,0]))
    print('Test Score: %.2f RMSE' % (testScore))


    # Visualising the results
    figure, axes = plt.subplots(figsize=(15, 6))
    axes.xaxis_date()

    axes.plot(df_ibm[len(df_ibm)-len(y_test):].index, y_test, color = 'red', label = 'Real IBM Stock Price')
    axes.plot(df_ibm[len(df_ibm)-len(y_test):].index, y_test_pred, color = 'blue', label = 'Predicted IBM Stock Price')
    #axes.xticks(np.arange(0,394,50))
    plt.title('IBM Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('IBM Stock Price')
    plt.legend()
    plt.savefig('ibm_pred.png')
    plt.show()

if __name__ == "__main__":
    load_files()
    #stocks_data()
    read_data()
    #plot_data()
    train_load()


