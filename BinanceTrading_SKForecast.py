# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 09:30:28 2023

@author: aidan
"""

# Advanced attempt at making a Binance bot

#______________________________________________________________________________
# import libraries...
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from binance.client import Client
import matplotlib.dates as dates
from scipy.signal import savgol_filter


#______________________________________________________________________________
# turn off annoying warnings
import warnings 
pd.options.mode.chained_assignment = None  # default='warn'
    
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

#______________________________________________________________________________
# connect to binance
api_key = '' # ENTER YOUR BINANCE API KEY
api_secret = '' # ENTER YOUR BINANCE SECRET KEY
client = Client(api_key, api_secret) # connect to Binance API

#______________________________________________________________________________
# define technical indicators
def calculate_indicators(df,close_smoothed_period):
    
    # smooth the time-series using Savitsky-Golay filter
    df['close_smoothed'] = savgol_filter(df['close'], window_length=20,polyorder=4)
    
    # calculate first and second derivatives of smoothed time-series
    slopeb = pd.Series(np.gradient(df['close_smoothed'],edge_order=2), df['close_smoothed'].index, name='slopeb')
    slope2b = pd.Series(np.gradient(slopeb,edge_order=2), df['close_smoothed'].index, name='slope2b')
    
    # add gradients to dataframe for use as technical indicators
    df['slopeb'] = slopeb
    df['slope2b'] = slope2b
    
    return df
    
#______________________________________________________________________________
# define machine learning prediction function

# goal is to predict time-series two periods ahead, to inform a decision

def ML_predictions(df,num_future_periods,interval):
    
    # import forecasting library
    from skforecast.ForecasterAutoreg import ForecasterAutoreg
    from skforecast.model_selection import random_search_forecaster
    from xgboost import XGBRegressor
    from datetime import timedelta
    from skforecast.model_selection import backtesting_forecaster
    
    # initialise forecasting model - use Extreme Gradient Boosting
    forecaster = ForecasterAutoreg(
                regressor = XGBRegressor(tree_method = 'hist',
                                 enable_categorical = 'auto',
                                 random_state = 123),
                lags      = 10)


    train_size = len(df) # train on all available data 
    
    # create training dataset
    df_train = pd.Series(df['close_smoothed'].iloc[-train_size:],index=df.index[-train_size:])
    
    # print most recent 10 periods from training dataset
    print(df_train[-10:])
    
    # backtest forecaster on available data
    backtesting_metric, predictions_backtest = backtesting_forecaster(
                                   forecaster         = forecaster,
                                   y                  = df_train,
                                   initial_train_size = round(train_size/2),
                                   fixed_train_size   = False,
                                   steps              = 2,
                                   metric             = 'mean_absolute_percentage_error',
                                   refit              = False,
                                   verbose            = False
                               )
    
    # output backtesting performance
    print('Backtesting metric:\n')
    print(backtesting_metric)
    
    # Lags used as predictors
    lags_grid = [1,2,3,5,8,10]
    
    # Regressor hyperparameters
    param_distributions = {'n_estimators': np.arange(start=250, stop=1000, step=250, dtype=int)}#,
                           #'max_depth': np.arange(start=5, stop=6, step=1, dtype=int)}
    
    # tune hyperparameters using random search over hyperparameter space
    hyper_fit_results = random_search_forecaster(
                  forecaster           = forecaster,
                  y                    = df_train,
                  steps                = 2,
                  lags_grid            = lags_grid,
                  param_distributions  = param_distributions,
                  n_iter               = 5,
                  metric               = 'mean_absolute_percentage_error',
                  refit                = True,
                  initial_train_size   = round(train_size/2),
                  fixed_train_size     = False,
                  return_best          = True,
                  random_state         = 123,
                  verbose              = False
                  )
    
    # output results of hyperparameter tuning
    print('Hyperparameter variation results:\n')
    print(hyper_fit_results)
    
    # train model using tuned hyperparameters
    forecaster.fit(y=df['close_smoothed'])
    forecaster.get_feature_importance()
    
    # generate predictions
    y_pred = forecaster.predict(steps=num_future_periods)    
    y_pred = y_pred.tolist()
    
    # generate datetimes for future predictions
    final_dti = df.index[len(df)-1]
    
    if (interval[-1] == 'm'):
        dti_list = [final_dti + timedelta(minutes=x+1) for x in range(num_future_periods)]
    
    elif (interval[-1] == 'h'):
        dti_list = [final_dti + timedelta(hours=x+1) for x in range(num_future_periods)]
    
    if (interval[-1] == 'd'):
        dti_list = [final_dti + timedelta(days=x+1) for x in range(num_future_periods)]
        
    y_pred = dict(zip(dti_list,y_pred))
    
    # convert series to dataframe
    y_pred = pd.DataFrame.from_dict(y_pred,orient='index',columns=['close_smoothed'])
    
    #backtest optimised forecaster and return metric    
    backtesting_metric_opt, predictions_backtest_opt = backtesting_forecaster(
                                   forecaster         = forecaster,
                                   y                  = df_train,
                                   initial_train_size = round(train_size/2),
                                   fixed_train_size   = False,
                                   steps              = 2,
                                   metric             = 'mean_squared_error',
                                   refit              = False,
                                   verbose            = False
                               )
    
    print('Optimised backtesting metric:\n')
    print(backtesting_metric_opt)
    
    return y_pred
    
#______________________________________________________________________________
# define strategy
def strategy(df):
    
    # require decreasing price (-ve gradient) and second derivative to be +ve 
    # (indicating that the price is about to increase) in order to buy
    buy = df['slopeb'] < 0
    buy &= df['slope2b'] > 0 # df['slope2b'].shift(-1)
    buy &= df['close_smoothed'].shift(-1) < df['close_smoothed'].shift(-2)
    
    # require increasing price (+ve gradient) and second derivative to be -ve 
    # (indicating that the price is about to decrease) in order to sell
    sell = df['slopeb'] > 0
    sell &= df['slope2b'] < 0 #df['slope2b'].shift(-1)
    sell &= df['close_smoothed'].shift(-1) > df['close_smoothed'].shift(-2)
    
    # print most recent 5 decisions
    print(buy.tail(5))
    print(df.tail(5))
    
    # if first sell occurs before first buy, set to False (error trap)
    true_buys = buy[buy == True] # only initially true buys
    sell.loc[sell.index < true_buys.index[0]] = False

    # add buy and sell signals to dataframe as a boolean value
    df['buy'] = buy
    df['sell'] = sell
    
    return df

#______________________________________________________________________________
# get historical backtesting data from specified start time until now
def download_data(client,symbol, interval, start_str):
    
    print(f'Downloading data for {symbol}. Interval {interval}. Starting from {start_str}')
    klines = client.get_historical_klines(symbol, interval, start_str)
    
    # downselect parameters of interest
    data = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
    
    # extract timestamp for each period in backtesting dataset
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
    
    # set datetimes as index
    data.set_index('timestamp', inplace=True)
    
    # add close prices for each period to dataframe
    data['close'] = data['close'].astype(float)
    
    return data

#______________________________________________________________________________
# plot cumulative returns

def plot_results(data):
    data[['cumulative_returns']].plot(figsize=(10, 6))
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.title('Backtesting Results')
    plt.show()

#______________________________________________________________________________
# plot time series data
def plot_ts_and_ta(data):
    
    axs = data[['close']].plot(figsize=(10, 6))
    data[['close_smoothed']].plot(ax = axs)
    fig = axs.get_figure()
    fig.savefig('TestTAGraph.png')
    plt.close()

#______________________________________________________________________________
# LIVE TEST MODE!
# - inbetween backtest and live trading, intended to process realtime data 

# this WILL place trades unless the order command is commented out

# create figure and axes for plotting
fig,axs = plt.subplots(2,1,figsize=(20, 12))

def live_trading_test():

    import time
    
    symbol = 'BTCUSDT' # crypto pait to be traded
    base_asset = 'USDT' # base asset
    spec_asset = 'BTC' # asset to be maximised
    interval = '5m' # time length of each period
    start_date = '08 Jun 2023 00:00:00' # start date for backtesting/training data
    
    # get current price of trading asset in terms of base asset
    current_price = client.get_symbol_ticker(symbol=symbol)
    current_price = float(current_price['price'])
    print('Current price is 1' + spec_asset + ' = ' + str(current_price) + base_asset)
    
    # get your account balance for base asset
    buy_balance = client.get_asset_balance(asset=base_asset)
    buy_balance = float(buy_balance['free'])
    
    # get your account balance for trading asset
    sell_balance = client.get_asset_balance(asset=spec_asset)
    sell_balance = float(sell_balance['free'])
    
    # both prev sell and prev buy start as false
    last_bought = False
    last_sold = True
    
    # only sell after actually having bought some!
    start_flag = True
    
    # empty lists preallocated for predicted values
    pred_dates = []
    pred_close_smoothed = []
    
    while True:
        
        # download historical data for training
        data = download_data(client,symbol, interval, start_date)
        
        # parameters for technical indicators
        close_smoothed_period = 10
        
        # calculate indicators for historical data
        calculate_indicators(data,close_smoothed_period)
        data.dropna(inplace=True)
        
        # number of future periods to predict
        num_future_periods = 2 
        
        # use ML model to predict data for future values   
        data_pred = ML_predictions(data,num_future_periods,interval)
        
        # combine past (training) and future (predicted) dataframes for this period
        data_temp = data.copy()
        data_temp = data[['close_smoothed']]
        data_combined = pd.concat([data_temp, data_pred], axis=0)
        data_combined.dropna(inplace=True)
        
        # print last few close prices plus the predictions
        print(data_combined.iloc[-10:])
        
        # store last predicted smoothed close price and associated datetime
        pred_dates.append(data_combined.index[-1])
        pred_close_smoothed.append(data_combined['close_smoothed'].iloc[-1])
        
        # get gradient of smoothed close price, including prediction
        slopeb = pd.Series(np.gradient(data_combined['close_smoothed'],edge_order=2), data_combined['close_smoothed'].index, name='slopeb')
        data_combined['slopeb'] = slopeb
        
        # get second derivative of smoothed close price, including prediction
        data_combined['slope2b'] = pd.Series(np.gradient(slopeb,edge_order=2), data_combined['close_smoothed'].index, name='slope2b')
        
        # get current price of trading asset in units of base asset
        current_price = client.get_symbol_ticker(symbol=symbol)
        current_price = float(current_price['price'])
        print('Current price is ' + str(current_price) + base_asset)
        
        # get current base asset balance
        buy_balance = client.get_asset_balance(asset=base_asset)
        buy_balance = float(buy_balance['free'])
        
        # get current trading asset balance
        sell_balance = client.get_asset_balance(asset=spec_asset)
        sell_balance = float(sell_balance['free'])
        
        # plot out the data and smoothed close price
        data[['close']].plot(ax=axs[0])
        data_combined[['close_smoothed']].plot(ax = axs[0])
        
        # pass data + ML predictions to strategy to make buy/sell decision
        strategy(data_combined)
        
        # get final element (catch for 1xN output)
        buy_signal = data_combined['buy'].iloc[-num_future_periods-1]
        sell_signal = data_combined['sell'].iloc[-num_future_periods-1]
        
        # print buy/sell decision
        print('This buy decision... ' + str(buy_signal) )
        print('This sell decision... ' + str(sell_signal) )
        
        # if buy (and not prev buy), place buy order
        if buy_signal and last_sold and not last_bought:

            # get balance
            buy_balance = client.get_asset_balance(asset=base_asset)
            buy_balance = float(buy_balance['free'])
            
            # get current price
            current_price = client.get_symbol_ticker(symbol=symbol)
            current_price = float(current_price['price'])
            
            # quantity to buy (leave 0.7% for fees)
            buy_quantity = round(0.993 * buy_balance/current_price,3)
            
            # place order
            order = client.create_order(
                symbol=symbol,
                side='BUY',
                type='MARKET',
                quantity=buy_quantity)
            
            # update boolean values
            last_bought = True
            last_sold = False
            
            # inform user
            print('Buy signal generated. Placing market buy order.')
            
            # update after first buy
            start_flag = False
            
        # if sell (and not prev sell), place sell order
        elif sell_signal and not start_flag and last_bought and not last_sold:
            
            # get current balance
            sell_balance = client.get_asset_balance(asset=spec_asset)
            sell_balance = float(sell_balance['free'])
            
            # quantity to buy (leave 0.7% for fees)
            sell_quantity = round(0.993 * sell_balance,3)
            
            # place order
            order = client.create_order(
                symbol=symbol,
                side='SELL',
                type='MARKET',
                quantity=sell_quantity)
            
            # update boolean values
            last_bought = False
            last_sold = True
            
            # inform user
            print('Sell signal generated. Placing market sell order.')
        
        # plot balance
        print('Current balance is ' + str(buy_balance + sell_balance*current_price) + base_asset)
        
        # calculate balance after placing any trades
        balance_after_trades = buy_balance + sell_balance*current_price
        
        # plot predictions
        axs[0].scatter(pred_dates,pred_close_smoothed,marker = '*',s=70,color='blue') 
        
        # plot buy and sell closes
        buy_closes = data_combined[(data_combined['buy'] == True)] 
        sell_closes = data_combined[(data_combined['sell'] == True)] 
        
        axs[0].scatter(buy_closes.index,buy_closes['close_smoothed'],marker = 'x',s=70,color='black')
        axs[0].scatter(sell_closes.index,sell_closes['close_smoothed'],marker = 'v',s=70,color='red')
        
        # remove legend
        axs[0].get_legend().remove()
        
        # plot balance time series in base currency
        axs[1].scatter(data.index[-1],balance_after_trades,marker='.')
        plt.xlim([data.index[0],data.index[-1]])
        axs[1].xaxis.set_major_formatter(dates.DateFormatter('%H:%M'))
        fig.savefig('TestTAGraph.png')
        
        axs[0].clear()
        
        # wait one minute and repeat
        print('Waiting...')
        time.sleep(60)
        
#______________________________________________________________________________
# run the test
live_trading_test()











