from binance.client import Client
from datetime import datetime
import pandas as pd
import mplfinance as mpf #https://github.com/matplotlib/mplfinance
import numpy as np
import matplotlib.pyplot as plt
api_key =""
api_secret = ""

client = Client(api_key, api_secret)


#%%
klines = client.get_historical_klines("BTCUSDT", Client.KLINE_INTERVAL_1MINUTE, "1 month ago UTC")

"""
1499040000000,      # Open time
"0.01634790",       # Open
"0.80000000",       # High
"0.01575800",       # Low
"0.01577100",       # Close
"148976.11427815",  # Volume
1499644799999,      # Close time
"2434.19055334",    # Quote asset volume
308,                # Number of trades
"1756.87402397",    # Taker buy base asset volume
"28.46694368",      # Taker buy quote asset volume
"17928899.62484339" # Can be ignored
"""




#the last 8 hour : 60mn * 8 = 480
view_window = 480

#%% GET ANALYTICS

test = pd.DataFrame(klines, columns = ['Time', 'Open','High', 'Low', 'Close','Volume','x','y','z','a','b','c'])
test.drop(['x','y','z','a','b','c'],inplace = True, axis = 1)
test[['Open','High', 'Low', 'Close','Volume']] = test[['Open','High', 'Low', 'Close','Volume']].astype(float)
test['Time'] = test['Time']/1000
test['Time'] = test['Time'].apply(datetime.fromtimestamp)

test['Mean']=test[['Open','High','Low','Close']].mean(axis=1)

test['SMA_3']= test.iloc[:,-1].rolling(window=2).mean()
test['SMA_10']= test.iloc[:,-2].rolling(window=10).mean()
test['SMA_50']= test.iloc[:,-3].rolling(window=30).mean()
test['signals']=0

#STRATEGY
for index, row in test.iterrows():
    #test['signals'][i] = np.where(test['SMA_3'][i] > test['SMA_10'][i] > test['SMA_50'][i],1,0)
    #print((row['SMA_3'] > row['SMA_10'] and row['SMA_3'] > row['SMA_50']))
    test.at[index,'signals'] = np.where((row['SMA_3'] > row['SMA_10'] and row['SMA_3'] > row['SMA_50']),1,0)
test['positions']=test['signals'].diff()

view_window = 200
# #check the 200 last points
plt.figure()
plt.plot(test['Time'][-view_window:], test['Mean'][-view_window:], label='Mean',lw=2)
plt.plot(test['Time'][-view_window:], test['SMA_3'][-view_window:], label='SMA_3')
plt.plot(test['Time'][-view_window:], test['SMA_10'][-view_window:], label='SMA_10')
plt.plot(test['Time'][-view_window:], test['SMA_50'][-view_window:], label='SMA_50')

position_to_plot = test[['Time','Mean','positions']][-view_window:]
plt.plot(position_to_plot['Time'][test.positions == 1],
         position_to_plot['Mean'][test.positions == 1],
         '^',
         markersize=5,
         color = 'g',
         label = 'buy')
plt.plot(position_to_plot['Time'][test.positions == -1],
         position_to_plot['Mean'][test.positions == -1],
         'v',
         markersize=5,
         color = 'r',
         label = 'sell')
plt.legend()
plt.show()

#ROI STRATEGY
initial_wallet = 100.0
money_per_trade = 20.0

account = pd.DataFrame(test[['Time','Mean']][-view_window:])
account['positions']=test['positions'].fillna(0)
account['current_trade']=0.0
account['Wallet']=0.0
account['total']=0.0
account['% Pnl']=0.0
buy_mean = 0
leverage = 75
#first row initialisation
account.at[account.head(1).index,'Wallet'] = initial_wallet
account.at[account.head(1).index,'total'] = initial_wallet
account.at[account.head(1).index,'positions'] = 0.0

from itertools import islice
for index, row in islice(account.iterrows(), 1, None):
    if row['positions'] == 1 and account.at[index -1,'Wallet'] >= money_per_trade:
        account.at[index,'Wallet'] = account.at[index -1,'Wallet'] - money_per_trade
        account.at[index,'current_trade']= money_per_trade
        buy_mean = account.at[index,'Mean']
        account.at[index,'total']= account.at[index,'Wallet'] + account.at[index,'current_trade']

    elif account.at[index,'positions'] == -1 and account.at[index-1,'current_trade']>0.0:
        account.at[index, 'current_trade']= money_per_trade *((((account.at[index, 'Mean']/buy_mean)-1)*leverage)+1)
        account.at[index,'Wallet'] = account.at[index -1,'Wallet'] + account.at[index,'current_trade']
        account.at[index,'% Pnl']=((account.at[index,'current_trade']/money_per_trade)-1)*leverage
        account.at[index,'total']= account.at[index,'Wallet'] + account.at[index,'current_trade']
        account.at[index,'current_trade']= 0.0
    else :
        if account.at[index-1, 'current_trade'] != 0.0:
            account.at[index, 'current_trade']= money_per_trade *((((account.at[index, 'Mean']/buy_mean)-1)*leverage)+1)

        account.at[index,'Wallet']=account.at[index-1,'Wallet']
        account.at[index,'% Pnl']=0.0
        account.at[index,'total']= account.at[index,'Wallet'] + account.at[index,'current_trade']



#%% REAL TIME
from binance.client import Client
from datetime import datetime
import pandas as pd
import mplfinance as mpf #https://github.com/matplotlib/mplfinance
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta

api_key ="aiNA7HfxZhhxEV4LIeHEv7rOTOs5RhuVhzmKVB3jo2JOdvdFQPs0Thn8wgURfBuV"
api_secret = "PL9kr8PGEvkiqDtZkQrGKWffS4eNb3mEkGlYyv5d0hfPtg3qYJkmVkhvIe2rr19h"

client = Client(api_key, api_secret)


duration = timedelta(hours=8)

leverage = 75
money_per_trade = 5
symbol = 'BTCUSDT' #min qty : 0.01

client.futures_change_leverage(symbol = symbol, leverage = leverage)
precision = client.get_symbol_info(symbol)['baseAssetPrecision']
precision = 3
#amount = money_per_trade/float(client.futures_mark_price(symbol =  symbol)['markPrice'])*leverage
#amount = "{:0.0{}f}".format(amount, precision)
amount = 0.01

current_tradeFlag = True
column_names = ['Time','Mean','SMA_3','SMA_10','SMA_50','signals','positions','current_position','Argent Fixe','Solde Net','%PnL','infos']
dfMarket = pd.DataFrame(columns = column_names)
launching_time = datetime.now()

#on s'assure de terminer le programme en ayant fini les transactions
while datetime.now() < (launching_time + duration) or  float(dfMarket.loc[dfMarket.index[-1], 'current_position']) != 0.0:
    apiMarkPrice = client.futures_mark_price(symbol =  symbol)
    time = datetime.fromtimestamp(apiMarkPrice['time']/1000).strftime("%Y-%m-%d %H:%M:%S")
    price = apiMarkPrice['markPrice']
    current_values = pd.Series([0]*len(column_names),column_names)

    dfMarket.loc[len(dfMarket)+1] = current_values

    dfMarket.loc[dfMarket.index[-1], 'Time'] = time
    dfMarket.loc[dfMarket.index[-1], 'Mean'] = price
    dfMarket['SMA_3']= dfMarket.iloc[:,1].rolling(window=20).mean()
    dfMarket['SMA_10']= dfMarket.iloc[:,1].rolling(window=100).mean()
    dfMarket['SMA_50']= dfMarket.iloc[:,1].rolling(window=300).mean()

    dfMarket.loc[dfMarket.index[-1], 'total'] = client.futures_account()['totalMarginBalance']
    dfMarket.loc[dfMarket.index[-1], 'current_position'] = client.futures_position_information()[0]['isolatedMargin']
    dfMarket.loc[dfMarket.index[-1], 'Argent Fixe']= client.futures_account()['maxWithdrawAmount']
    dfMarket.loc[dfMarket.index[-1], 'Solde Net'] = client.futures_account()['totalWalletBalance']
    try :
        dfMarket.loc[dfMarket.index[-1], '%PnL'] = (float(client.futures_position_information()[0]['markPrice'])/float(client.futures_position_information()[0]['entryPrice']) -1)* float(client.futures_position_information()[0]['leverage'])*100
    except :
        dfMarket.loc[dfMarket.index[-1], '%PnL'] =  0.0

    #check if the last row of SMA_50 is not empty
    if not (pd.isnull(dfMarket.loc[dfMarket.index[-1], 'SMA_50'])):
        #STRATEGY
        dfMarket.loc[dfMarket.index[-1], 'signals'] = np.where((dfMarket.loc[dfMarket.index[-1], 'SMA_3'] > dfMarket.loc[dfMarket.index[-1], 'SMA_10'] and dfMarket.loc[dfMarket.index[-1], 'SMA_3'] > dfMarket.loc[dfMarket.index[-1], 'SMA_50']),1,0)
        dfMarket.loc[dfMarket.index[-1], 'positions'] = dfMarket.loc[dfMarket.index[-1], 'signals'] - dfMarket.loc[dfMarket.index[-2], 'signals']
        print(float(dfMarket.loc[dfMarket.index[-1], 'current_position']) )
        print(dfMarket.loc[dfMarket.index[-1], 'positions'])
        if dfMarket.loc[dfMarket.index[-1], 'positions'] == 1 and float(dfMarket.loc[dfMarket.index[-1], 'current_position']) == 0.0:
            client.futures_create_order(symbol = symbol, side = 'BUY', type = "MARKET", quantity = float(amount))
            print('BUY')
        elif dfMarket.loc[dfMarket.index[-1], 'positions'] == -1 and float(dfMarket.loc[dfMarket.index[-1], 'positions'])!=0.0:
            client.futures_create_order(symbol = symbol, side = 'SELL', type = "MARKET", quantity = float(client.futures_position_information()[0]['positionAmt']))
            print('SELL')
        else :
            print('HOLD')

dfMarket.to_csv('tradind_result_30042020.csv', index=False))
#%% RESULT
dfMarket = pd.read_csv('tradind_result_30042020.csv', index_col = None)
try :
    dfMarket = dfMarket.drop('Unnamed: 0', 1)
except:
    pass
#if dfMarket.loc[dfMarket.index[-1], 'positions']
dfMarket = dfMarket.drop(dfMarket.index[-1], axis=0)


dfMarket['Time'] = pd.to_datetime(dfMarket['Time'])+ timedelta(hours = 1)#, "%Y-%m-%d %H:%M:%S")
launching_time = dfMarket.loc[0,'Time']


commission_sum = 0
number_of_trades = 0
fee_sum = 0
pnl_sum = 0
for trade in client.futures_account_trades():
    time = datetime.fromtimestamp(trade['time']/1000) + timedelta(hours = 1)
    if time > launching_time:
        time = time.strftime("%Y-%m-%d %H:%M:%S")
        symbol = trade['symbol']
        side = trade['side']
        price = trade["price"]
        #qty =
        fee = trade['commission']
        pnl = trade['realizedPnl']
        print(time, symbol, side, price, fee, pnl)
        fee_sum += float(fee)
        pnl_sum += float(pnl)
        number_of_trades += 1

print('total trades : ' + str(number_of_trades) , ' total fees : ' + str(fee_sum) + ' total pnl : ' + str(pnl_sum))
print('bénéfices nets : : ' + str(pnl_sum - fee_sum))


plt.figure()
plt.plot(dfMarket['Time'][:],dfMarket['Solde Net'][:].astype(float))
plt.title('Solde Net/T')
plt.show()



view_window = 1000
# #check the 200 last points
plt.figure()
plt.plot(dfMarket['Time'][-view_window:], dfMarket['Mean'][-view_window:], label='Mean',lw=2)
plt.plot(dfMarket['Time'][-view_window:], dfMarket['SMA_3'][-view_window:], label='SMA_3')
plt.plot(dfMarket['Time'][-view_window:], dfMarket['SMA_10'][-view_window:], label='SMA_10')
plt.plot(dfMarket['Time'][-view_window:], dfMarket['SMA_50'][-view_window:], label='SMA_50')

position_to_plot = dfMarket[['Time','Mean','positions']][-view_window:]
plt.plot(position_to_plot['Time'][dfMarket.positions == 1],
         position_to_plot['Mean'][dfMarket.positions == 1],
         '^',
         markersize=5,
         color = 'g',
         label = 'buy')
plt.plot(position_to_plot['Time'][dfMarket.positions == -1],
         position_to_plot['Mean'][dfMarket.positions == -1],
         'v',
         markersize=5,
         color = 'r',
         label = 'sell')
plt.legend()
plt.show()
