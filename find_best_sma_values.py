# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 16:35:11 2020

@author: avales
"""


#FINDING BEST SMA 

import pandas as pd 
from datetime import datetime
import numpy as np
import random
from itertools import islice
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
dataset = pd.read_csv('1 month ago.csv')
dataset['Time'] = dataset['Time']/1000
dataset['Time'] = dataset['Time'].apply(datetime.fromtimestamp)

display = False
final_benchmark = []
subset_size=600
nb_subsets = 5
random_start_index = random.sample(range(0,len(dataset)-subset_size),nb_subsets)




#%%

random_SMA_50_range = random.sample(range(1,121),16)
random_SMA_10_range = random.sample(range(1,121),16)
random_SMA_3_range = random.sample(range(1,121),16)
for SMA_50_window in random_SMA_50_range:
    for SMA_10_window in random_SMA_10_range:
        for SMA_3_window  in random_SMA_3_range:
            result_list = []
            for i in range(0,nb_subsets):
                starting_test_index = 120
                
                start_index = random_start_index[i]
                end_index = start_index + subset_size
                subset = dataset.iloc[start_index:end_index,:]
                subset.reset_index(drop=True, inplace=True)
                subset['Mean']=subset[['Open','High','Low','Close']].mean(axis=1)
            
                subset['SMA_3']= subset.iloc[:,-1].rolling(window=SMA_3_window).mean()
                subset['SMA_10']= subset.iloc[:,-2].rolling(window=SMA_10_window).mean()
                subset['SMA_50']= subset.iloc[:,-3].rolling(window=SMA_50_window).mean()
                subset['signals']=0
                
                #STRATEGY
                for index, row in islice(subset.iterrows(), starting_test_index, None):
                    #test['signals'][i] = np.where(test['SMA_3'][i] > test['SMA_10'][i] > test['SMA_50'][i],1,0)
                    #print((row['SMA_3'] > row['SMA_10'] and row['SMA_3'] > row['SMA_50']))
                    subset.at[index,'signals'] = np.where((row['SMA_3'] > row['SMA_10'] and row['SMA_3'] > row['SMA_50']),1,0)
                subset['positions']=subset['signals'].diff()
                
                if display : 
                    fig, ax = plt.subplots()
                    plt.plot(subset['Time'][starting_test_index:], subset['Mean'][starting_test_index:], label='Mean',lw=2)
                    plt.plot(subset['Time'][starting_test_index:], subset['SMA_3'][starting_test_index:], label='SMA_3')
                    plt.plot(subset['Time'][starting_test_index:], subset['SMA_10'][starting_test_index:], label='SMA_10')
                    plt.plot(subset['Time'][starting_test_index:], subset['SMA_50'][starting_test_index:], label='SMA_50')
                    
                    position_to_plot = subset[['Time','Mean','positions']][starting_test_index:]
                    plt.plot(position_to_plot['Time'][subset.positions == 1],
                             position_to_plot['Mean'][subset.positions == 1],
                             '^',
                             markersize=5,
                             color = 'g',
                             label = 'buy')
                    plt.plot(position_to_plot['Time'][subset.positions == -1],
                             position_to_plot['Mean'][subset.positions == -1],
                             'v',
                             markersize=5,
                             color = 'r',
                             label = 'sell')
                    plt.legend()
                    title ='SubSet ' + str(i) + ' '+ str(SMA_3_window) +'-' + str(SMA_10_window) +'-'+ str(SMA_50_window)
                    
                    # rotate and align the tick labels so they look better
                    fig.autofmt_xdate()
                    # use a more precise date string for the x axis locations in the
                    # toolbar
                    ax.fmt_xdata = mdates.DateFormatter('%Y-%m-%d')
                    ax.set_title('')
                    plt.show()
                
                
                #ROI STRATEGY
                initial_wallet = 100.0
                quantity = 0.01
                leverage = 75
                fee_rate = 0.04/100
            
                account = pd.DataFrame(subset[['Time','Mean']][starting_test_index:])
                account.reset_index(drop=True, inplace=True)
                account['positions']=subset['positions'].fillna(0)
                account['current_trade']=0.0
                account['Wallet']=0.0
                account['total']=0.0
                account['$ Pnl']=0.0
                account['$ fees']= 0.0
                buy_mean = 0
                
                #first row initialisation
                account.at[account.head(1).index,'Wallet'] = initial_wallet
                account.at[account.head(1).index,'total'] = initial_wallet
                account.at[account.head(1).index,'positions'] = 0.0
            
                for index, row in islice(account.iterrows(), 1, None):
                    money_per_trade = account.at[index,'Mean'] * quantity / leverage
                    
                    if row['positions'] == 1 and account.at[index -1,'Wallet'] >= money_per_trade:
                        account.at[index,'Wallet'] = account.at[index -1,'Wallet'] - money_per_trade
                        account.at[index,'current_trade']= money_per_trade
                        buy_mean = account.at[index,'Mean']
                        account.at[index,'total']= account.at[index,'Wallet'] + account.at[index,'current_trade']  
                        account.at[index,'$ fees'] = quantity * account.at[index,'Mean'] * fee_rate
                       
                    elif account.at[index,'positions'] == -1 and account.at[index-1,'current_trade'] >0.0:
                        account.at[index, 'current_trade']= money_per_trade *((((account.at[index, 'Mean']/buy_mean)-1)*leverage)+1)
                        account.at[index,'Wallet'] = account.at[index -1,'Wallet'] + account.at[index,'current_trade']
                        account.at[index,'$ Pnl']=(account.at[index,'Mean'] - buy_mean)*quantity
                        account.at[index,'total']= account.at[index,'Wallet'] + account.at[index,'current_trade']
                        account.at[index,'current_trade']= 0.0
                        account.at[index,'$ fees'] = quantity * account.at[index,'Mean'] * fee_rate
                    else :
                        if account.at[index-1, 'current_trade'] != 0.0:
                            account.at[index, 'current_trade']= money_per_trade *((((account.at[index, 'Mean']/buy_mean)-1)*leverage)+1)
                            account.at[index,'$ Pnl']=(account.at[index,'Mean'] - buy_mean)*quantity
                            
                        account.at[index,'Wallet']=account.at[index-1,'Wallet']
                        account.at[index,'$ Pnl']=0.0
                        account.at[index,'total']= account.at[index,'Wallet'] + account.at[index,'current_trade']
                    
                pnl_sum = account['$ Pnl'].sum()
                fee_sum = account['$ fees'].sum()
                total = pnl_sum - fee_sum
                result_list.append(total)
            print(SMA_3_window, SMA_10_window, SMA_50_window, sum(result_list) / len(result_list))
            final_benchmark.append([SMA_3_window, SMA_10_window, SMA_50_window, sum(result_list) / len(result_list)])
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    