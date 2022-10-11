import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linprog
from pulp import *
import requests
import re
import sys
import cvxopt

recLim = 3500
sys.setrecursionlimit(recLim)

headers = {
    'User-Agent': 'market-tool'
    }
# api url paths
latest_url = 'https://prices.runescape.wiki/api/v1/osrs/latest'
hour_url = 'https://prices.runescape.wiki/api/v1/osrs/1h'

# pull most recent data
getlatest = requests.get(latest_url, headers = headers)
# csv formatting
getlatest = getlatest.text
getlatest = re.sub('":{"high', ',', getlatest)
getlatest = re.sub('},', ',\n', getlatest)
getlatest = re.sub('[^0-9,\n]', '', getlatest)

# pull latest 1 hour average data
gethour = requests.get(hour_url, headers = headers)
# csv formatting
gethour = gethour.text
gethour = re.sub('":{"avgHigh', ',', gethour)
gethour = re.sub('},', ',\n', gethour)
gethour = re.sub('[^0-9,\n]', '', gethour)

store_out = sys.stdout

# write formatted data to files
with open('latest_prices.txt', 'w') as f:
    sys.stdout = f
    print(getlatest)
    sys.stdout = store_out
    
with open('one_hour_market_data.txt', 'w') as f:
    sys.stdout = f
    print(gethour)
    sys.stdout = store_out

# make pandas dataframes
hour_df = pd.read_csv('one_hour_market_data.txt', names = 
                     ['id', 'avg_high', 'high_vol', 'avg_low', 'low_vol', 'delete'])
hour_df = hour_df.drop(columns = ['delete'])
latest_df = pd.read_csv('latest_prices.txt', names = 
                        ['id', 'high', 'high_time', 'low', 'low_time', 'delete'])
latest_df = latest_df.drop(columns = ['high_time', 'low_time', 'delete'])

latest_df = latest_df.merge(hour_df, on = ['id'])
latest_id = latest_df['id']

latest_df['Spread'] = latest_df['high'] - latest_df['low']
latest_df['ROI'] = latest_df['Spread'] / latest_df['low']

# read in (mostly) static buy limit data to dataframe
buylim_df = pd.read_json('items-buylimits.json', typ = 'series')
buylim_df.name = 'buy_limit'
itemname_df = pd.read_json('items-summary.json')
itemname_df = itemname_df.transpose()
namemerge = itemname_df.set_index('name')
item_df = namemerge.merge(buylim_df, left_index = True, right_index = True)
item_df = item_df.set_index('id')
item_df = itemname_df.merge(item_df, left_on = ['id'], right_index = True)

latest_df = latest_df.merge(item_df, on = ['id'])
latest_df = latest_df.fillna(0)

temp_array = latest_df.to_numpy()

# calculate important price info
hour_high = temp_array[0:, 3]
hour_low = temp_array[0:, 5]
hour_spread = np.subtract(hour_high, hour_low)

temp_id = temp_array[0:, 0]
temp_low = temp_array[0:, 2]
temp_high_vol = temp_array[0:, 4]
temp_low_vol = temp_array[0:, 6]
temp_spread = temp_array[0:, 1] - temp_low
temp_bl = temp_array[0:, 10]

temp_id = np.int64(temp_id)
temp_bl = np.int64(temp_bl)
temp_low = np.int64(temp_low)
temp_high_vol = np.int64(temp_high_vol)
temp_low_vol = np.int64(temp_low_vol)

# create linear programming model
model = LpProblem(name = 'profit-maximization', sense = LpMaximize)
DV_variablesX = LpVariable.matrix('X', temp_id, cat = 'Integer', lowBound = 0)
DV_variablesY = LpVariable.matrix('Y', temp_id, cat = 'Binary', lowBound = 0)

# maximize for quantity * spread
obj_func = lpSum(DV_variablesX * hour_spread)
model += obj_func   

# parameter input
# avalcap: amount of gp available
# maxTrades: number of trade windows available
print("Enter gp available to trade (0-2147483647): ")
capital = int(input())
print("Enter number of windows available (0-8): ")
windows = int(input())
avalcap = np.array([capital])
maxTrades = np.array([windows])

# add constraints to model
model += lpSum((DV_variablesX[i] * hour_low[i]) for i in range(temp_bl.size)) <= avalcap, 'Capital Constraint'
model += lpSum((DV_variablesY[i]) for i in range(temp_bl.size)) <= maxTrades, 'Trade Number Constraint'

# set quantity contraints equal to min(recently traded quantity, buy limit)
for i in range(temp_bl.size):
    model += DV_variablesX[i] <= (temp_bl[i] * DV_variablesY[i]), str(i) + ' Quantity Constraint'

for i in range(temp_bl.size):
    model += DV_variablesX[i] <= (np.minimum(temp_low_vol[i], temp_high_vol[i]) * DV_variablesY[i]), str(i) + ' Hourly Available Quantity'

# run solution
model.writeMPS("model.mps")
model.solve()

status = LpStatus[model.status]

# output result overview
print(status)

solution = np.empty((1, 2))

# output model results into solution array
for v in model.variables():
    if(v.value() > 0):
        v.name = re.sub('X_', '', v.name)
        v.name = re.sub('Y_', '', v.name)
        solution = np.vstack((solution, [int(v.name), v.value()]))

# format solution data
solution = np.delete(solution, 0, axis = 0)   
solution = np.delete(solution, np.s_[maxTrades[0] : (maxTrades[0] * 2)], axis = 0) 
solution_df = pd.DataFrame(solution, columns = ['id', 'Quantity'])
solution_df = pd.merge(solution_df, latest_df, on = ['id'])
solution_df['Average Spread (1H)'] = (solution_df['avg_high'] - solution_df['avg_low'])
solution_df['Average Profit (1H)'] = solution_df['Average Spread (1H)'] * solution_df['Quantity'] 
solution_df['Current Profit'] = solution_df['Spread'] * solution_df['Quantity']
solution_df['Volume (1H)'] = solution_df[['high_vol', 'low_vol']].min(axis = 1)
solution_df = solution_df.drop(columns = ['id', 'high', 'avg_high', 'avg_low', 'high_vol', 'low_vol', 'buy_limit'])
solution_df = solution_df.rename(columns = {'low': 'Unit Cost', 'name': 'Name', 'Spread': 'Current Spread', 'ROI': 'Current ROI'})

temp_4h_profit = np.minimum(temp_high_vol, temp_low_vol) * temp_spread * 4
temp_max_profit = temp_spread * temp_bl
temp_bl_profit = np.minimum(temp_4h_profit, temp_max_profit)

# write solution data to csv
solution_df.to_csv('solution.csv')
print(solution_df)
