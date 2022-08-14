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
latest_url = 'https://prices.runescape.wiki/api/v1/osrs/latest'
hour_url = 'https://prices.runescape.wiki/api/v1/osrs/1h'

getlatest = requests.get(latest_url, headers = headers)
getlatest = getlatest.text
getlatest = re.sub('":{"high', ',', getlatest)
getlatest = re.sub('},', ',\n', getlatest)
getlatest = re.sub('[^0-9,\n]', '', getlatest)

gethour = requests.get(hour_url, headers = headers)
gethour = gethour.text
gethour = re.sub('":{"avgHigh', ',', gethour)
gethour = re.sub('},', ',\n', gethour)
gethour = re.sub('[^0-9,\n]', '', gethour)

store_out = sys.stdout

with open('latest_prices.txt', 'w') as f:
    sys.stdout = f
    print(getlatest)
    sys.stdout = store_out

with open('one_hour_market_data.txt', 'w') as f:
    sys.stdout = f
    print(gethour)
    sys.stdout = store_out

hour_df = pd.read_csv('one_hour_market_data.txt', names = 
                     ['id', 'avg_high', 'high_vol', 'avg_low', 'low_vol', 'delete'])
hour_df = hour_df.drop(columns = ['delete'])
latest_df = pd.read_csv('latest_prices.txt', names = 
                        ['id', 'high', 'high_time', 'low', 'low_time', 'delete'])
latest_df = latest_df.drop(columns = ['high_time', 'low_time', 'delete'])

latest_df = latest_df.merge(hour_df, on = ['id'])
latest_id = latest_df['id']

#avgspread_array = np.empty((0, 0))
#avgvol_array = np.empty((0, 0))

#for i in latest_id:
    #volume_url = 'https://prices.runescape.wiki/api/v1/osrs/timeseries?timestep=1h&id=' + str(i)
    #getvolume = requests.get(volume_url, headers = headers)
    #getvolume = getvolume.text
    #gethour = re.sub('":{"timestamp', ',', getvolume)
    #gethour = re.sub('},', ',\n', getvolume)
    #gethour = re.sub('[^0-9,\n]', '', getvolume)
    #with open('store_series.txt', 'w') as f:
#        sys.stdout = f
#        print(gethour)
#         sys.stdout = store_out
#    timeseries_df = pd.read_csv('one_hour_market_data.txt', names = 
#                        ['id', 'avg_high', 'high_vol', 'avg_low', 'low_vol', 'delete'])
#    timeseries_array = timeseries_df.to_numpy()
#    high_array = timeseries_array[0:, 1]
#    highvol_array = timeseries_array[0:, 2]
#    low_array = timeseries_array[0:, 3]
#    lowvol_array = timeseries_array[0:, 4]
#    spread_array = high_array - low_array

#    highvol_ave = np.mean(high_array)
#    lowvol_ave = np.mean(low_array)
#    vol_ave = np.minimum(highvol_ave, lowvol_ave)
#    spread_ave = np.mean(spread_array)
#    avgspread_array = np.append(avgspread_array, spread_ave)
#    avgvol_array = np.append(avgvol_array, vol_ave)

      
#avgspread_array = np.transpose(avgspread_array)
#avgvol_array = np.transpose(avgvol_array)
        
#print (avgspread_array)
#print (avgvol_array)


latest_df['Spread'] = latest_df['high'] - latest_df['low']
latest_df['ROI'] = latest_df['Spread'] / latest_df['low']

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


model = LpProblem(name = 'profit-maximization', sense = LpMaximize)
DV_variablesX = LpVariable.matrix('X', temp_id, cat = 'Integer', lowBound = 0)
DV_variablesY = LpVariable.matrix('Y', temp_id, cat = 'Binary', lowBound = 0)

obj_func = lpSum(DV_variablesX * hour_spread)
model += obj_func   

avalcap = np.array([26000000])
maxTrades = np.array([4])

for i in range(temp_bl.size):
    model += DV_variablesX[i] <= (temp_bl[i] * DV_variablesY[i]), str(i) + ' Quantity Constraint'

for i in range(temp_bl.size):
    model += DV_variablesX[i] <= (np.minimum(temp_low_vol[i], temp_high_vol[i]) * DV_variablesY[i]), str(i) + ' Hourly Available Quantity'

model += lpSum((DV_variablesX[i] * hour_low[i]) for i in range(temp_bl.size)) <= avalcap, 'Capital Constraint'
model += lpSum((DV_variablesY[i]) for i in range(temp_bl.size)) <= maxTrades, 'Trade Number Constraint'
    
model.writeMPS("model.mps")
model.solve()

status = LpStatus[model.status]

print(status)

solution = np.empty((1, 2))

for v in model.variables():
    if(v.value() > 0):
        v.name = re.sub('X_', '', v.name)
        v.name = re.sub('Y_', '', v.name)
        solution = np.vstack((solution, [int(v.name), v.value()]))

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

solution_df.to_csv('solution.csv')
print(solution_df)