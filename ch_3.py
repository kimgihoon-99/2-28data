#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 14:59:45 2025

@author: gim-yebin
"""




#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 14:50:00 2025

@author: gim-yebin
"""

'''
3. 시간이 지남에 따라 다양한 장소 간을 비행하는 사람의 수는 어떻게 변하나?

'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

path = '/Users/gim-yebin/Downloads/merged_flight_data.csv'
df = pd.read_csv(path, encoding='latin-1', low_memory=False)

df = df[['Year', 'Month', 'FlightNum', 'Origin', 'Dest']]

df.isnull().sum()

'''
Year         0
Month        0
FlightNum    0
Origin       0
Dest         0
dtype: int64
'''

df.describe()

df["Year"].value_counts().sort_index()

df = df[(df["Year"] > 1987) & (df["Year"] < 2008)]  



'''
            Year         Month     FlightNum
count  1.189145e+08  1.189145e+08  1.189145e+08
mean   1.998260e+03  6.483093e+00  1.333094e+03
std    6.060609e+00  3.462478e+00  1.368317e+03
min    1.987000e+03  1.000000e+00  1.000000e+00
25%    1.993000e+03  3.000000e+00  4.470000e+02
50%    1.999000e+03  6.000000e+00  9.310000e+02
75%    2.004000e+03  1.000000e+01  1.685000e+03
max    2.008000e+03  1.200000e+01  9.912000e+03
'''


Q1 = df["FlightNum"].quantile(0.25)  
Q3 = df["FlightNum"].quantile(0.75)  
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df = df[(df["FlightNum"] >= lower_bound) & (df["FlightNum"] <= upper_bound)]

df.describe()

flights_by_year = df.groupby("Year").size()


plt.figure(figsize=(12, 6))
sns.lineplot(x=flights_by_year.index, y=flights_by_year.values, marker="o", color="blue")

plt.title("연도별 항공편 운항 횟수 변화")
plt.xlabel("연도")
plt.ylabel("운항 횟수")

plt.xticks(ticks=flights_by_year.index, labels=flights_by_year.index)
plt.ticklabel_format(style='plain', axis='y')

plt.show()


