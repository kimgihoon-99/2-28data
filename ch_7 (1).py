#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 14:28:13 2025

@author: gim-yebin
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 12:15:05 2025

@author: gim-yebin
"""

'''
7. 가장 자주 비행하는 두 도시 간을 오가는 모든 항공편을 비교
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 전처리
path = '/Users/gim-yebin/Downloads/merged_flight_data.csv'
df = pd.read_csv(path, encoding='latin-1', low_memory=False)

df[['Origin', 'Dest', 'FlightNum', 'Year']].isnull().sum()
'''
Origin       0
Dest         0
FlightNum    0
Year         0
dtype: int64

결측값 없음
'''

df = df[(df["Year"] > 1987) & (df["Year"] < 2008)]  

df.describe()

Q1 = df["FlightNum"].quantile(0.25)  
Q3 = df["FlightNum"].quantile(0.75)  
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df = df[(df["FlightNum"] >= lower_bound) & (df["FlightNum"] <= upper_bound)]

df.describe()


top_route = df.groupby(["Origin", "Dest"]).size().reset_index(name="FlightCount")

top_route = top_route.sort_values(by="FlightCount", ascending=False).head(1)

# 출발 공항과 도착 공항 저장
top_origin, top_dest = top_route.iloc[0][["Origin", "Dest"]]

print(f"가장 자주 운항된 노선: {top_origin} -> {top_dest} ({top_route.iloc[0]['FlightCount']}회)")

'''
가장 자주 운항된 노선: SFO -> LAX (329370회)
'''

top_route_flights = df[(df["Origin"] == top_origin) & (df["Dest"] == top_dest)]
flights_by_year = top_route_flights.groupby("Year").size()




plt.figure(figsize=(12, 6))
sns.lineplot(x=flights_by_year.index, y=flights_by_year.values, marker="o", color="red")
plt.title(f"{top_origin} → {top_dest} 노선의 연도별 운항 횟수 변화")
plt.xticks(ticks=flights_by_year.index, labels=flights_by_year.index)
plt.ticklabel_format(style='plain', axis='y')
plt.show()



















