#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 12:15:05 2025

@author: gim-yebin
"""

'''
8. 시카고(ORD)와 같은 주요 공항을 오가는 모든 항공편을 비교
'''

import pandas as pd
import numpy as np


path = '/Users/gim-yebin/Downloads/merged_flight_data.csv'
df = pd.read_csv(path, encoding='latin-1', low_memory=False)

df[['Origin', 'Dest', 'FlightNum']].isnull().sum()
'''
'Dest']].isnull().sum())
Origin    0
Dest      0
dtype: int64

'''

df.describe()

df = df[(df["Year"] > 1987) & (df["Year"] < 2008)]  



Q1 = df["FlightNum"].quantile(0.25)  
Q3 = df["FlightNum"].quantile(0.75)  
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df = df[(df["FlightNum"] >= lower_bound) & (df["FlightNum"] <= upper_bound)]

df.describe()


top_3_origin_airports = df['Origin'].value_counts().head(3)

'''
ORD    6365070
ATL    5823454
DFW    5522740
Name: count, dtype: int64
'''

top_3_dest_airports = df['Dest'].value_counts().head(3)

'''
ORD    6405594
ATL    5816615
DFW    5557269
Name: count, dtype: int64
'''

top_5_flights_from_top3 = df[df['Origin'].isin(top_3_origin_airports.index)]['FlightNum'].value_counts().head(5)
''' '출발 공항 TOP 3에서 가장 많이 운항된 항공편:'''
top_5_flights_from_top3
'''
FlightNum
500    21282
376    20347
578    20211
690    19726
694    19139
Name: count, dtype: int64
'''

top_5_flights_to_top3 = df[df['Dest'].isin(top_3_dest_airports.index)]['FlightNum'].value_counts().head(5)
top_5_flights_to_top3

'''
도착 공항 TOP 3로 가장 많이 운항된 항공편:
FlightNum
551    21550
691    20274
410    19001
816    18772
409    18633
Name: count, dtype: int64
'''
  

import matplotlib.pyplot as plt
import seaborn as sns

# 한글을 표기하기 위한 글꼴 변경
from matplotlib import font_manager, rc
import platform

if platform.system() == 'Darwin':  # Mac OS
    rc('font', family='AppleGothic')  
elif platform.system() == 'Windows':  # Windows
    font_path = "C:/Windows/Fonts/malgun.ttf"  # '맑은 고딕' 설정
    font_name = font_manager.FontProperties(fname=font_path).get_name()
    rc('font', family=font_name)
else:  # 리눅스 환경 (추가 설정 가능)
    print("⚠ 리눅스 환경에서는 별도 폰트 설치 필요!")

plt.rcParams['axes.unicode_minus'] = False





ord_atl_dfw_routes = df[(df['Origin'].isin(['ORD', 'ATL', 'DFW'])) & 
                            (df['Dest'].isin(['ORD', 'ATL', 'DFW']))]

# ORD → ORD 제거
ord_atl_dfw_routes = ord_atl_dfw_routes[ord_atl_dfw_routes['Origin'] != ord_atl_dfw_routes['Dest']]

flight_matrix = ord_atl_dfw_routes.groupby(['Origin', 'Dest']).size().unstack().fillna(0)

plt.figure(figsize=(8, 6))
sns.heatmap(flight_matrix, annot=True, fmt=".0f", cmap="Blues")
plt.title("ORD, ATL, DFW 주요 공항 간 항공편 수")
plt.xlabel("도착 공항")
plt.ylabel("출발 공항")
plt.show()






