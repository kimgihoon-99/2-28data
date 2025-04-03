# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 15:33:24 2025

@author: Admin
"""

import matplotlib.pyplot as plt
import seaborn as sns

df_flight = pd.read_csv('./backup_week_data.csv')

# 비행기 명을 비행기 출시연도와 연결
key: 'tailnum'
usecols = ['tailnum', 'issue_date']
air = pd.read_csv('./dataverse_files_2000-2008/plane-data.csv', usecols=usecols).dropna()
air.info()
air['issue_date'] = air['issue_date'].dropna().astype(str).str[-4:]
result = dict(zip(air['tailnum'], air['issue_date']))
print(result)


df_flight['TailNum'] = df_flight['TailNum'].map(result)

print(df_other)

df_flight.isna().sum()


df_flight.dropna()
plane = df_flight['TailNum'].value_counts().sort_index(ascending=True)

plt.figure(figsize=(12,6))

sns.barplot(x = plane.index, y = plane.values, hue = plane.index, palette = 'RdGy')
                                
plt.title('', fontsize=16)
plt.xlabel('비행기년도', fontsize=14)
plt.ylabel('지연 비행기 수', fontsize=14)

plt.grid(axis='x') # x를 기준으로 grid 그리겠다
plt.show()

