# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 09:59:29 2025

@author: 오정우
--1번 코드--
"""
# 파일병합
import pandas as pd
import time

sleep_time = 300

usecols = ['ArrDelay', 'DayOfWeek', 'TailNum', 'DepTime']

dfs_1987_1999 = []
dfs_2000_2008 = []

# 1987 ~ 1999 파일 읽기
for year in range(1987, 2000):
    try:
        print(f"{year}년 데이터 읽는 중...")
        df = pd.read_csv(f'./dataverse_files_1987-1999/{year}.csv', usecols=usecols).dropna()
        dfs_1987_1999.append(df)
        print(f"{year}년 데이터 완료!")
        time.sleep(sleep_time)
    except Exception as e:
        print(f"오류 발생: {year}년 데이터에서 오류 발생 - {e}")
        continue

# 2000 ~ 2008 파일 읽기
for year in range(2000, 2009):
    try:
        print(f"{year}년 데이터 읽는 중...")
        if year in [2001, 2002]:  # 인코딩 문제 있는 파일
            df = pd.read_csv(f'./dataverse_files_2000-2008/{year}.csv', 
                             encoding='latin-1', low_memory=False, usecols=usecols).dropna()
        else:
            df = pd.read_csv(f'./dataverse_files_2000-2008/{year}.csv', usecols=usecols).dropna()
        
        dfs_2000_2008.append(df)
        print(f"{year}년 데이터 완료!)
        time.sleep(sleep_time)
    except Exception as e:
        print(f"오류 발생: {year}년 데이터에서 오류 발생- {e}")
        continue

# 데이터프레임 합치기
df_flight = pd.concat(dfs_1987_1999 + dfs_2000_2008, ignore_index=True)


df_flight = pd.read_csv('./backup_week_data.csv')

print(df_flight.head())
print("전체 데이터 개수:", len(df_flight))


# 공통 전처리
df_flight = df_flight[df_flight['ArrDelay'] >= 0]
# 요일 기준 전처리



week = df_flight['DayOfWeek'].value_counts()
print(week.head())

week = week.reindex(range(1, 8))

# 라이브러리 임포트
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
import platform


sns.set_style("whitegrid")
plt.figure(figsize=(12, 6))

if platform.system() == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'
elif platform.system() == 'Darwin':  # macOS
    plt.rcParams['font.family'] = 'AppleGothic'
else:
    plt.rcParams['font.family'] = 'NanumGothic'

plt.rcParams['axes.unicode_minus'] = False  # 마이너스 깨짐 방지


x_values = week.index
y_values = week.values


plt.bar(x_values, y_values, color="skyblue", edgecolor="black")
  # y축 최소값 설정 (예: 100부터 시작)


plt.ylim(1250000, max(y_values) + 200000)


plt.title("요일별 데이터 시각화", fontsize=14, fontweight="bold")
plt.xlabel("요일", fontsize=12)
plt.ylabel("지연 비행기 수", fontsize=12)


day_labels = ['일', '월', '화', '수', '목', '금', '토']
plt.xticks(ticks=range(1, 8), labels=day_labels)

plt.show()

plt.ylim(3250000, max(y_values) + 200000)

# DepTime을 문자열로 변환 후 hh만 추출
df_flight['DepTime'] = df_flight['DepTime'].dropna().astype(int).astype(str).str.zfill(4).str[:-2]

# hh별 빈도수 구하고 정렬
time = df_flight['DepTime'].value_counts().sort_index()
print(time)


### 시간대 그래프

sns.set_style("whitegrid")
plt.figure(figsize=(12, 6))

if platform.system() == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'
elif platform.system() == 'Darwin':  # macOS
    plt.rcParams['font.family'] = 'AppleGothic'
else:
    plt.rcParams['font.family'] = 'NanumGothic'

plt.rcParams['axes.unicode_minus'] = False 

x_values = time.index
y_values = time.values


plt.plot(x_values, y_values, marker="o", linestyle="-", color="b", linewidth=2, markersize=8)


plt.title("시간대별 비행기 지연", fontsize=14, fontweight="bold")
plt.xlabel("시간대", fontsize=12)
plt.ylabel("지연 비행기 수", fontsize=12)


plt.show()






print(week.sample(5))


df_flight.to_csv('backup_week_data.csv', index=False)

week.to_pickle('week_data_backup.pkl')


pd.df_flight[]