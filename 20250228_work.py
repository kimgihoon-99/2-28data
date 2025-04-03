# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 09:14:07 2025

@author: Admin
"""
'''데이터 불러오기 및 전처리 '''

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 한글 폰트 설정 (Windows 기준)
from matplotlib import font_manager, rc
import platform

if platform.system() == 'Windows':
    path = 'c:/Windows/Fonts/malgun.ttf'
    font_name = font_manager.FontProperties(fname=path).get_name()
    rc('font', family=font_name)
elif platform.system() == 'Darwin':
    rc('font', family='AppleGothic')
else:
    print('Check your OS system')

df_1987 = pd.read_csv('./dataverse_files_1987-1999/1987.csv')
df_1988 = pd.read_csv('./dataverse_files_1987-1999/1988.csv')
df_1989 = pd.read_csv('./dataverse_files_1987-1999/1989.csv')
df_1990 = pd.read_csv('./dataverse_files_1987-1999/1990.csv')
df_1991 = pd.read_csv('./dataverse_files_1987-1999/1991.csv')
df_1992 = pd.read_csv('./dataverse_files_1987-1999/1992.csv')
df_1993 = pd.read_csv('./dataverse_files_1987-1999/1993.csv')
df_1994 = pd.read_csv('./dataverse_files_1987-1999/1994.csv')
df_1995 = pd.read_csv('./dataverse_files_1987-1999/1995.csv')
df_1996 = pd.read_csv('./dataverse_files_1987-1999/1996.csv')
df_1997 = pd.read_csv('./dataverse_files_1987-1999/1997.csv')
df_1998 = pd.read_csv('./dataverse_files_1987-1999/1998.csv')
df_1999 = pd.read_csv('./dataverse_files_1987-1999/1999.csv')

df_2000 = pd.read_csv('./dataverse_files_2000-2008/2000.csv')
df_2001 = pd.read_csv('./dataverse_files_2000-2008/2001.csv', encoding='latin-1', low_memory=False)
df_2002 = pd.read_csv('./dataverse_files_2000-2008/2002.csv', encoding='latin-1', low_memory=False)
df_2003 = pd.read_csv('./dataverse_files_2000-2008/2003.csv')
df_2004 = pd.read_csv('./dataverse_files_2000-2008/2004.csv')
df_2005 = pd.read_csv('./dataverse_files_2000-2008/2005.csv')
df_2006 = pd.read_csv('./dataverse_files_2000-2008/2006.csv')
df_2007 = pd.read_csv('./dataverse_files_2000-2008/2007.csv')
df_2008 = pd.read_csv('./dataverse_files_2000-2008/2008.csv')


'''1번 지연을 최소화하려면 비행에 적합한 시간대/요일/시간은 언제인가?'''

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
        print(f"{year}년 데이터 완료!")
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



'''2번 오래된 비행기일수록 지연이 더 잦나?'''
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

'''
3번 시간이 지남에 따라 다양한 장소 간을 비행하는 사람의 수는 어떻게 변하나?

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


''' 4번. 날씨는 비행기 지연을 얼마나 잘 예측할 수 있나? '''

# seaborn의 스타일 적용
sns.set_theme(style="whitegrid")


# 파일 병합
# 기상 지연 데이터는 2004~2008에만 존재하기 때문에 해당 파일만 저장해서 병합
file_list = [f'./data/dataverse_files_2000-2008/200{i}.csv' for i in range(4, 9)]
dfs = []

for file_path in file_list:
    try:
        df = pd.read_csv(file_path, encoding='latin-1',
                         usecols=['WeatherDelay', 'ArrDelay', 'DepDelay', 'Origin', 'Dest', 'Month'])
        dfs.append(df)
        print(f"File {file_path} loaded, shape: {df.shape}")
    except Exception as e:
        print(f"Error reading {file_path}: {e}")

total_df = pd.concat(dfs, ignore_index=True)
total_df.shape

# 데이터 전처리
for col in ['WeatherDelay', 'ArrDelay', 'DepDelay', 'Month']:
    total_df[col] = pd.to_numeric(total_df[col], errors='coerce')

# 결측치 제거
total_df.dropna(subset=['WeatherDelay', 'ArrDelay', 'DepDelay', 'Month', 'Origin', 'Dest'], inplace=True)
total_df.shape

# 필터링 (WeatherDelay, ArrDelay, DepDelay 모두 0보다 크고, WeatherDelay < 1200)
df_filtered = total_df[
    (total_df['WeatherDelay'] > 0) &
    (total_df['ArrDelay'] > 0) &
    (total_df['DepDelay'] > 0) &
    (total_df['WeatherDelay'] < 1200)
].copy()

df_filtered.shape
df_filtered[['WeatherDelay', 'ArrDelay', 'DepDelay']].describe()
# 상관관계 분석
corr_matrix = df_filtered[['WeatherDelay', 'ArrDelay', 'DepDelay']].corr()
plt.figure(figsize=(6, 5))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("지연 변수 간 상관관계")
plt.show()

# 산점도 그래프 (WeatherDelay vs. ArrDelay/DepDelay)
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.scatterplot(data=df_filtered, x='WeatherDelay', y='ArrDelay', alpha=0.3, color='tab:blue')
plt.title('기상 지연 vs. 도착 지연')

plt.subplot(1, 2, 2)
sns.scatterplot(data=df_filtered, x='WeatherDelay', y='DepDelay', alpha=0.3, color='tab:orange')
plt.title('기상 지연 vs. 출발 지연')

plt.tight_layout()
plt.show()

''' 5번 한 공항의 지연으로 인해 다른 공항의 지연이 발생하는 연쇄적 실패를 감지할 수 있는지?'''
# 1. CSV 파일 병합 및 전처리

file_list = [f'./data/dataverse_files_2000-2008/200{i}.csv' for i in range(4, 9)]
dfs = []

for file_path in file_list:
    try:
        df_temp = pd.read_csv(
            file_path,
            encoding='latin-1',
            usecols=[
                'Year', 'Month', 'DayofMonth',
                'DepTime', 'ArrTime', 'CRSDepTime', 'CRSArrTime',
                'TailNum', 'Origin', 'Dest',
                'ArrDelay', 'DepDelay', 'LateAircraftDelay'
            ]
        )
        dfs.append(df_temp)
    except Exception as e:
        print(f"{e}")

df = pd.concat(dfs, ignore_index=True)
df.shape

## 숫자형으로 컬럼들 변환
num_cols = ['Year','Month','DayofMonth','DepTime','ArrTime',
            'CRSDepTime','CRSArrTime','ArrDelay','DepDelay',
            'LateAircraftDelay']
for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# 결측치 제거하기
df.dropna(subset=num_cols + ['TailNum','Origin','Dest'], inplace=True)
df.shape

# 2. 출발편이 가장 많은 공항 선정 및 일별 평균 지연 집계

# Origin 기준으로 선정
airport_counts = df['Origin'].value_counts().reset_index()
airport_counts.columns = ['Airport', 'FlightCount']
hub_airport = airport_counts.iloc[0]['Airport']
hub_airport

# 해당 공항에 도착한 항공편 → 일별 평균 도착 지연
hub_arr = (
    df[df['Dest'] == hub_airport]
    .groupby('Date')['ArrDelay']
    .mean()
    .reset_index()
    .rename(columns={'ArrDelay': 'AvgArrDelay'})
)

# 해당 공항에서 출발한 항공편 → 일별 평균 출발 지연
hub_dep = (
    df[df['Origin'] == hub_airport]
    .groupby('Date')['DepDelay']
    .mean()
    .reset_index()
    .rename(columns={'DepDelay': 'AvgDepDelay'})
)

hub_df = pd.merge(hub_arr, hub_dep, on='Date', how='inner')
hub_df.sort_values('Date', inplace=True)

# 전날 도착 지연(1일 lag)
hub_df['LagAvgArrDelay'] = hub_df['AvgArrDelay'].shift(1)

# 3. 이동평균(7일) & 시각화
hub_df = hub_df.sort_values('Date')
hub_df['AvgArrDelay_7d'] = hub_df['AvgArrDelay'].rolling(7).mean()
hub_df['AvgDepDelay_7d'] = hub_df['AvgDepDelay'].rolling(7).mean()
hub_df['LagAvgArrDelay_7d'] = hub_df['LagAvgArrDelay'].rolling(7).mean()

plt.figure(figsize=(14, 6))
plt.plot(hub_df['Date'], hub_df['AvgDepDelay_7d'],
         label='7일 이동평균 출발 지연',
         marker='o', markersize=3, linewidth=2)
plt.plot(hub_df['Date'], hub_df['AvgArrDelay_7d'],
         label='7일 이동평균 도착 지연',
         marker='s', markersize=3, linewidth=2)
plt.plot(hub_df['Date'], hub_df['LagAvgArrDelay_7d'],
         label='(전날) 7일 이동평균 도착 지연',
         marker='^', markersize=3, linestyle='--', linewidth=2)

plt.title(f'{hub_airport} 공항 - 7일 이동평균 지연 추세')
plt.xlabel('날짜')
plt.ylabel('지연 시간 (분)')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 4. 상위 5개 기체(TailNum) 선택
tail_stats = df.groupby('TailNum').agg(flight_count=('TailNum', 'count')).reset_index()
top_tails = tail_stats.sort_values('flight_count', ascending=False).head(5)['TailNum'].tolist()
top_tails

## 상위 기체 데이터만 선택
df_subset = df[df['TailNum'].isin(top_tails)].copy()

# 5. 기체별로 날짜+DepTime 정렬
df_subset.sort_values(['TailNum','Date','DepTime'], inplace=True)

# 6. 이전 항공편 정보 연결 (shift)
## 이전 비행의 도착 공항, 도착 지연, 도착 시각
df_subset['PrevDest'] = df_subset.groupby('TailNum')['Dest'].shift(1)
df_subset['PrevArrDelay'] = df_subset.groupby('TailNum')['ArrDelay'].shift(1)
df_subset['PrevArrTime'] = df_subset.groupby('TailNum')['ArrTime'].shift(1)
df_subset['PrevDate'] = df_subset.groupby('TailNum')['Date'].shift(1)

## 이전 비행의 도착 공항 == 현재 비행의 출발 공항 → 연결된 항공편
df_subset['IsConnected'] = (df_subset['Origin'] == df_subset['PrevDest'])

# 7. 연결된 항공편만 추출
df_connected = df_subset[df_subset['IsConnected']].copy()
df_connected.shape[0]

# 8. 0보다 큰 지연만 필터링
df_filtered = df_connected.dropna(subset=['PrevArrDelay','DepDelay'])
df_filtered = df_filtered[(df_filtered['PrevArrDelay'] > 0) & (df_filtered['DepDelay'] > 0)]
df_filtered.shape[0]

# 9. 전체 상관분석 및 산점도
overall_corr = df_filtered[['PrevArrDelay', 'DepDelay']].corr().iloc[0, 1]
overall_corr

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_filtered, x='PrevArrDelay', y='DepDelay', hue='TailNum')
plt.title("이전 도착 지연 vs 현재 출발 지연")
plt.xlabel("이전 도착 지연")
plt.ylabel("현재 출발 지연")
plt.legend(title='TailNum')
plt.tight_layout()
plt.show()

''' 6번 9.11테러 이전과 이후 비행 패턴 변화 '''

# 공항 우회 여부 
df_1987['Diverted'].value_counts() 
df_1988['Diverted'].value_counts() 
df_1989['Diverted'].value_counts()
df_1990['Diverted'].value_counts()
df_1991['Diverted'].value_counts()
df_1992['Diverted'].value_counts()
df_1993['Diverted'].value_counts()
df_1994['Diverted'].value_counts()
df_1995['Diverted'].value_counts()
df_1996['Diverted'].value_counts()
df_1997['Diverted'].value_counts()
df_1998['Diverted'].value_counts()
df_1999['Diverted'].value_counts()
df_2000['Diverted'].value_counts()
df_2001['Diverted'].value_counts()
df_2002['Diverted'].value_counts()
df_2003['Diverted'].value_counts()
df_2004['Diverted'].value_counts()
df_2005['Diverted'].value_counts()
df_2006['Diverted'].value_counts()
df_2007['Diverted'].value_counts()
df_2008['Diverted'].value_counts()


# 데이터 설정
years = list(range(1987, 2009))  # 1987년 ~ 2008년
values = [3815, 14436, 14839, 15954, 12585, 11384, 10333, 12106, 10492, 
          14121, 12081, 13161, 13555, 14254, 12909, 8356, 11381, 13784, 
          14028, 16186, 17179, 5654]

# 그래프 스타일 적용
sns.set_style("whitegrid")
plt.figure(figsize=(12, 6))

# 선 그래프
plt.plot(years, values, marker="o", linestyle="-", color="b", linewidth=2, markersize=6, label="연도별 데이터")

# 바 그래프 추가 (선택 사항)
plt.bar(years, values, alpha=0.3, color="skyblue", label="연도별 값")

# 그래프 제목 및 축 레이블
plt.title("Whether to detour the flight route", fontsize=14, fontweight="bold")
plt.xlabel("years", fontsize=12)
plt.ylabel("values", fontsize=12)

# 데이터 값 표시
for i, value in enumerate(values):
    plt.text(years[i], value, f"{value:,}", ha="center", va="bottom", fontsize=10, color="black")

# 범례 추가
plt.legend()

# x축 눈금 회전
plt.xticks(rotation=45)

# 그래프 출력
plt.show()


# 비행 취소 여부
df_1987['Cancelled'].value_counts() 
df_1988['Cancelled'].value_counts() 
df_1989['Cancelled'].value_counts()
df_1990['Cancelled'].value_counts()
df_1991['Cancelled'].value_counts()
df_1992['Cancelled'].value_counts()
df_1993['Cancelled'].value_counts()
df_1994['Cancelled'].value_counts()
df_1995['Cancelled'].value_counts()
df_1996['Cancelled'].value_counts()
df_1997['Cancelled'].value_counts()
df_1998['Cancelled'].value_counts()
df_1999['Cancelled'].value_counts()
df_2000['Cancelled'].value_counts()
df_2001['Cancelled'].value_counts()
df_2002['Cancelled'].value_counts()
df_2003['Cancelled'].value_counts()
df_2004['Cancelled'].value_counts()
df_2005['Cancelled'].value_counts()
df_2006['Cancelled'].value_counts()
df_2007['Cancelled'].value_counts()
df_2008['Cancelled'].value_counts()


# 연도 데이터
years = list(range(1987, 2009))

# 두 번째 데이터셋
data1 = [19685, 50163, 74165, 52458, 43505, 52836, 59845, 66740, 91905, 128536, 
         97763, 144509, 154311, 187490, 231198, 65143, 101469, 127757, 133730, 121934, 160748, 64442]

# 그래프 생성
plt.figure(figsize=(12, 6))
plt.plot(years, data1, marker='o', linestyle='-', color='b', label='데이터 1')


# 그래프 제목 및 축 레이블 설정
plt.title('Cancellation of flight')
plt.xlabel('years')
plt.ylabel('values')
plt.legend()
plt.grid(True)
plt.show()

'''
7. 가장 자주 비행하는 두 도시 간을 오가는 모든 항공편을 비교
'''

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

'''
8번 시카고(ORD)와 같은 주요 공항을 오가는 모든 항공편을 비교
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


''' 9번 어느 달에 비행기 이용수가 가장 많을까 '''
df_1987['Month'].value_counts() 
df_1988['Month'].value_counts() 
df_1989['Month'].value_counts()
df_1990['Month'].value_counts()
df_1991['Month'].value_counts()
df_1992['Month'].value_counts()
df_1993['Month'].value_counts()
df_1994['Month'].value_counts()
df_1995['Month'].value_counts()
df_1996['Month'].value_counts()
df_1997['Month'].value_counts()
df_1998['Month'].value_counts()
df_1999['Month'].value_counts()
df_2000['Month'].value_counts()
df_2001['Month'].value_counts()
df_2002['Month'].value_counts()
df_2003['Month'].value_counts()
df_2004['Month'].value_counts()
df_2005['Month'].value_counts()
df_2006['Month'].value_counts()
df_2007['Month'].value_counts()
df_2008['Month'].value_counts()


# 월별 데이터
months = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"]
values = [10272489, 9431225, 10448039, 10081982, 9724174, 9618281, 9944011, 10034556, 
          9435046, 10202453, 9694904, 10027298]

# 그래프 스타일 설정
sns.set_style("whitegrid")
plt.figure(figsize=(12, 6))

# 선 그래프 그리기
plt.plot(months, values, marker="o", linestyle="-", color="b", linewidth=2, markersize=8)

# 그래프 제목 및 레이블 설정
plt.title("Number of flight users per month", fontsize=14, fontweight="bold")
plt.xlabel("Month", fontsize=12)
plt.ylabel("values", fontsize=12)

# 데이터 값 표시
for i, value in enumerate(values):
    plt.text(i, value, f"{value:,}", ha="center", va="bottom", fontsize=10, color="black")

# x축 회전 (가독성 증가)
plt.xticks(rotation=45)

# 그래프 출력
plt.show()


# 연도 데이터
years = list(range(1987, 2009))

# 값 데이터
values = [1311826, 5202096, 5041200, 5270893, 5076925, 5092157, 5070501, 5180048, 
          5327435, 5351983, 5411843, 5384721, 5527884, 5683047, 5967780, 5271359, 
          6488540, 7129270, 7140596, 7141922, 7453215, 2389217]

# 그래프 크기 설정
plt.figure(figsize=(14, 7))

# 선 그래프
plt.plot(years, values, marker='o', linestyle='-', color='blue', label='years')

# 그래프 제목 및 레이블
plt.title('Number of flights per year', fontsize=16)
plt.xlabel('years', fontsize=12)
plt.ylabel('values', fontsize=12)
plt.xticks(years, rotation=45)  # 연도 눈금 회전
plt.grid(True, linestyle='--', alpha=0.7)

# 데이터 값 표시
for i, value in enumerate(values):
    plt.text(years[i], value, f"{value:,}", ha='center', va='bottom', fontsize=9, color='black')

# 범례 추가
plt.legend()

# 그래프 출력
plt.tight_layout()
plt.show()

















