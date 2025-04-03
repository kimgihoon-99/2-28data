# --- 5. 한 공항의 지연으로 인해 다른 공항의 지연이 발생하는 연쇄적 실패를 감지할 수 있는지? ---

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
