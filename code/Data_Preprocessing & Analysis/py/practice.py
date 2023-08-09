## 잡다한거 종합적으로 다 모아놓음

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
from statsmodels.stats.weightstats import ztest
import numpy as np
from sklearn.preprocessing import MinMaxScaler

import seaborn as sns
from operator import itemgetter
from matplotlib.patches import Rectangle

from statsmodels.stats.weightstats import ztest
from statsmodels.tsa.seasonal import seasonal_decompose

import scipy.stats as stats

### CSV 파일 불러오기
stock_df = pd.read_csv('C:/Users/user/Desktop/Securities_Data_Analysis(Junsu)/dataset/original_data/나스닥(1985~2023)_yfinance.csv')
gold_df = pd.read_csv('C:/Users/user/Desktop/Securities_Data_Analysis(Junsu)/dataset/original_data/금(1950~2023)_캐글.csv')
fund_df = pd.read_csv('C:/Users/user/Desktop/Securities_Data_Analysis(Junsu)/dataset/original_data/미국금리(1954.7~2023.5)_구글서치.csv')
house_df = pd.read_csv('C:/Users/user/Desktop/Securities_Data_Analysis(Junsu)/dataset/original_data/케이스-쉴러_미국주택가격지수(1987.1~2023.4).csv')
bond_df = pd.read_csv('C:/Users/user/Desktop/Securities_Data_Analysis(Junsu)/dataset/original_data/10년만기 미국채 선물 과거 데이터.csv')
gdp_df = pd.read_csv('C:/Users/user/Desktop/Securities_Data_Analysis(Junsu)/dataset/original_data/세계GDP성장률_캐글(1961~2020)_month.csv')
snp_df = pd.read_csv('C:/Users/user/Desktop/Securities_Data_Analysis(Junsu)/dataset/original_data/S&P 500 과거 데이터 Feb 1970 ~.csv')
unemploy_df = pd.read_csv('C:/Users/user/Desktop/Securities_Data_Analysis(Junsu)/dataset/original_data/US_Unemployment(1950~2023.07)_kag.csv')

### 불러온 데이터 확인하기
# print(stock_df.head)
# print(gold_df.head)
# print(fund_df.head)
# print(house_df.head)
# print(bond_df.head)
# print(gdp_df.head)

### 데이터프레임 컬럼 이름 바꾸기
gold_df = gold_df.rename(columns={'Price USD per Oz':'Gold_Price'})
fund_df = fund_df.rename(columns={'FEDFUNDS':'Funds_Rate', "DATE":"Date"})
house_df = house_df.rename(columns={'SPCS10RSA':'House_Price', "DATE":"Date"})
bond_df = bond_df.rename(columns={'날짜':'Date', '종가':'Bond_Close'})
snp_df = snp_df.rename(columns={'날짜':'Date', '종가':'Snp_Close'})
unemploy_df = unemploy_df.rename(columns={'Unemplyment Rate':'Unemployment_Rate'})

### 불러올 날짜 구간 설정(공통 1987-01-01 ~ 2023-07-01)
start = "2019-11-01" # 최소 1950-01-01
end = "2020-12-01" # 최대 2023-07-01
stock_df = stock_df[stock_df['Date'].between(start, end)]
gold_df = gold_df[gold_df['Date'].between(start, end)]
fund_df = fund_df[fund_df['Date'].between(start, end)]
house_df = house_df[house_df['Date'].between(start, end)]
bond_df = bond_df[bond_df['Date'].between(start, end)]
snp_df = snp_df[snp_df['Date'].between(start, end)]
unemploy_df = unemploy_df[unemploy_df['Date'].between('1961-01-01', '2023-07-01')]
unemploy_df = unemploy_df[['Date', 'Unemployment_Rate']]
# gdp_df = gdp_df[gdp_df['Date'] >= '1990-07-01']

### 날짜 datatime 형식으로 전환하기
stock_df.loc[:,'Date'] = pd.to_datetime(stock_df.Date)
gold_df.loc[:,'Date'] = pd.to_datetime(gold_df.Date)
fund_df.loc[:,'Date'] = pd.to_datetime(fund_df.Date)
house_df.loc[:,'Date'] = pd.to_datetime(house_df.Date)
bond_df.loc[:,'Date'] = pd.to_datetime(bond_df.Date)
gdp_df.loc[:,'Date'] = pd.to_datetime(gdp_df.Date)
snp_df.loc[:,'Date'] = pd.to_datetime(snp_df.Date)
unemploy_df.loc[:,'Date'] = pd.to_datetime(unemploy_df.Date)


### index를 날짜로 변경하기
stock_df = stock_df.set_index('Date')
gold_df = gold_df.set_index('Date')
fund_df = fund_df.set_index('Date')
house_df = house_df.set_index('Date')
bond_df = bond_df.set_index('Date')
gdp_df = gdp_df.set_index('Date')
snp_df = snp_df.set_index('Date')
unemploy_df = unemploy_df.set_index('Date')


### 특정 열의 타입 변경하기
# snp_df = snp_df.astype({'Snp_Close' : 'float64'})

### 데이터프레임 속성 확인하기
# print(df.columns) # 해당 파일의 행렬 개수와 열 이름 확인
# print(df.shape) # (행의 개수, 열의 개수) 출력
# print(df.info()) # 해당 파일 열의 타입과 null행 수 확인 

######------------------------------------------------------- 데이터 전처리 -----------------------------------------------------------unt)

### 특정 원소 또는 행 알아보기
# df.loc[[2,4], ['컬럼1', '컬럼2', '컬럼3']]
# df.loc[['b', 'e'], ['컬럼1', '컬럼2']]
# print(df.loc[[188,191,194]]) # 특정 열 출력

### 평균, 표준편차
# df.mean() # 컬럼의 평균
# df.mean()['C1'] # C1 컬럼의 평균
# df.mean(1) # 행의 평균
# df.std() # 컬럼의 표준편차

### 데이터 프레임 간 연산(컬럼 이름이 동일하지 않으면 더했을 때 모두 NaN 처리)
# df_1 + df_2





# --------------- 결측치 ---------------- #

### 컬럼별 NA 개수 확인
# print(gdp_df.isna().sum())
# print(gold_df.isna().sum())
# print(fund_df.isna().sum())
# print(house_df.isna().sum())
# print(bond_df.isna().sum())
# print(stock_df.isnull().sum()) # 컬럼 별 결측치 확인
# print(len(stock_df)-stock_df.count()) # 컬럼 별 결측치 확인
gdp_df = gdp_df.interpolate(method = "time") # 선형보간법으로 채우기(method = value : 선형 / method = time : 시간)

### 특정 칼럼(열) 삭제
stock_df = stock_df.drop(['Open', 'High', 'Low', 'Adj Close', 'Volume'], axis = 1)
bond_df = bond_df.drop(['시가', '고가', '저가', '변동 %'], axis = 1)
snp_df = snp_df.drop(['시가', '고가', '저가', '거래량', '변동 %'], axis = 1)

### 결측치가 있는 행 or 열 제거하는 함수 (axis = 0 : 행 / axis = 1 : 열 / 생략 시 0 디폴트)
stock_df = stock_df.dropna(axis=0)
fund_df = fund_df.dropna(axis=0)
house_df = house_df.dropna(axis=0)
bond_df = bond_df.dropna(axis=0)
snp_df = snp_df.dropna(axis=0)
unemploy_df = unemploy_df.dropna(axis=1)
gdp_df = gdp_df.dropna(axis=1)
# df = df[['컬럼1', '컬럼2']].dropna()

### 결측치 채우기
# df.loc[191, '수출금액'] = (df.loc[188, '수출금액'] + df.loc[194, '수출금액']) / 2 
# df.loc[191, '무역수지'] = df.loc[188, '수출금액'] - df.loc[194, '수입금액']
# df = df.fillna(0) # 결측치 0으로 채우기
# df.fillna(method='ffill') # 결측치 위에서 아래 방향으로 채우기
# df.fillna(method='bfill') # 결측치 아래에서 위 방향으로 채우기
# df.fillna(method='ffill', limit=1) # 결측치 위에서 아래 방향으로 1번 채우기
# df.fillna(df.mean()['C1':'C2']) # 컬럼의 평균으로 C1, C2 컬럼 채우기
# df = df.interpolate(method = value) # 선형보간법으로 채우기(method = value : 선형 / method = time : 시간)
# df.replace({'col1': old_val}, {'col1': new_val}) # 특정 컬럼 값 replace로 변경하기


# --------------- 중복치 ---------------- #

### 중복데이터 확인하기
# print(fund_df[fund_df.duplicated()])
# print(stock_df[stock_df.duplicated()])
# print(house_df[house_df.duplicated()])
# print(bond_df[bond_df.duplicated()])
# print(gold_df[gold_df.duplicated()])

### 중복데이터 삭제하기
# df.drop_duplicates(['컬럼'], keep = 'first')
# print("삭제 완료")
# print(df[df.duplicated()])
# df.drop_duplicates(subset=['id'], keep='last') # 특정 열이 고유한 key를 가지는 경우 중복된 데이터 중 뒤를 남김







# --------------- 이상치 ---------------- #
# fig, ax = plt.subplots(figsize=(9,6)) # 정규분포를 따르는지 그래프로 확인
# _ = plt.hist(df.Close, 100, density=True, alpha=0.75)
# plt.show()

# _, p = ztest(df.Close) # p가 0.05이하로 나온다면 정규분포와 거리가 멀다는 뜻
# print(p)

# 위 주식 데이터의 분포 확인
# fig, ax = plt.subplots(figsize=(9,6))
# _ = plt.hist(stock_df.Close, 100, density=True, alpha=0.75)
# plt.show()

# _, p = ztest(stock_df.Close)
# print(p)

# 계절적 성분 50일로 가정
# extrapolate_trend='freq' : Trend 성분을 만들기 위한 rolling window 때문에 필연적으로 trend, resid에는 Nan 값이 발생하기 때문에, 이 NaN값을 채워주는 옵션이다.
# result = seasonal_decompose(stock_df.Close, model='additive', two_sided=True, 
#                             period=50, extrapolate_trend='freq') 
# result.plot()
# plt.show()
# result.seasonal[:100].plot()
# plt.show()

# Residual의 분포 확인
# fig, ax = plt.subplots(figsize=(9,6))
# _ = plt.hist(result.resid, 100, density=True, alpha=0.75)
# plt.show()

# r = result.resid.values
# st, p = ztest(r)
# print(st,p)

# 평균과 표준편차 출력
# mu, std = result.resid.mean(), result.resid.std()
# print("평균:", mu, "표준편차:", std)
# 평균: -0.3595321143716522 표준편차: 39.8661527194307

# 3-sigma(표준편차)를 기준으로 이상치 판단
# print("이상치 갯수:", len(result.resid[(result.resid>mu+4*std)|(result.resid<mu-4*std)]))
# 이상치 갯수: 71
# print(stock_df.Date[result.resid[(result.resid>mu+4*std)|(result.resid<mu-4*std)].index])






#######---------------------------------------- 그래프그리기 ----------------------------------------------------######
scaler = MinMaxScaler() # min-max 정규화를 위한 스케일러
plt.rcParams["figure.figsize"] = (16,8)
# 유니코드 깨짐현상 해결
plt.rcParams['axes.unicode_minus'] = False
 
# 나눔고딕 폰트 적용
plt.rcParams["font.family"] = 'NanumGothic'

plt.subplot(121)
plt.title('GDP GROWTH & UNEMPLOYMENT RATE (US)', fontsize=20) 
plt.rcParams["figure.figsize"] = (16,8)
plt.ylabel('GROWTH RATE', fontsize=14)
plt.xlabel('Date', fontsize=14)
plt.plot(gdp_df.index, gdp_df.GDP, color='black', linewidth = 3)
plt.plot(unemploy_df.index, unemploy_df.Unemployment_Rate, color='y', linewidth = 3)
plt.ylabel('figure', fontsize=12)
plt.xlabel('Date', fontsize=12)
plt.legend(['GDP', 'UNEMPLOY'], fontsize=12, loc='best')

economic_df = pd.concat([gdp_df, unemploy_df], axis = 1)

economic_df.columns = ['GDP', 'UNEMPLOYMENT']
economic_df = economic_df.dropna()
print(economic_df)

gdp_growth_df = pd.DataFrame(data=economic_df.GDP)
scaler.fit(gdp_growth_df)
gdp_growth_scaled = scaler.transform(gdp_growth_df)
gdp_growth_df_scaled = pd.DataFrame(data=gdp_growth_scaled)

unemploy_rate_df = pd.DataFrame(data=economic_df.UNEMPLOYMENT)
scaler.fit(unemploy_rate_df)
unemploy_rate_scaled = scaler.transform(unemploy_rate_df)
unemploy_rate_df_scaled = pd.DataFrame(data=unemploy_rate_scaled)

economic_scaled_df = pd.concat([gdp_growth_df_scaled, unemploy_rate_df_scaled], axis = 1)
economic_cor = economic_df.corr()


plt.subplot(122)
plt.plot(gdp_growth_df.index, gdp_growth_df_scaled, color='black') # 표준화 된 지표 그래프
plt.plot(unemploy_rate_df.index, unemploy_rate_df_scaled, color='y')

plt.title("GDP_UNEMPLOY_SCALED (" + start[2:4] + "." + start[5:7]+ "~" + end[2:4] + "." + end[5:7] + ")", fontsize=20) 
plt.ylabel('figure', fontsize=12)
plt.xlabel('Date', fontsize=12)
plt.legend(['GDP', 'UNEMPLOY'], fontsize=12, loc='best')


### 상관관계 짧게 분석 (상관계수, p-value)
X = gdp_df.GDP.values 
Y = unemploy_df.Unemployment_Rate.values 
print(stats.pearsonr(X,Y))
print(stats.kendalltau(X,Y))
print(stats.spearmanr(X,Y))






sns.set(style="white")
f, ax = plt.subplots(figsize=(5, 5)) # 표준화 된 지표 상관분석 표
cmap = sns.diverging_palette(200, 10, as_cmap=True)
sns.heatmap(economic_cor, cmap = cmap, center=0.0, square=True,
            linewidths=0.5, cbar_kws={"shrink": 0.75}, annot=True)

plt.title("GDP_UNEMPLOY_CORR", size=15)
ax.set_xticklabels(list(economic_cor.columns), size=10, rotation=90)
ax.set_yticklabels(list(economic_cor.columns), size=10, rotation=0)
plt.show()

### 원본 그래프
# Line Graph by matplotlib with wide-form DataFrame
plt.subplot(121)
plt.plot(stock_df.index, stock_df.Close, color='r') # 원본 지표
plt.plot(gold_df.index, gold_df.Gold_Price, color='y')
plt.plot(fund_df.index, fund_df.Funds_Rate, color='b')
plt.plot(house_df.index, house_df.House_Price, color='g')
plt.plot(bond_df.index, bond_df.Bond_Close, color='m')

plt.title('ORIGINAL (' + start[2:4] + "." + start[5:7]+ "~" + end[2:4] + "." + end[5:7] + ")", fontsize=20) 
plt.ylabel('figure', fontsize=12)
plt.xlabel('Date', fontsize=12)
plt.legend(['Nasdaq', 'Gold', 'Fund', 'House', 'bond'], fontsize=12, loc='best')
# plt.show()

### 표준화(0~1) 된 지표의 그래프 및 상관관계 분석
stock_close_df = pd.DataFrame(data=stock_df.Close)
scaler.fit(stock_close_df)
stock_scaled = scaler.transform(stock_close_df)
stock_df_scaled = pd.DataFrame(data=stock_scaled)

gold_close_df = pd.DataFrame(data=gold_df.Gold_Price)
scaler.fit(gold_close_df)
gold_scaled = scaler.transform(gold_close_df)
gold_df_scaled = pd.DataFrame(data=gold_scaled)

fund_close_df = pd.DataFrame(data=fund_df.Funds_Rate)
scaler.fit(fund_close_df)
fund_scaled = scaler.transform(fund_close_df)
fund_df_scaled = pd.DataFrame(data=fund_scaled)

house_close_df = pd.DataFrame(data=house_df.House_Price)
scaler.fit(house_close_df)
house_scaled = scaler.transform(house_close_df)
house_df_scaled = pd.DataFrame(data=house_scaled)

bond_close_df = pd.DataFrame(data=bond_df.Bond_Close)
scaler.fit(bond_close_df)
bond_scaled = scaler.transform(bond_close_df)
bond_df_scaled = pd.DataFrame(data=bond_scaled)

gdp_close_df = pd.DataFrame(data=gdp_df.GDP)
scaler.fit(gdp_close_df)
gdp_scaled = scaler.transform(gdp_close_df)
gdp_df_scaled = pd.DataFrame(data=gdp_scaled)

snp_close_df = pd.DataFrame(data=snp_df.Snp_Close)
scaler.fit(snp_close_df)
snp_scaled = scaler.transform(snp_close_df)
snp_df_scaled = pd.DataFrame(data=snp_scaled)

# # print('종가들의 정규화 최소 값')
# # print(stock_df_scaled.min())
# # print('\n종가들의 정규화 최대 값')
# # print(stock_df_scaled.max())
# # print(stock_df_scaled)

### 표준화 된 주식 그래프 그리기

plt.subplot(122)
plt.plot(stock_df.index, stock_df_scaled, color='r') # 표준화 된 지표 그래프
plt.plot(gold_df.index, gold_df_scaled, color='y')
plt.plot(fund_df.index, fund_df_scaled, color='b')
plt.plot(house_df.index, house_df_scaled, color='g')
plt.plot(bond_df.index, bond_df_scaled, color='m')

plt.title("ORIGINAL_SCALED (" + start[2:4] + "." + start[5:7]+ "~" + end[2:4] + "." + end[5:7] + ")", fontsize=20) 
plt.ylabel('figure', fontsize=12)
plt.xlabel('Date', fontsize=12)
plt.legend(['Nasdaq', 'Gold', 'Fund', 'House', 'bond'], fontsize=12, loc='best')


corr_df = pd.concat([stock_df_scaled, gold_df_scaled, fund_df_scaled, house_df_scaled, bond_df_scaled], axis = 1) # 표준화 된 5지표 상관관계 분석
corr_df.columns = ['stock', 'gold', 'fund', 'house', 'bond']

corr_df = corr_df.dropna()
cor = corr_df.corr()

sns.set(style="white")
f, ax = plt.subplots(figsize=(5, 5)) # 표준화 된 지표 상관분석 표
cmap = sns.diverging_palette(200, 10, as_cmap=True)
sns.heatmap(cor, cmap = cmap, center=0.0, square=True,
            linewidths=0.5, cbar_kws={"shrink": 0.75}, annot=True)

plt.title("ORIGINAL_CORR (" + start[2:4] + "." + start[5:7]+ "~" + end[2:4] + "." + end[5:7] + ")", size=15)
ax.set_xticklabels(list(corr_df.columns), size=10, rotation=90)
ax.set_yticklabels(list(corr_df.columns), size=10, rotation=0)
plt.show()






### 수익률 그래프 및 상관관계 분석
types = ['Stock', 'Gold', 'Fund', 'House', 'Bond', 'SnP']
new_pd = pd.concat([stock_df, gold_df, fund_df, house_df, bond_df, snp_df], axis = 1)
new_pd.columns = [['Stock', 'Gold', 'Fund', 'House', 'Bond', 'SnP']] # 상관관계가 제대로 안되는지 확인 위한 SnP500 추가
# new_pd = pd.concat([stock_df, gold_df, fund_df, house_df, bond_df], axis = 1)
# new_pd.columns = [['Stock', 'Gold', 'Fund', 'House', 'Bond']]
new_pd = new_pd.dropna()

asset_rate_df = new_pd.pct_change() # 수익률 계산
asset_rate_df = asset_rate_df.dropna()
asset_rate_df = asset_rate_df * 100

# 수익률 그래프
# plt.subplot(121)
# if 'Stock' in types:
#     plt.plot(asset_rate_df.index, asset_rate_df['Stock'], color = 'r')
# if 'Gold' in types:
#     plt.plot(asset_rate_df.index, asset_rate_df['Gold'], color = 'y')
# if 'Fund' in types:
#     plt.plot(asset_rate_df.index, asset_rate_df['Fund'], color = 'b')
# if 'House' in types:
#     plt.plot(asset_rate_df.index, asset_rate_df['House'], color = 'g')
# if 'Bond' in types:
#     plt.plot(asset_rate_df.index, asset_rate_df['Bond'], color = 'm')
# if 'SnP' in types:
#     plt.plot(asset_rate_df.index, asset_rate_df['SnP'], color = 'cyan')

# plt.title("PROFIT (" + start[2:4] + "." + start[6]+ "~" + end[2:4] + "." + end[6] + ")", fontsize=20) 
# plt.ylabel('figure', fontsize=12)
# plt.xlabel('Date', fontsize=12)
# plt.legend(types, fontsize=12, loc='best')

# # 수익률 상관관계 표
# asset_rate_corr_df = asset_rate_df.corr(method = 'pearson') # 상관관계 계산
# sns.set(style="white")
# f, ax = plt.subplots(figsize=(5, 5))
# cmap = sns.diverging_palette(200, 10, as_cmap=True)
# sns.heatmap(asset_rate_corr_df, cmap = cmap, square = True, annot = True, fmt = '.2f', 
#             linewidths = .5, cbar_kws={"shrink": .5})
# plt.title("PROFIT_CORR (" + start[2:4] + "." + start[6]+ "~" + end[2:4] + "." + end[6] + ")", fontsize = 15)
# plt.show()

### 수익률 정규화 그래프 및 상관관계 분석
plot = []
if 'Stock' in types:
    stock_profit_df = pd.DataFrame(data=asset_rate_df.Stock) # 정규화(minmax)
    scaler.fit(stock_profit_df)
    stock_profit_scaled = scaler.transform(stock_profit_df)
    stock_profit_scaled = pd.DataFrame(data=stock_profit_scaled)
if 'Gold' in types:
    gold_profit_df = pd.DataFrame(data=asset_rate_df.Gold)
    scaler.fit(gold_profit_df)
    gold_profit_scaled = scaler.transform(gold_profit_df)
    gold_profit_scaled = pd.DataFrame(data=gold_profit_scaled)
if 'Fund' in types:
    fund_profit_df = pd.DataFrame(data=asset_rate_df.Fund)
    scaler.fit(fund_profit_df)
    fund_profit_scaled = scaler.transform(fund_profit_df)
    fund_profit_scaled = pd.DataFrame(data=fund_profit_scaled)
if 'House' in types:
    house_profit_df = pd.DataFrame(data=asset_rate_df.House)
    scaler.fit(house_profit_df)
    house_profit_scaled = scaler.transform(house_profit_df)
    house_profit_scaled = pd.DataFrame(data=house_profit_scaled)
if 'Bond' in types:
    bond_profit_df = pd.DataFrame(data=asset_rate_df.Bond)
    scaler.fit(bond_profit_df)
    bond_profit_scaled = scaler.transform(bond_profit_df)
    bond_profit_scaled = pd.DataFrame(data=bond_profit_scaled)
if 'SnP' in types:
    snp_profit_df = pd.DataFrame(data=asset_rate_df.SnP)
    scaler.fit(snp_profit_df)
    snp_profit_scaled = scaler.transform(snp_profit_df)
    snp_profit_scaled = pd.DataFrame(data=snp_profit_scaled)

# plt.subplot(122)
if 'Stock' in types:
    plt.plot(stock_profit_df.index, stock_profit_scaled, color = 'r')
    plot.append(stock_profit_scaled)
if 'Gold' in types:
    plt.plot(gold_profit_df.index, gold_profit_scaled, color = 'y')
if 'Fund' in types:
    plt.plot(fund_profit_df.index, fund_profit_scaled, color = 'b')
if 'House' in types:
    plt.plot(house_profit_df.index, house_profit_scaled, color = 'g')
if 'Bond' in types:
    plt.plot(bond_profit_df.index, bond_profit_scaled, color = 'm')
if 'SnP' in types:
    plt.plot(snp_profit_df.index, snp_profit_scaled, color = 'cyan')


plt.title("PROFIT_SCALED (" + start[2:4] + "." + start[5:7]+ "~" + end[2:4] + "." + end[5:7] + ")", fontsize=20) # 정규화 된 수익률 그래프
plt.ylabel('figure', fontsize=12)
plt.xlabel('Date', fontsize=12)
plt.legend(types, fontsize=12, loc='best')
# plt.show()

# 수익률 상관관계 표
asset_rate_corr_df = asset_rate_df[types].corr(method = 'pearson') # 상관관계 계산
sns.set(style="white")
f, ax = plt.subplots(figsize=(5, 5))
cmap = sns.diverging_palette(200, 10, as_cmap=True)
sns.heatmap(asset_rate_corr_df, cmap = cmap, square = True, annot = True, fmt = '.2f', 
            linewidths = .5, cbar_kws={"shrink": .5})
plt.title("PROFIT_CORR (" + start[2:4] + "." + start[5:7]+ "~" + end[2:4] + "." + end[5:7] + ")", fontsize = 15)
plt.show()

# asset_rate_scaled_df = pd.concat([stock_profit_scaled, gold_profit_scaled, fund_profit_scaled, house_profit_scaled, bond_profit_scaled, snp_profit_scaled], axis = 1)
# asset_rate_scaled_df.columns = [['Stock', 'Gold', 'Fund', 'House', 'Bond', 'SnP']] # 상관관계가 제대로 안되는지 확인 위한 SnP500 추가

# asset_rate_scaled_corr_df = asset_rate_scaled_df.corr(method = 'pearson') # 상관관계 적용

# sns.set(style="white")
# f, ax = plt.subplots(figsize=(5, 5)) # 수익률 상관관계 표
# cmap = sns.diverging_palette(200, 10, as_cmap=True)
# sns.heatmap(asset_rate_scaled_corr_df, cmap = cmap, square = True, annot = True, fmt = '.2f', 
#             linewidths = .5, cbar_kws={"shrink": .5})
# plt.title("PROFIT_MINMAX_CORR", fontsize = 23)
# plt.show()
