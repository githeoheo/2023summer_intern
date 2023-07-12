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

# CSV 파일 불러오기
stock_df = pd.read_csv('C:/Users/user/Desktop/학기별 문서/현장실습/real_price/나스닥(1985~2023)_yfinance.csv')
gold_df = pd.read_csv('C:/Users/user/Desktop/학기별 문서/현장실습/real_price/금(1950~2023)_캐글.csv')
interest_df = pd.read_csv('C:/Users/user/Desktop/학기별 문서/현장실습/real_price/미국금리(1954.7~2023.5)_구글서치.csv')
house_df = pd.read_csv('C:/Users/user/Desktop/학기별 문서/현장실습/real_price/케이스-쉴러_미국주택가격지수(1987.1~2023.4).csv')
bond_df = pd.read_csv('C:/Users/user/Desktop/학기별 문서/현장실습/real_price/10년만기 미국채 선물 과거 데이터.csv')
gdp_df = pd.read_csv('C:/Users/user/Desktop/학기별 문서/현장실습/real_price/1인당GDP.csv')

# 데이터프레임 컬럼 이름 바꾸기
gold_df = gold_df.rename(columns={'Price USD per Oz':'Gold_Price'})
interest_df = interest_df.rename(columns={'FEDFUNDS':'Funds_Rate', "DATE":"Date"})
house_df = house_df.rename(columns={'SPCS10RSA':'House_Price', "DATE":"Date"})
bond_df = bond_df.rename(columns={'날짜':'Date', '종가':'Bond_Close'})

# 불러올 날짜 구간 설정(공통 1987-01-01 ~ 2023-07-01)
start = "1987-01-01" # 최소 1950-01-01
end = "2023-06-01" # 최대 2023-07-01
stock_df = stock_df[stock_df['Date'].between(start, end)]
gold_df = gold_df[gold_df['Date'].between(start, end)]
interest_df = interest_df[interest_df['Date'].between(start, end)]
house_df = house_df[house_df['Date'].between(start, end)]
bond_df = bond_df[bond_df['Date'].between(start, end)]
gdp_df = gdp_df[gdp_df['Date'].between(start, end)]

# 금리를 가격화하기
interest_df = interest_df.reset_index(drop=True) # 날짜 자르면서 index도 같이 잘림 -> index를 0으로 초기화 시키기

interest_df['Funds_Rate'] = interest_df['Funds_Rate'] / 100 + 1 # 금리를 가격화 하기 위해 공식을 적용하여 값 변환 => 현재금리 = {(현재금리 / 100) + 1}

for i in range(len(interest_df)-1): # result = 현재 값 * 다음 값 => 복리 계산 / 기준값 = 1달러 가정
    interest_df.loc[i+1,'Funds_Rate'] = interest_df.loc[i,'Funds_Rate'] * interest_df.loc[i+1, 'Funds_Rate']

# 날짜 datatime 형식으로 전환하기
stock_df.loc[:,'Date'] = pd.to_datetime(stock_df.Date)
gold_df.loc[:,'Date'] = pd.to_datetime(gold_df.Date)
interest_df.loc[:,'Date'] = pd.to_datetime(interest_df.Date)
house_df.loc[:,'Date'] = pd.to_datetime(house_df.Date)
bond_df.loc[:,'Date'] = pd.to_datetime(bond_df.Date)
gdp_df.loc[:,'Date'] = pd.to_datetime(gdp_df.Date)

# index를 날짜로 변경하기
stock_df = stock_df.set_index('Date')
gold_df = gold_df.set_index('Date')
interest_df = interest_df.set_index('Date')
house_df = house_df.set_index('Date')
bond_df = bond_df.set_index('Date')
gdp_df = gdp_df.set_index('Date')

# 특정 칼럼(열) 삭제
stock_df = stock_df.drop(['Open', 'High', 'Low', 'Adj Close', 'Volume'], axis = 1)
bond_df = bond_df.drop(['시가', '고가', '저가', '변동 %'], axis = 1)

# 지수 값을 실제 가격(달러)으로 전환하기
house_df['House_Price'] = 130000 * (house_df['House_Price'] * 0.01)



######------------------------------------------------------- 데이터 전처리 -----------------------------------------------------------unt)

# --------------- 결측치 ---------------- #

# 선형보간법으로 빈 행 채우기(method = value : 선형 / method = time : 시간)
gdp_df = gdp_df.interpolate(method = "time")

# 결측치가 있는 행 or 열 제거하는 함수 (axis = 0 : 행 / axis = 1 : 열 / 생략 시 0 디폴트)
stock_df = stock_df.dropna(axis=0)
gold_df = gold_df.dropna(axis=0)
interest_df = interest_df.dropna(axis=0)
house_df = house_df.dropna(axis=0)
bond_df = bond_df.dropna(axis=0)
gdp_df = gdp_df.dropna(axis=1)  # 열 이름 없는데도 값이 비어있는 열들이 있어서 추가 삭제
gdp_df = gdp_df.dropna(axis=0)

# --------------- 중복치 ---------------- #
#--practice.py 참고--#
# --------------- 이상치 ---------------- #
#--practice.py 참고--#

#######---------------------------------------- 그래프그리기 ----------------------------------------------------######
# 전역으로 그래프 사이즈 고정
plt.rcParams["figure.figsize"] = (16,8)

# 유니코드 깨짐현상 해결
plt.rcParams['axes.unicode_minus'] = False
 
# 나눔고딕 폰트 적용
plt.rcParams["font.family"] = 'NanumGothic'

# 그래프 타이틀
plt.title('GDP & 5-asset (' + start[2:4] + "." + start[5:7]+ "~" + end[2:4] + "." + end[5:7] + ")", fontsize=20) 

# 데이터 주입
plt.plot(gdp_df.index, gdp_df.GDP, color='black', linewidth = 3)
plt.plot(stock_df.index, stock_df.Close, color='r', linewidth = 3)
plt.plot(gold_df.index, gold_df.Gold_Price, color='y', linewidth = 3)
plt.plot(house_df.index, house_df.House_Price, color='b', linewidth = 3)
plt.plot(interest_df.index, interest_df.Funds_Rate, color='g', linewidth = 3)
plt.plot(bond_df.index, bond_df.Bond_Close, color='m', linewidth = 3)

# x축, y축, 각 데이터의 이름 설정
plt.ylabel('$', fontsize=12)
plt.xlabel('Date', fontsize=12)
plt.legend(['GDP', '주가', '금', '부동산', '금리', '채권'], fontsize=12, loc='best')

# 상관분석을 한번에 하기 위해 5가지 지표 합치기
economic_df = pd.concat([gdp_df, stock_df, gold_df, house_df, interest_df, bond_df], axis = 1)

# 합친 데이터프레임의 열 이름 변환
economic_df.columns = ['GDP', '주가', '금', '부동산', '금리', '채권']

# 상관관계 분석 함수 사용
economic_cor = economic_df.corr()

# heatmap 설정 후 그리기
sns.set(style="white", font="Malgun Gothic", rc={"axes.unicode_minus":False})
f, ax = plt.subplots(figsize=(5, 5)) # 표준화 된 지표 상관분석 표
cmap = sns.diverging_palette(200, 10, as_cmap=True)
sns.heatmap(economic_cor, cmap = cmap, center=0.0, square=True,
            linewidths=0.5, cbar_kws={"shrink": 0.75}, annot=True)

# heatmap 타이틀과 x축, y축 이름 설정
plt.title("GDP & 5-asset (" + start[2:4] + "." + start[5:7]+ "~" + end[2:4] + "." + end[5:7] + ")", size=15)
ax.set_xticklabels(list(economic_cor.columns), size=10, rotation=90)
ax.set_yticklabels(list(economic_cor.columns), size=10, rotation=0)

# 그래프, heatmap 동시 출력
plt.show()

