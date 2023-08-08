import pandas as pd
pd.options.plotting.backend = "plotly"
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
from datetime import datetime

### 파일 불러오기
debt_df = pd.read_csv('C:/Users/user/Desktop/학기별 문서/현장실습/데이터자료/dept_gdp_y-value.csv')
interest_df = pd.read_csv('C:/Users/user/Desktop/학기별 문서/현장실습/데이터자료/미국금리(1954.7~2023.5)_구글서치.csv')
average_df = pd.read_csv('C:/Users/user/Desktop/학기별 문서/현장실습/데이터자료/금리-평균.csv')
stock_df = pd.read_csv('C:/Users/user/Desktop/학기별 문서/현장실습/real_price/나스닥(1985~2023)_yfinance.csv')
bond_df = pd.read_csv('C:/Users/user/Desktop/학기별 문서/현장실습/real_price/10년만기 미국채 선물 과거 데이터.csv')
gold_df = pd.read_csv('C:/Users/user/Desktop/학기별 문서/현장실습/real_price/금(1950~2023)_캐글.csv')

### 열 이름 수정
debt_df = debt_df.rename(columns={'Debt to GDP':'Debt'})
interest_df = interest_df.rename(columns={'FEDFUNDS':'Interest_Rate', 'DATE':'Date'})
bond_df = bond_df.rename(columns={'종가':'Bond_Close', "날짜":"Date"})
gold_df = gold_df.rename(columns={'Price USD per Oz':'Gold_Price'})

### 결측치 보간
debt_df = debt_df.interpolate(method='values')
debt_df = debt_df.drop(['GDP'], axis = 1)

### 필요한 열 생성
debt_df['Cycle'] = debt_df['Debt'] - debt_df['y']

### 분석 날짜 지정
start = "1790-01-01" # 최소 1790-01-01
end = "2023-01-01" # 최대 2023-01-01
debt_df = debt_df[debt_df['Date'].between(start,end)]

### 날짜 타입 변경
debt_df.loc[:,'Date'] = pd.to_datetime(debt_df.Date)
interest_df.loc[:,'Date'] = pd.to_datetime(interest_df.Date)
average_df.loc[:,'Date'] = pd.to_datetime(average_df.Date)
stock_df.loc[:,'Date'] = pd.to_datetime(stock_df.Date)
bond_df.loc[:,'Date'] = pd.to_datetime(bond_df.Date)
gold_df.loc[:,'Date'] = pd.to_datetime(gold_df.Date)

### index를 날짜로 변경하기
debt_df = debt_df.set_index('Date')
interest_df = interest_df.set_index('Date')
average_df = average_df.set_index('Date')
stock_df = stock_df.set_index('Date')
bond_df = bond_df.set_index('Date')
gold_df = gold_df.set_index('Date')

### 결측치(NaN) 및 필요없는 열 삭제
debt_df = debt_df.dropna(axis = 1)
stock_df = stock_df.drop(['Open', 'High', 'Low', 'Adj Close', 'Volume'], axis=1)
bond_df = bond_df.drop(['시가', '고가', '저가', '변동 %'], axis = 1)

### 시각화에 필요한 데이터 전처리 후 데이터프레임 생성
debt_average_df = pd.concat([debt_df, average_df], axis=1)
debt_average_df.columns = ['Debt', 'y', 'Cycle', 'Average_Cycle']
debt_average_df = debt_average_df.drop(['Debt', 'y'], axis = 1)
debt_average_df = debt_average_df.dropna(axis=0)
debt_average_df['Average_Cycle'] = (debt_average_df['Average_Cycle'] / debt_average_df['Average_Cycle'].abs().max()) * debt_df['Cycle'].abs().max()

debt_interest_df = pd.concat([debt_df, interest_df], axis=1)
debt_interest_df.columns = ['Debt', 'y', 'Cycle', 'Interest_Rate']
debt_interest_df = debt_interest_df.drop(['Debt', 'y'], axis=1)
debt_interest_df = debt_interest_df.dropna(axis=0)

debt_stock_df = pd.concat([debt_df, stock_df], axis=1)
debt_stock_df.columns = ['Debt', 'y', 'Cycle', 'Close']
debt_stock_df = debt_stock_df.drop(['Debt', 'y'], axis=1)
debt_stock_df = debt_stock_df.dropna(axis=0)

debt_bond_df = pd.concat([debt_df, bond_df], axis=1)
debt_bond_df.columns = ['Debt', 'y', 'Cycle', 'Bond_Close']
debt_bond_df = debt_bond_df.drop(['Debt', 'y'], axis=1)
debt_bond_df = debt_bond_df.dropna(axis=0)

debt_gold_df = pd.concat([debt_df, gold_df], axis=1)
debt_gold_df.columns = ['Debt', 'y', 'Cycle', 'Gold_Price']
debt_gold_df = debt_gold_df.drop(['Debt', 'y'], axis=1)
debt_gold_df = debt_gold_df.dropna(axis=0)

debt_asset_df = pd.concat([debt_df, interest_df, stock_df, bond_df, gold_df], axis=1)
debt_asset_df.columns = ['Debt', 'y', 'Debt_Cycle', 'Interest_Rate', 'Stock_Close', 'Bond_Close', 'Gold_Price']
debt_asset_df = debt_asset_df.drop(['Debt', 'y'], axis=1)
debt_asset_df = debt_asset_df.dropna(axis=0)


### 부채 사이클의 여러 상승구간 구하기 VERSION_1
def mean(df, i, num):
    total = 0
    for j in range(num):
        total = total + df.iloc[i+j].Cycle
    return total / num

def outlier(df, num):
    list = []
    for i in range(len(df) - num): # mean 함수 위에 정의
        if(mean(df, i, num) < df.iloc[i+num].Cycle and df.iloc[i+num-1].Cycle < df.iloc[i+num].Cycle):
            list.append(df.index[i+num])
    return list

outlier_time = outlier(debt_df, 8) # 상승구간에 포함된 datetime 인덱스를 반환받음

### 부채 사이클의 여러 상승구간 구하기 VERSION_2
# 계절적 성분 50일로 가정
# extrapolate_trend='freq' : Trend 성분을 만들기 위한 rolling window 때문에 필연적으로 trend, resid에는 Nan 값이 발생하기 때문에, 이 NaN값을 채워주는 옵션이다.
result = seasonal_decompose(debt_df.Cycle, model='additive', two_sided=True, 
                            period=50, extrapolate_trend='freq') 
result.plot()

result.seasonal[:100].plot()

# Residual의 분포 확인
fig, ax = plt.subplots(figsize=(9,6))
_ = plt.hist(result.resid, 100, density=True, alpha=0.75)
plt.show()

r = result.resid.values
st, p = ztest(r)
print(st,p)

# 평균과 표준편차 출력
mu, std = result.resid.mean(), result.resid.std()
print("평균:", mu, "표준편차:", std)
# 평균: -0.3595321143716522 표준편차: 39.8661527194307

# n-sigma(표준편차)를 기준으로 이상치 판단
print("이상치 갯수:", len(result.resid[(result.resid>mu+3*std)|(result.resid<mu-3*std)]))
print(result.resid[(result.resid>mu+3*std)|(result.resid<mu-3*std)])

outlier_date = result.resid[(result.resid>mu+3*std)|(result.resid<mu-3*std)].index # residual 의 정규분포 생성 -> 이상치에 해당하는 구간의 datetime index을 반환


# ---------------------------------------------------- 시각화 -------------------------------------------------------- #

### 그래프 세팅
def Setting():
    # 전역으로 그래프 사이즈 고정
    plt.rcParams["figure.figsize"] = (16,8)

    # 유니코드 깨짐현상 해결
    plt.rcParams['axes.unicode_minus'] = False
    
    # 나눔고딕 폰트 적용
    plt.rcParams["font.family"] = 'NanumGothic'
Setting()

# 상승 구간에 해당하는 미국 대표 사건들 정리 후 구간 시각화에 사용 -> Cycle() 에 적용
def history():
    his_start = [] # 시작지점과 끝지점을 리스트에 저장 후 반환
    his_end = []
    his_start.append(datetime(1812, 1, 1)) # 미영전쟁
    his_start.append(datetime(1846, 1, 1)) # 멕시코전쟁
    his_start.append(datetime(1861, 1, 1)) # 남북전쟁
    his_start.append(datetime(1898, 1, 1)) # 스페인전쟁
    his_start.append(datetime(1914, 1, 1)) # 1차세계대전
    his_start.append(datetime(1929, 1, 1)) # 대공황
    his_start.append(datetime(1940, 1, 1)) # 2차세계대전
    his_start.append(datetime(1980, 1, 1)) # 경제회복 -> 세금인하 국방비증가
    his_start.append(datetime(1987, 1, 1)) # 주식폭락
    his_start.append(datetime(2019, 1, 1)) # COVID-19
    his_end.append(datetime(1815, 1, 1))
    his_end.append(datetime(1848, 1, 1))
    his_end.append(datetime(1865, 1, 1))
    his_end.append(datetime(1899, 1, 1))  
    his_end.append(datetime(1919, 1, 1))
    his_end.append(datetime(1939, 1, 1))
    his_end.append(datetime(1945, 1, 1))
    his_end.append(datetime(1986, 1, 1))
    his_end.append(datetime(1988, 1, 1))
    his_end.append(datetime(2022, 1, 1))
    return his_start, his_end
span_start, span_end = history()

### 초기 그래프
def Graph():
    # 단일축 그래프
    plt.title('Dept to GDP', fontsize=20) # 제목
    plt.plot(debt_df.index, debt_df.Debt, color='r', linewidth = 3) # x축, y축, 각 데이터의 이름
    
    plt.plot(debt_df.index, debt_df.y, color='black')
    ma120 = debt_df['Debt'].rolling(window=120).mean()
    debt_df.insert(len(debt_df.columns), "MA120", ma120)
    
    plt.plot(debt_df.index, debt_df['MA120'])
    
    for i in range(len(span_start)):
        plt.axvspan(span_start[i], span_end[i], facecolor='gray', alpha=0.5)  
    
    plt.ylabel('Dept to GDP', fontsize=12)
    plt.xlabel('Date', fontsize=12)
    plt.legend(['GDP 대비 연방부채', '성장률', '120이동평균선'], fontsize=12, loc='best')
    print(debt_df.head)
    plt.grid()
    plt.show()
# Graph()

### 최종 사이클
def Cycle():
    ### 그래프 1 : figure.부채 사이클
    plt.title('Dept - productivity', fontsize=15) # 제목
    
    plt.plot(debt_df.index, debt_df.Cycle, color='r', linewidth = 3) # x축, y축, 각 데이터의 이름
    plt.scatter(outlier_time, debt_df.loc[outlier_time].Cycle, c = 'red', edgecolor = 'blue', s = 50)
    
    plt.hlines(0, debt_df.index[0], debt_df.index[len(debt_df)-1], color='black', linewidth=1)
    
    for i in range(len(span_start)):
        plt.axvspan(span_start[i], span_end[i], facecolor='gray', alpha=0.5)  
    
    plt.ylabel('부채비율 - 생산률', fontsize=12)
    plt.legend(['부채비율-생산률 사이클'], fontsize=12, loc='best')
    plt.grid()
    plt.show()
    
    ### 사이클 2 : figure. 부채 사이클, 금리 사이클
    plt.title('Dept Cycle & Average Cycle', fontsize=15) # 제목
    
    plt.plot(debt_average_df.index, debt_average_df.Cycle, color='r', linewidth = 3) # x축, y축, 각 데이터의 이름
    
    plt.hlines(0, debt_average_df.index[0], debt_average_df.index[len(debt_average_df)-1], color='black', linewidth=1)
    
    plt.plot(debt_average_df.index, debt_average_df.Average_Cycle, color='b', linewidth = 3) # x축, y축, 각 데이터의 이름
    
    # for i in range(len(span_start)):
    #     plt.axvspan(span_start[i], span_end[i], facecolor='gray', alpha=0.5)  
    
    plt.ylabel('%', fontsize=12)
    plt.legend(['부채비율-생산률 사이클', '금리-평균 사이클'], fontsize=12, loc='best')
    plt.grid()
    plt.show()
    
    ### 그래프3 : figure.부채 사이클, 금리
    # left side
    fig, ax1 = plt.subplots()
    color_1 = 'tab:red'
    ax1.set_title('Dept Cycle & Interest Rate', fontsize=16)
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Dept Cycle(%)', fontsize=14, color=color_1)
    ax1.plot(debt_interest_df.index, debt_interest_df.Cycle, color=color_1)
    for i in range(len(span_start)):
        if span_start[i] > debt_interest_df.index[0]:
            plt.axvspan(span_start[i], span_end[i], facecolor='gray', alpha=0.5)
    plt.hlines(0, debt_interest_df.index[0], debt_interest_df.index[len(debt_interest_df)-1], color='black', linewidth=1)
    ax1.tick_params(axis='y', labelcolor=color_1)
    plt.legend(['부채비율-생산률'], fontsize=12, loc='best')

    # right side with different scale
    ax2 = ax1.twinx() # instantiate a second axes that shares the same x-axis
    color_2 = 'tab:blue'
    ax2.set_ylabel('Interest Rate(%)', fontsize=14, color=color_2)
    ax2.plot(debt_interest_df.index, debt_interest_df.Interest_Rate, color=color_2)
    ax2.tick_params(axis='y', labelcolor=color_2)

    fig.tight_layout()
    plt.legend(['금리'], fontsize=12, loc='best')
    
    plt.grid()
    plt.show()
    
    ### 그래프4 : figure.부채 사이클, 나스닥
    # left side
    fig, ax1 = plt.subplots()
    color_1 = 'tab:red'
    ax1.set_title('Dept Cycle & Nasdaq Rate', fontsize=16)
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Dept Cycle(%)', fontsize=14, color=color_1)
    ax1.plot(debt_stock_df.index, debt_stock_df.Cycle, color=color_1)
    plt.hlines(0, debt_stock_df.index[0], debt_stock_df.index[len(debt_stock_df)-1], color='black', linewidth=1)
    ax1.tick_params(axis='y', labelcolor=color_1)
    plt.legend(['부채비율-생산률'], fontsize=12, loc='best')

    # right side with different scale
    ax2 = ax1.twinx() # instantiate a second axes that shares the same x-axis
    color_2 = 'tab:blue'
    ax2.set_ylabel('Stock Close', fontsize=14, color=color_2)
    ax2.plot(debt_stock_df.index, debt_stock_df.Close, color=color_2)
    ax2.tick_params(axis='y', labelcolor=color_2)

    fig.tight_layout()
    plt.legend(['나스닥'], fontsize=12, loc='best')
    
    plt.grid()
    plt.show()
    
    ### 그래프4 : figure.부채 사이클, 채권
    # left side
    fig, ax1 = plt.subplots()
    color_1 = 'tab:red'
    ax1.set_title('Dept Cycle & Bond', fontsize=16)
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Dept Cycle(%)', fontsize=14, color=color_1)
    ax1.plot(debt_bond_df.index, debt_bond_df.Cycle, color=color_1)
    plt.hlines(0, debt_bond_df.index[0], debt_bond_df.index[len(debt_bond_df)-1], color='black', linewidth=1)
    ax1.tick_params(axis='y', labelcolor=color_1)
    plt.legend(['부채비율-생산률'], fontsize=12, loc='best')

    # right side with different scale
    ax2 = ax1.twinx() # instantiate a second axes that shares the same x-axis
    color_2 = 'tab:blue'
    ax2.set_ylabel('Stock Close', fontsize=14, color=color_2)
    ax2.plot(debt_bond_df.index, debt_bond_df.Bond_Close, color=color_2)
    ax2.tick_params(axis='y', labelcolor=color_2)

    fig.tight_layout()
    plt.legend(['채권'], fontsize=12, loc='best')
    
    plt.grid()
    plt.show()
    
    ### 그래프4 : figure.부채 사이클, 금
    # left side
    fig, ax1 = plt.subplots()
    color_1 = 'tab:red'
    ax1.set_title('Dept Cycle & Gold', fontsize=16)
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Dept Cycle(%)', fontsize=14, color=color_1)
    ax1.plot(debt_gold_df.index, debt_gold_df.Cycle, color=color_1)
    plt.hlines(0, debt_gold_df.index[0], debt_gold_df.index[len(debt_gold_df)-1], color='black', linewidth=1)
    ax1.tick_params(axis='y', labelcolor=color_1)
    plt.legend(['부채비율-생산률'], fontsize=12, loc='best')

    # right side with different scale
    ax2 = ax1.twinx() # instantiate a second axes that shares the same x-axis
    color_2 = 'tab:blue'
    ax2.set_ylabel('Gold Price', fontsize=14, color=color_2)
    ax2.plot(debt_gold_df.index, debt_gold_df.Gold_Price, color=color_2)
    ax2.tick_params(axis='y', labelcolor=color_2)

    fig.tight_layout()
    plt.legend(['금'], fontsize=12, loc='best')
    
    plt.grid()
    plt.show()
Cycle()

### 상관분석
def heatmap():
    cor = debt_asset_df.corr()
    print(cor)

    sns.set(style="white")
    f, ax = plt.subplots(figsize=(5, 5))
    cmap = sns.diverging_palette(200, 10, as_cmap=True)
    sns.heatmap(cor, cmap = cmap, center=0.0, square=True,
                linewidths=0.5, cbar_kws={"shrink": 0.75}, annot=True)

    plt.title('Debt with Asset', size=10)
    ax.set_xticklabels(list(debt_asset_df.columns), size=8, rotation=90)
    ax.set_yticklabels(list(debt_asset_df.columns), size=8, rotation=0)

    plt.show()
heatmap()
