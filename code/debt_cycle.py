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

### 열 이름 수정
debt_df = debt_df.rename(columns={'Debt to GDP':'Debt'})
interest_df = interest_df.rename(columns={'FEDFUNDS':'Interest_Rate', 'DATE':'Date'})

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

### index를 날짜로 변경하기
debt_df = debt_df.set_index('Date')
interest_df = interest_df.set_index('Date')

### 결측치(NaN) 삭제
debt_df = debt_df.dropna(axis = 1)

### 상관분석을 위한 데이터 전처리
debt_interest_df = pd.concat([debt_df, interest_df], axis = 1)
debt_interest_df.columns = ['Debt', 'y', 'Cycle', 'Interest_Rate']
debt_interest_df = debt_interest_df.dropna(axis = 0)
debt_interest_corr_df = debt_interest_df.drop(['y', 'Cycle'], axis = 1)
print(debt_interest_corr_df)

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

span_start = []
span_end = []
span_start.append(datetime(1940, 1, 1))
span_start.append(datetime(1929, 1, 1))
span_start.append(datetime(1861, 1, 1))
span_start.append(datetime(1914, 1, 1))
span_start.append(datetime(1980, 1, 1))
span_start.append(datetime(2020, 1, 1))
span_end.append(datetime(1945, 1, 1))
span_end.append(datetime(1939, 1, 1))
span_end.append(datetime(1865, 1, 1))
span_end.append(datetime(1919, 1, 1))
span_end.append(datetime(1982, 1, 1))
span_end.append(datetime(2022, 1, 1))

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

### 최종 사이클
def Cycle():
    ### 사이클 1
    plt.title('Dept - productivity', fontsize=15) # 제목
    
    plt.plot(debt_df.index, debt_df.Cycle, color='r', linewidth = 3) # x축, y축, 각 데이터의 이름
    
    plt.hlines(0, debt_df.index[0], debt_df.index[len(debt_df)-1], color='black', linewidth=1)
    
    for i in range(len(span_start)):
        plt.axvspan(span_start[i], span_end[i], facecolor='gray', alpha=0.5)  
    
    plt.ylabel('부채비율 - 생산률', fontsize=12)
    plt.legend(['부채비율 - 생산률, 이동평균선'], fontsize=12, loc='best')
    plt.grid()
    plt.show()
    
    # left side
    fig, ax1 = plt.subplots()
    color_1 = 'tab:blue'
    ax1.set_title('Dept - productivity', fontsize=16)
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Dept - productivity', fontsize=14, color=color_1)
    ax1.plot(debt_interest_df.index, debt_interest_df.Cycle, color=color_1)
    for i in range(len(span_start)):
        if span_start[i] > debt_interest_df.index[0]:
            plt.axvspan(span_start[i], span_end[i], facecolor='gray', alpha=0.5)
    plt.hlines(0, debt_interest_df.index[0], debt_interest_df.index[len(debt_interest_df)-1], color='black', linewidth=1)
    ax1.tick_params(axis='y', labelcolor=color_1)
    plt.legend(['부채비율-생산률'], fontsize=12, loc='best')

    # right side with different scale
    ax2 = ax1.twinx() # instantiate a second axes that shares the same x-axis
    color_2 = 'tab:red'
    ax2.set_ylabel('Interest Rate(%)', fontsize=14, color=color_2)
    ax2.plot(debt_interest_df.index, debt_interest_df.Interest_Rate, color=color_2)
    ax2.tick_params(axis='y', labelcolor=color_2)

    fig.tight_layout()
    plt.legend(['금리'], fontsize=12, loc='best')
    
    plt.grid()
    plt.show()
    
    # ### 사이클 2           ------------------------------------ 폐기 ------------------------------------            ###
    # plt.subplot(212)
    # plt.title('Dept - 120 average line', fontsize=15) # 제목
    
    # plt.plot(debt_df.Date, debt_df.Debt - debt_df['MA120'], color='r', linewidth = 3) # x축, y축, 각 데이터의 이름
    
    # plt.hlines(0, debt_df.loc[0, 'Date'], debt_df.loc[len(debt_df)-1, 'Date'], color='black', linewidth=1)
    
    # plt.ylabel('부채비율 - 이동평균선', fontsize=12)
    # plt.xlabel('Date', fontsize=12)
    # plt.legend(['부채비율 - 이동평균선'], fontsize=12, loc='best')
    
    # plt.grid()
    # plt.show()
    
    # ### 이중축 그래프 그리기        ------------------------------------ 폐기 ------------------------------------            ###
    # # left side
    # fig, ax1 = plt.subplots()
    # color_1 = 'tab:blue'
    # ax1.set_title('Debt as % of GDP & Interest', fontsize=16)
    # ax1.set_xlabel('Date')
    # ax1.set_ylabel('Debt Rate(%)', fontsize=14, color=color_1)
    # ax1.plot(debt_df.index, debt_df.Debt, color=color_1)
    # ax1.tick_params(axis='y', labelcolor=color_1)
    # plt.legend(['부채'], fontsize=12, loc='best')

    # # right side with different scale
    # ax2 = ax1.twinx() # instantiate a second axes that shares the same x-axis
    # color_2 = 'tab:red'
    # ax2.set_ylabel('Interest Rate(%)', fontsize=14, color=color_2)
    # ax2.bar(interest_df.index, interest_df.Interest_Rate, color=color_2)
    # ax2.tick_params(axis='y', labelcolor=color_2)

    # fig.tight_layout()
    # plt.legend(['금리'], fontsize=12, loc='best')

Graph()
Cycle()

def heatmap():
    cor = debt_interest_corr_df.corr()
    print(cor)

    sns.set(style="white")
    f, ax = plt.subplots(figsize=(5, 5))
    cmap = sns.diverging_palette(200, 10, as_cmap=True)
    sns.heatmap(cor, cmap = cmap, center=0.0, square=True,
                linewidths=0.5, cbar_kws={"shrink": 0.75}, annot=True)

    plt.title('Debt and Interest', size=10)
    ax.set_xticklabels(list(debt_interest_corr_df.columns), size=8, rotation=90)
    ax.set_yticklabels(list(debt_interest_corr_df.columns), size=8, rotation=0)

    plt.show()
heatmap()