## Module Importing and Aliasing ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ

# Data Analysis and Visualization
import pandas                as pd
import numpy                 as np
from   sklearn.preprocessing import MinMaxScaler
import talib

# Visualization
from   matplotlib           import font_manager, rc
import matplotlib.pyplot    as plt
import seaborn              as sns
import mplfinance           as mpf
import plotly.graph_objects as go
import plotly.subplots      as ms

# Time Handling
import time
from   pytz     import timezone
from   datetime import date, datetime, timedelta

# HTTP Requests
from urllib.request import urlopen

# Data Sources
from   pykrx             import stock, bond
import pandas_datareader as pdr

# Configurations
import warnings


## Configurations ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
plt.rcParams['font.family'] = 'Malgun Gothic' # 한글 폰트 설정 (맑은 고딕체)
plt.rcParams['axes.unicode_minus'] = False    # 마이너스 깨짐 방지
warnings.filterwarnings('ignore')             # 경고 무시


## Constants ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
# Date Strings
YESTERDAY             = datetime.strftime(datetime.now(timezone('Asia/Seoul')) - timedelta(1)  , "%Y%m%d") # Yesterday (Format:"YYYYMMDD")
PREVIOUS_BUSINESS_DAY = datetime.strftime(datetime.now(timezone('Asia/Seoul')) - timedelta(3)  , "%Y%m%d") if datetime.now(timezone('Asia/Seoul')).weekday() == 0 else YESTERDAY # Previous Business Day (Format:"YYYYMMDD")
TODAY                 = datetime.strftime(datetime.now(timezone('Asia/Seoul'))                 , "%Y%m%d") # Yesterday (Format:"YYYYMMDD")
TOMORROW              = datetime.strftime(datetime.now(timezone('Asia/Seoul')) + timedelta(1)  , "%Y%m%d") # Yesterday (Format:"YYYYMMDD")
LAST_YEAR             = datetime.strftime(datetime.now(timezone('Asia/Seoul')) - timedelta(365), "%Y")     # Last year (Format:"YYYY")
CURRENT_YEAR          = datetime.strftime(datetime.now(timezone('Asia/Seoul'))                 , "%Y")     # Current year (Format:"YYYY")

# Tickers of World Indexes
WORLD_INDEX_TICKERS = [ {'ticker':'^GSPC',     'nation':'US',          'name':'S&P 500'},
                        {'ticker':'^DJI',      'nation':'US',          'name':'Dow Jones Industrial Average'},
                        {'ticker':'^IXIC',     'nation':'US',          'name':'NASDAQ Composite'},
                        # {'ticker':'^NYA',      'nation':'US',          'name':'NYSE COMPOSITE (DJ)'},
                        # {'ticker':'^XAX',      'nation':'US',          'name':'NYSE AMEX COMPOSITE INDEX'},
                        # {'ticker':'^BUK100P',  'nation':'UK',          'name':'Cboe UK 100'},
                        # {'ticker':'^RUT',      'nation':'US',          'name':'Russell 2000'},
                        # {'ticker':'^VIX',      'nation':'US',          'name':'Vix'},
                        {'ticker':'\^FTSE',    'nation':'UK',          'name':'FTSE 100'},
                        {'ticker':'^GDAXI',    'nation':'Germany',     'name':'DAX PERFORMANCE-INDEX'},
                        {'ticker':'^FCHI',     'nation':'France',      'name':'CAC 40'},
                        # {'ticker':'^STOXX50E', 'nation':'Europe',      'name':'ESTX 50 PR.EUR'},
                        # {'ticker':'^N100',     'nation':'France',      'name':'Euronext 100 Index'},
                        # {'ticker':'^BFX',      'nation':'Belgium',     'name':'BEL 20'},
                        # {'ticker':'IMOEX.ME',  'nation':'Russia',      'name':'MOEX Russia Index'},
                        {'ticker':'^N225',     'nation':'Japan',       'name':'Nikkei 225'},
                        # {'ticker':'^HSI',      'nation':'Taiwan',      'name':'HANG SENG INDEX'},
                        # {'ticker':'000001.SS', 'nation':'China',       'name':'SSE Composite Index'},
                        {'ticker':'399001.SZ', 'nation':'China',       'name':'Shenzhen Index'},
                        # {'ticker':'\^STI',     'nation':'Singapore',   'name':'STI Index'},
                        # {'ticker':'^AXJO',     'nation':'Australia',   'name':'S&P/ASX 200'},
                        # {'ticker':'^AORD',     'nation':'Australia',   'name':'ALL ORDINARIES'},
                        # {'ticker':'^BSESN',    'nation':'India',       'name':'S&P BSE SENSEX'},
                        # {'ticker':'^JKSE',     'nation':'Indonesia',   'name':'Jakarta Composite Index'},
                        # {'ticker':'\^KLSE',    'nation':'Malaysia',    'name':'FTSE Bursa Malaysia KLCI'},
                        # {'ticker':'^NZ50',     'nation':'New Zealand', 'name':'S&P/NZX 50 INDEX GROSS'},
                        {'ticker':'^KS11',     'nation':'Korea',       'name':'KOSPI Composite Index'},
                        # {'ticker':'^TWII',     'nation':'Taiwan',      'name':'TSEC weighted index'},
                        # {'ticker':'^GSPTSE',   'nation':'Canada',      'name':'S&P/TSX Composite index'},
                        # {'ticker':'^BVSP',     'nation':'Brazil',      'name':'IBOVESPA'},
                        # {'ticker':'^MXX',      'nation':'Mexico',      'name':'IPC MEXICO'},
                        # {'ticker':'^IPSA',     'nation':'Chile',       'name':'S&P/CLX IPSA'},
                        # {'ticker':'^MERV',     'nation':'Argentina',   'name':'MERVAL'},
                        # {'ticker':'^TA125.TA', 'nation':'Israel',      'name':'TA-125'},
                        # {'ticker':'^CASE30',   'nation':'Egypt',       'name':'EGX 30 Price Return Index'},
                        # {'ticker':'^JN0U.JO',  'nation':'Republic of South Africa', 'name':'Top 40 USD Net TRI Index'},
]



## General Functions ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
def get_world_index(ticker:str, startDt:str="20000101", endDt:str=YESTERDAY):
    """
    세계 주요 주가 지수의 일별 OHCLV(Open, High, Close, Low, Volume) 데이터를 담은 DataFrame을 반환한다.
    
    [Parameters]
    ticker  (str) : 조회할 지수의 Ticker
    startDt (str) : 조회할 데이터의 시작 일자 (YYYYMMDD) (Default: "20000101")
    endDt   (str) : 조회할 데이터의 종료 일자 (YYYYMMDD) (Default: 전일)
    
    [Returns]
    pandas.core.frame.DataFrame : 세계 주요 주가 지수의 일별 OHCLV 데이터를 담은 DataFrame
    """
    
    startDt_datetime = datetime.strptime(startDt, '%Y%m%d')
    endDt_datetime   = datetime.strptime(endDt,   '%Y%m%d')
    
    try:
        return pdr.DataReader(ticker, 'yahoo', startDt_datetime, endDt_datetime)
    except:
        print(f"Fail: Invalid index name {ticker}")
        
def get_normalization(df_ts):
    """
    시계열 데이터를 MinMaxScaler로 정규화한 결과를 반환한다.

    [Parameters]
    df_ts (pandas.core.frame.DataFrame) : 정규화할 시계열 데이터가 담긴 DataFrame
    
    [Returns]
    pandas.core.frame.DataFrame : 정규화된 시계열 데이터가 담긴 DataFrame
    """
    
    scaler = MinMaxScaler()
    scaler.fit(df_ts)
    return scaler.transform(df_ts)

def get_RSI(df_stock):
    """
    종목에 대한 DataFrame 형식의 OHLCV를 받아와 14일, 30일, 50일, 200일 RSI를 반환한다.

    [Parameters]
    df_stock (pandas.core.frame.DataFrame) : RSI를 계산할 데이터가 담긴 DataFrame
    
    [Returns]
    pandas.core.frame.DataFrame : RSI가 계산된 DataFrame
    """

    try:
        for n in [14, 30, 50, 200]:
            df_stock['RSI' + str(n)] = talib.RSI(df_stock['Close'].values, timeperiod = n)     
    except:
        df_stock = df_stock.astype('float64')
        for n in [14, 30, 50, 200]:
            df_stock['RSI' + str(n)] = talib.RSI(df_stock['Close'].values, timeperiod = n)     

    return df_stock

def get_RSI_OHLCV(df_stock):
    """
    종목에 대한 DataFrame 형식의 OHLCV를 받아와 14일, 30일, 50일, 200일 RSI를 반환한다.

    [Parameters]
    df_stock (pandas.core.frame.DataFrame) : RSI를 계산할 데이터가 담긴 DataFrame
    
    [Returns]
    pandas.core.frame.DataFrame : RSI가 계산된 DataFrame
    """
    
    try:
        for n in [14, 30, 50, 200]:
            df_stock['RSI' + str(n)] = talib.RSI(df_stock['종가'].values, timeperiod = n)     
    except:
        df_stock = df_stock.astype('float64')
        for n in [14, 30, 50, 200]:
            df_stock['RSI' + str(n)] = talib.RSI(df_stock['종가'].values, timeperiod = n)     

    return df_stock

def preprocessing_rsi_backtesting(df_stock):
    """
    기존 전처리한 데이터로 RSI 수익률을 산출하여 반환한다.
    
    [Parameters]
    df_stock (pandas.core.frame.DataFrame) : RSI 수익률을 계산할 데이터가 담긴 DataFrame
    
    [Returns]
    pandas.core.frame.DataFrame : RSI 수익률이 계산된 DataFrame
    """

    df_stock = df_stock.set_index('date')
    
    # 매매신호 컬럼 생성
    df_stock.loc[df_stock['RSI14']<30,'매매신호'] = True # 1이면 매수신호
    df_stock.loc[df_stock['RSI14']>70,'매매신호'] = False # 0이면 매도신호

    # 일간수익률 컬럼 생성
    df_stock['일간수익률'] = df_stock['Close'].pct_change() + 1
    
    # 보유여부 컬럼 생성
    df_stock.loc[df_stock['매매신호'].shift(1) == True, '보유여부']=True # 1이면 현재 보유
    df_stock.loc[df_stock['매매신호'].shift(1) == False, '보유여부']=False # 0이면 현재 보유x
    df_stock['보유여부'].ffill(inplace=True)
    df_stock['보유여부'].fillna(False,inplace=True)
    
    # 보유수익률 컬럼 생성 - 보유하지 않은 날에는 원금을 그대로 유지하므로 해당 거래일의 수익률은 1로 지정.
    df_stock['보유수익률'] = df_stock.loc[df_stock['보유여부']==True,'일간수익률']
    df_stock['보유수익률'].fillna(1,inplace=True)
    
    # RSI 누적수익률 컬럼 생성
    df_stock['RSI수익률'] = df_stock['보유수익률'].cumprod()
    df_stock['단순보유수익률'] = df_stock['Close'] / df_stock.iloc[0,0]

    return df_stock

def preprocessing_rsi_backtesting_OHLCV(df_stock): 
    """
    KRX API를 통해 새로 불러온 데이터로 RSI 수익률을 반환한다.
    
    [Parameters]
    df_stock (pandas.core.frame.DataFrame) : RSI 수익률을 계산할 데이터가 담긴 DataFrame
    
    [Returns]
    pandas.core.frame.DataFrame : RSI 수익률이 계산된 DataFrame
    """

    # 매매신호 컬럼 생성
    df_stock.loc[df_stock['RSI14']<30,'매매신호'] = True # 1이면 매수신호
    df_stock.loc[df_stock['RSI14']>70,'매매신호'] = False # 0이면 매도신호

    # 일간수익률 컬럼 생성
    df_stock['일간수익률'] = df_stock['종가'].pct_change() + 1
    
    # 보유여부 컬럼 생성
    df_stock.loc[df_stock['매매신호'].shift(1) == True, '보유여부']=True # 1이면 현재 보유
    df_stock.loc[df_stock['매매신호'].shift(1) == False, '보유여부']=False # 0이면 현재 보유x
    df_stock['보유여부'].ffill(inplace=True)
    df_stock['보유여부'].fillna(False,inplace=True)
    
    # 보유수익률 컬럼 생성 - 보유하지 않은 날에는 원금을 그대로 유지하므로 해당 거래일의 수익률은 1로 지정.
    df_stock['보유수익률'] = df_stock.loc[df_stock['보유여부']==True,'일간수익률']
    df_stock['보유수익률'].fillna(1,inplace=True)
    
    # RSI 누적수익률 컬럼 생성
    df_stock['RSI수익률'] = df_stock['보유수익률'].cumprod()
    df_stock['단순보유수익률'] = df_stock['종가'] / df_stock.iloc[0,0]

    return df_stock

def scale_rsi_backtesting(scale):
    """
    규모별(코스피,대형주,중형주,소형주) 인덱스에 해당하는 모든 종목들의 rsi 백테스팅 수익률을 반환한다.
    
    [Parameters]
    scale (str) : RSI 수익률을 계산할 인덱스 (코스피:'kospi' | 대형주:'large' | 중형주:'medium' | 소형주:'small')
    
    [Returns]
    pandas.core.series.Series : 규모별 각 종목의 RSI 수익률이 계산된 Series
    """

    yeild=[]
    # s 는 각 인덱스에 해당하는 종목
    for s in scale:
        df = stock.get_market_ohlcv('20030301','20220901',s)       # 2000년 3월1일부터 2022년 9월 1일까지의 정보들
        df = get_RSI_OHLCV(df)                              # RSI 컬럼을 생성하는 함수 호출
        df = preprocessing_rsi_backtesting_OHLCV(df)        # RSI 수익률을 구하는 함수 호출
        yeild.append(df.iloc[-1].loc['RSI수익률'])
    s = pd.Series(yeild)
    return s

def df_manipulate_kospi(df_stock):
    """
    시가총액별 KOSPI DataFrame을 받아와 60일, 120일, 360일 MA 및 MA와 지수 사이의 이격도를 반환한다.

    [Parameters]
    df_stock (pandas.core.frame.DataFrame) : MA를 계산할 데이터가 담긴 DataFrame
    
    [Returns]
    pandas.core.frame.DataFrame : MA가 계산된 DataFrame
    """
    
    df_stock = df_stock.set_index('date')
    df_stock = df_stock[['Open', 'High', 'Low', 'Close', 'Volume']]
    
    df_stock["MA60"] = df_stock['Close'].rolling(window=60).mean() # MA60값 계산
    df_stock["MA120"] = df_stock['Close'].rolling(window=120).mean()
    df_stock["MA360"] = df_stock['Close'].rolling(window=360).mean()

    # MA_sep은 MA선과 그 날의 종가 사이의 이격도를 나타내는 값
    # 1 이상일 때, MA선보다 크게 하락하는 장을 의미하고, 1 이하일 때 MA선보다 크게 상승하는 장을 의미함.
    
    df_stock["MA60_sep"] = (df_stock["MA60"] / df_stock['Close']) # 지수화된 MA60 계산
    df_stock["MA120_sep"] = (df_stock["MA120"] / df_stock['Close'])
    df_stock["MA360_sep"] = (df_stock["MA360"] / df_stock['Close'])

    return df_stock

def set_date(df_stock, buy, sell):
    """
    문자열 형식의 구매 날짜와 판매 날짜를 입력 받아 해당 기간의 데이터만을 가진 DataFrame을 반환한다.
    
    [Parameters]
    df_stock (pandas.core.frame.DataFrame) : 조회하고자 하는 데이터가 담긴 DataFrame
    buy      (str)                         : 매수일
    sell     (str)                         : 매도일
    
    [Returns]
    pandas.core.frame.DataFrame : 해당 기간의 데이터만을 가진 DataFrame
    """

    buy = datetime.strptime(buy,'%Y-%m-%d')
    sell = datetime.strptime(sell,'%Y-%m-%d')
    cond1 = df_stock['date'] >= buy
    cond2 = df_stock['date'] <= sell

    return df_stock.loc[cond1 & cond2]

def get_beta(df_stock, df_index, start, end):
    """
    특정 종목과 시장지수를 DataFrame형식으로 받아 시장지수 대비 특정 종목의 베타값을 반환한다.
    
    [Parameters]
    df_stock (pandas.core.frame.DataFrame) : 조회하고자 하는 특정종목의 DataFrame
    df_index (pandas.core.frame.DataFrame) : 조회하고자 하는 시장지수 인덱스 DataFrame
    start    (str)                         : 매수일
    end      (str)                         : 매도일
    
    [Returns]
    float64 : beta 값
    """

    df_stock = set_date(df_stock, start, end)
    df_index = set_date(df_index, start, end)
    
    data = pd.concat([df_stock['close'].reset_index(drop=True), df_index['close'].reset_index(drop=True)], axis=1)
    daily_return = np.log(data / data.shift(1))

    cov = daily_return.cov() * 250
    cov_market = cov.iloc[0,1]
    market_var = daily_return.iloc[:,1].var() * 250
    beta = cov_market / market_var

    return round(beta, 3)

def get_corr(df_stock1, df_stock2, start, end):
    """
    특정 두 종목을 DataFrame형식으로 받아 두 종목 간의 상관계수를 반환한다.
    
    [Parameters]
    df_stock1 (pandas.core.frame.DataFrame) : 조회하고자 하는 특정종목의 DataFrame
    df_stock2 (pandas.core.frame.DataFrame) : 조회하고자 하는 특정종목의 DataFrame
    start    (str)                          : 매수일
    end      (str)                          : 매도일
    
    [Returns]
    float64 : 상관계수 값
    """

    df_stock1 = set_date(df_stock1, start, end)
    df_stock2 = set_date(df_stock2, start, end)
    
    data = pd.concat([df_stock1['close'].reset_index(drop=True), df_stock2['close'].reset_index(drop=True)], axis=1)
    daily_return = np.log(data / data.shift(1))
    corr = daily_return.corr().iloc[1,0]

    return round(corr,3)

def get_mdd(df_stock, start, end):
    """
    특정 종목을 DataFrame형식으로 받아 mdd를 반환한다.
    
    [Parameters]
    df_stock (pandas.core.frame.DataFrame) : mdd를 구하고자 하는 특정종목의 DataFrame
    start    (str)                         : 시작 날짜
    end      (str)                         : 종료 날짜
    
    [Returns]
    float64 : mdd 값
    """
    
    x_mdd = set_date(df_stock,start,end)
    저가 = x_mdd['low'].min() #최저점
    index = x_mdd[x_mdd['low'] == x_mdd['low'].min()].index

    # 저가 기준 전고점 구하기
    전고점기간 = x_mdd.loc[:index[0], :]
    전고점 = 전고점기간['high'].max()

    # MDD 계산
    mdd = round((저가 - 전고점) / 전고점, 4) * 100
    
    return mdd

def get_mdd_5(df_stock):
    """
    특정 종목을 DataFrame형식으로 받아 2017-09-01 ~ 2022-09-01 5년간의 mdd를 반환한다.
    
    [Parameters]
    df_stock (pandas.core.frame.DataFrame) : mdd를 구하고자 하는 특정종목의 DataFrame
    
    [Returns]
    float64 : 5년 간의 mdd 값
    """

    return get_mdd(df_stock, '2017-09-01', '2022-09-01')

def get_mdd_10(df_stock):
    """
    특정 종목을 DataFrame형식으로 받아 2012-09-01 ~ 2022-09-01 10년간의 mdd를 반환한다.
    
    [Parameters]
    df_stock (pandas.core.frame.DataFrame) : mdd를 구하고자 하는 특정종목의 DataFrame
    
    [Returns]
    float64 : 10년 간의 mdd 값
    """

    return get_mdd(df_stock, '2012-09-01', '2022-09-01') 

def get_mdd_200(df_stock):
    """
    특정 종목을 DataFrame형식으로 받아 2018-10-22 ~ 2022-09-01 200주 간의 mdd를 반환한다.
    
    [Parameters]
    df_stock (pandas.core.frame.DataFrame) : mdd를 구하고자 하는 특정종목의 DataFrame
    
    [Returns]
    float64 : 2018-10-22 ~ 2022-09-01 200주 간의 mdd 값
    """

    return get_mdd(df_stock, '2018-10-22', '2022-09-01') 
















tmp = []
for ticker in stock.get_index_ticker_list():
    tmp.append([ticker, stock.get_index_ticker_name(ticker)])

# 데이터 프레임 생성
scale_2022 = pd.DataFrame(tmp).iloc[:4,:]

# 열 명 변경
scale_2022.columns = ['ticker','name2022']

# DataFrame 구축

# DF 생성
df_scale_day = pd.DataFrame([])          # 일단위 데이터
df_scale_month = pd.DataFrame([])        # 월단위 데이터
df_scale_year = pd.DataFrame([])         # 년단위 데이터

# DF 내용 쓰기
for j in range(scale_2022.iloc[:,0].count()):
    # ticker 값을 i에 저장
    i = scale_2022.iloc[j,0]
    
    # temp DF에 2000.01.01~2022.09.01 규모별 ohlcv와 fundamental 쓰기(merge 이용하여 옆으로 붙임.)
    temp_day = pd.merge(stock.get_index_ohlcv("20030301", "20220901", str(i)).reset_index(), 
                     stock.get_index_fundamental("20030301", "20220901", str(i)).reset_index(), 
                     how='outer')
    temp_month = stock.get_index_ohlcv("20030301", "20220901", str(i),'m').reset_index()
                
    temp_year = stock.get_index_ohlcv("20030301", "20211231", str(i),'y').reset_index()
    
    # 해당 섹터 ticker, 섹터 명칭 쓰기
    temp_day['scale_name'] = stock.get_index_ticker_name(i)
    temp_day['scale_ticker'] = i
    
    temp_month['scale_name'] = stock.get_index_ticker_name(i)
    temp_month['scale_ticker'] = i
    
    temp_year['scale_name'] = stock.get_index_ticker_name(i)
    temp_year['scale_ticker'] = i
    
    # 임시 DF를 결과물 DF에 합치기
    df_scale_day = pd.concat([df_scale_day,temp_day])
    df_scale_month = pd.concat([df_scale_month,temp_month])
    df_scale_year = pd.concat([df_scale_year,temp_year])

# 칼럼명 영어로 변환
df_scale_day.columns = ['date','Open','High','Low','Close','Volume','Volume($)',
                     'Market_Cap','Change','PER','FWDPER','PBR','dividend_Yield','scale_name','scale_ticker']
df_scale_month.columns = ['date','Open','High','Low','Close','Volume','Volume($)','scale_name','scale_ticker']
df_scale_year.columns = ['date','Open','High','Low','Close','Volume','Volume($)','scale_name','scale_ticker']

# 일별 데이터를 규모별로 분리
df_scale_day_kospi = df_scale_day[df_scale_day['scale_name']=='코스피'] 
df_scale_day_small = df_scale_day[df_scale_day['scale_name']=='코스피 소형주']
df_scale_day_medium = df_scale_day[df_scale_day['scale_name']=='코스피 중형주']
df_scale_day_large = df_scale_day[df_scale_day['scale_name']=='코스피 대형주']

# 월별 데이터를 규모별로 분리
df_scale_month_kospi = df_scale_month[df_scale_month['scale_name']=='코스피']
df_scale_month_small = df_scale_month[df_scale_month['scale_name']=='코스피 소형주']
df_scale_month_medium = df_scale_month[df_scale_month['scale_name']=='코스피 중형주']
df_scale_month_large = df_scale_month[df_scale_month['scale_name']=='코스피 대형주']

# 연도별 데이터를 규모별로 분리
df_scale_year_kospi = df_scale_year[df_scale_year['scale_name']=='코스피']
df_scale_year_small = df_scale_year[df_scale_year['scale_name']=='코스피 소형주']
df_scale_year_medium = df_scale_year[df_scale_year['scale_name']=='코스피 중형주']
df_scale_year_large = df_scale_year[df_scale_year['scale_name']=='코스피 대형주']

# df_scale_year_kospi_index = (df_scale_year_kospi['Close'] / df_scale_year_kospi['Close'][0]) * 100
# df_scale_year_small_index = (df_scale_year_small['Close'] / df_scale_year_small['Close'][0]) * 100
# df_scale_year_medium_index = (df_scale_year_medium['Close'] / df_scale_year_medium['Close'][0]) * 100
# df_scale_year_large_index = (df_scale_year_large['Close'] / df_scale_year_large['Close'][0]) * 100

# df_scale_month_kospi_index = (df_scale_month_kospi['Close'] / df_scale_month_kospi['Close'][0]) * 100
# df_scale_month_small_index = (df_scale_month_small['Close'] / df_scale_month_small['Close'][0]) * 100
# df_scale_month_medium_index = (df_scale_month_medium['Close'] / df_scale_month_medium['Close'][0]) * 100
# df_scale_month_large_index = (df_scale_month_large['Close'] / df_scale_month_large['Close'][0]) * 100

# df_scale_day_kospi_index = (df_scale_day_kospi['Close'] / df_scale_day_kospi['Close'][0]) * 100
# df_scale_day_small_index = (df_scale_day_small['Close'] / df_scale_day_small['Close'][0]) * 100
# df_scale_day_medium_index = (df_scale_day_medium['Close'] / df_scale_day_medium['Close'][0]) * 100
# df_scale_day_large_index = (df_scale_day_large['Close'] / df_scale_day_large['Close'][0]) * 100






# MA를 위한 시각화 및 데이터프레임 조정

df_scale_day_kospi_MA = df_manipulate_kospi(df_scale_day_kospi)
df_scale_day_small_MA = df_manipulate_kospi(df_scale_day_small)
df_scale_day_medium_MA = df_manipulate_kospi(df_scale_day_medium)
df_scale_day_large_MA = df_manipulate_kospi(df_scale_day_large)
df_scale_month_kospi_MA = df_manipulate_kospi(df_scale_month_kospi)
df_scale_month_small_MA = df_manipulate_kospi(df_scale_month_small)
df_scale_month_medium_MA = df_manipulate_kospi(df_scale_month_medium)
df_scale_month_large_MA = df_manipulate_kospi(df_scale_month_large)



# Visualization: 코스피 일별 60일선과 코스피 소형주 / 중형주 / 대형주 비교

fig = plt.figure(figsize=(14, 6))
ax = fig.add_subplot(1, 1, 1)

ax.plot(df_scale_day_small_MA['Close'], label="소형주")
ax.plot(df_scale_day_medium_MA['Close'], label="중형주")
ax.plot(df_scale_day_large_MA['Close'], label="대형주")
ax.plot(df_scale_day_kospi_MA["Close"], label='코스피')
ax.plot(df_scale_day_kospi_MA["MA60"], label='코스피 ma60')

my_date_start = date(2020, 1, 1)
my_date_end = date(2022, 9, 30)

plt.xlim([my_date_start, my_date_end])
plt.ylim([1000, 3800])
ax.legend(loc='best')
plt.title('코스피 일별 60일선과 코스피 소형주 / 중형주 / 대형주 비교')
plt.show()








