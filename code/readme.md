## CODE
---
1. csv 파일 불러오기
```python
stock_df = pd.read_csv('C:/Users/user/Desktop/학기별 문서/현장실습/데이터자료/나스닥(1985~2023)_yfinance.csv')
```

2. 불러온 데이터의 컬럼 이름 변경하기
```python
house_df = house_df.rename(columns={'SPCS10RSA':'House_Price', "DATE":"Date"})
```

3. 불러온 데이터의 구간 설정하기
```python
start = "2019-11-01" # 최소 1950-01-01
end = "2020-12-01" # 최대 2023-07-01
stock_df = stock_df[stock_df['Date'].between(start, end)]
```

4. 날짜를 datatime 형식으로 전환하기
```python
stock_df.loc[:,'Date'] = pd.to_datetime(stock_df.Date)
```

5. 날짜를 데이터프레임 index로 전환하기
```python
stock_df = stock_df.set_index('Date')
```

6. 

