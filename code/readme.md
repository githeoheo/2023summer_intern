## CODE
---
> ### 데이터 수집
- csv 파일 불러오기
```python
stock_df = pd.read_csv('C:/Users/user/Desktop/학기별 문서/현장실습/데이터자료/나스닥(1985~2023)_yfinance.csv')
```

- 생성한 데이터프레임 확인하기
```python
stock_df = stock_df.set_index('Date')
```

- 불러온 데이터의 컬럼 이름 변경하기
```python
house_df = house_df.rename(columns={'SPCS10RSA':'House_Price', "DATE":"Date"})
```

- 불러온 데이터의 구간 설정하기
```python
start = "2019-11-01" # 최소 1950-01-01
end = "2020-12-01" # 최대 2023-07-01
stock_df = stock_df[stock_df['Date'].between(start, end)]
```

- 날짜를 datatime 형식으로 전환하기
```python
stock_df.loc[:,'Date'] = pd.to_datetime(stock_df.Date)
```

- 날짜를 데이터프레임 index로 전환하기
```python
stock_df = stock_df.set_index('Date')
```   
> ### 데이터 전처리(결측치/중복치)
- 필요없는 행열 삭제하기(axis = 0 : 행 / axis = 1 : 열 / 생략 시 0 디폴트)
```python
stock_df = stock_df.drop(['Open', 'High', 'Low', 'Adj Close', 'Volume'], axis = 1) 
```

- 결측치가 있는 행 or 열 제거하는 함수(axis = 0 : 행 / axis = 1 : 열 / 생략 시 0 디폴트)
```python
stock_df = stock_df.dropna(axis=0)
```

- 중복데이터 확인하기
```python
print(fund_df[fund_df.duplicated()])
```

- 중복데이터 삭제하기
```python
df.drop_duplicates(['컬럼'], keep = 'first')
```
---
## 분석그래프(원본, min-max정규화, 상관분석)
![image](https://github.com/githeoheo/2023summer_intern/assets/113009722/40f62e69-5ab0-4c56-a9cc-debbea264041)
- 미국의 경제사이클 분석을 위한 지표로 GDP와 실업률을 선택했다
- 상관관계 분석기법(pearsonr, kendalltau, spearmanr) 사용 시 GDP, 실업률은 관계가 없음을 알아냈다
```python
    stats.pearsonr(X,Y) -> PearsonRResult(statistic=-0.06032551323899124, pvalue=0.09854796999125372)
    stats.kendalltau(X,Y) -> SignificanceResult(statistic=-0.009059600526627815, pvalue=0.7129098440414525)
    stats.spearmanr(X,Y) -> SignificanceResult(statistic=-0.009833604183356904, pvalue=0.7878977953390143)
```
![image](https://github.com/githeoheo/2023summer_intern/assets/113009722/365fdcd6-6bea-497c-bc4d-ddba1c8d112a)
- 2019.11 ~ 2020.12 까지의 5가지 자산 지표 그래프(원본&정규화)
![image](https://github.com/githeoheo/2023summer_intern/assets/113009722/21ad809f-e4a3-4366-a991-17576f541867)
- 상관관계 분석 결과 해당 기간에 주식과 부동산은 양의 관계, 채권과 금리는 음의 관계가 두드러짐을 확인가능하다
![image](https://github.com/githeoheo/2023summer_intern/assets/113009722/5e0351c0-8c70-4ea8-8971-a879a3cb3cd7)
- 기간 내 5가지 지표 그래프의 더 알아보기 위해 수익률을 구해 정규화하여 그래프를 그려보았다
- 상관분석이 제대로 이루어지는지 확인을 위해 SnP500지수(나스닥과 흐름이 거의 같은 지수)를 추가하였다
![image](https://github.com/githeoheo/2023summer_intern/assets/113009722/2e6e2df4-c916-4e5b-9a8a-dac30751c7a2)
- 상관관계 분석 결과 나스닥과 SnP 가 양의 관계인 것으로 보아 분석은 잘되는 것으로 확인된다
- 하지만 5가지 지표의 관계를 알아내기에 좋은 방법으로 느껴지지 않는다
