## 미국 연간 GDP 증가량
![image](https://github.com/sejin1129/summer_Intern/assets/113009722/d0c3b9cd-bffc-4995-aa2c-c2e7fdae6b12)
- GDP증가량을 바탕으로 자산 사이클의 등락 구간을 분석할 예정
- 이후 아래 5가지 증권데이터 지표를 미국 경제성장률(GDP)의 사이클에 대응하여 분석하고자 함

## 결측치 & 중복치 해결 후 증권데이터 그래프
![image](https://github.com/sejin1129/summer_Intern/assets/113009722/c0cb6ce8-4d0d-4428-b58f-04f17f2db5a1)
- 이상치 제거는 할 필요 없다고 판단
- csv 파일을 가져오기전 미리 정제해놓은 상태로 이상치로는 특이치만 있을 것임. 특이치는 자산데이터에 필요한 데이터
- 왼쪽 그래프는 지표마다 측정단위가 달라 지표 간 관계가 잘 보이지 않음
- 기존 자산 데이터와 min-max 정규화 기법(0~1로 값 정규화)을 적용한 자산 데이터를 그래프로 표현
- 2년만기 미국채 데이터를 10년만기 미국채로 변경함 (채권)
- 2023-07-01 까지 데이터를 찾아 csv 파일을 업데이트함
- 시작구간과 끝구간을 정해서 분석 가능(장기 & 단기)
 
## 결측치 & 중복치 해결 후 증권데이터 상관관계 분석
![image](https://github.com/sejin1129/summer_Intern/assets/113009722/a29ced87-f1eb-49bf-9bff-76d33325b5cb)
- 정규화 된 5가지 증권데이터에 대해 피어슨 상관분석 후 표로 시각화 함
- 분석 구간을 어떻게 정하는지에 따라 값이 상이하이 때문에 다양한 각도로 분석이 필요함

## 수익률 분석
![image](https://github.com/sejin1129/summer_Intern/assets/113009722/7b77f84a-cd7e-4793-a31c-5faaa25ab6dc)
- 수익률을 이용해 어느구간에 무엇을 가지고다면 수익이 높았는지를 알아보려고 함
- 이전 달과 현재 달의 차이를 이용해 수익률이 얼마만큼 증가한지를 나타내는 그래프
- 달마다 수익률이 + 또는 - 로 매번 바뀌기 때문에 장기적인 그래프로는 분석이 까다로움
- 분석 기간이 길어질 수록 진동수가 많아져서 위와 같이 상관분석이 제대로 이루어지지 않음

![image](https://github.com/sejin1129/summer_Intern/assets/113009722/d5cc3bb7-a0a4-49dd-a689-bf169a79ab8d)
- 분석 기간을 짧게해서 단기간 수익률 추세정도는 확인 가능할지도..


