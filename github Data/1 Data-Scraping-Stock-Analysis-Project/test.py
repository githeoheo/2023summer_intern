i=4
try:
    while (i-2)/i <1.0:
        print(i)
        i=i-4
    print(i)
    print("Hi")
except:
    print("Hui")


import numpy as np
import pandas as pd



df=pd.read_csv("C:\\Users\\gopib\\Downloads\\covid.csv")
df['Still_Infected']= df['Confirmed'] - df['Deaths'] - df['Recovered']
df=df.groupby('Country/Region').sum(['Still_Infected'])
df1=df.sort_values(by='Still_Infected',ascending=False)
print(df1)

df=pd.read_csv("C:\\Users\\gopib\\Downloads\\games1.csv")
df=df.groupby('platform').sum('Total_sales')
df=df[df['Total_sales']>100]
df.sort_values(by='Total_sales',ascending=False,inplace=True)
print(df)


