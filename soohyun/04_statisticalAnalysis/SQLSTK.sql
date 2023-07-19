# 4. 종목 테이블 이해하기

## 1) STOCK 테이블의 정체
desc stock;

desc history_dt;
-- dt는 주식이 거래된 날짜 값을 저장하는 컬럼 date 자료형으로 정의
SELECT  
-- 	T1.*
 	T1.STK_CD ,T1.DT ,T1.O_PRC ,T1.H_PRC ,T1.L_PRC ,T1.C_PRC ,T1.VOL ,T1.CHG_RT
FROM    HISTORY_DT T1
WHERE   T1.STK_CD = '005930' # 삼성전자의 종목코드
AND     T1.DT = STR_TO_DATE('20190108','%Y%m%d');

select
  '일별주가' 테이블명,
  t1.stk_cd, 
  t1.dt, 
  t1.o_prc, 
  t1.h_prc, 
  t1.l_prc, 
  t1.c_prc, 
  t1.vol, 
  t1.chg_rt
from
  history_dt t1
where
  t1.stk_cd = '005930'
  and t1.dt >= STR_TO_DATE('20190301', '%Y%m%d')
  and t1.dt < str_to_date('20190401', '%Y%m%d')
order by t1.dt asc;


SELECT T1.DT
       ,DATE_ADD(T1.DT, interval +2 day) AF2_DT
       ,T1.DT + 2 AF2_DT_ADD
FROM   HISTORY_DT T1
WHERE  T1.STK_CD = '005930'
AND    T1.DT = STR_TO_DATE('20190131','%Y%m%d');


select ex_cd
-- stk_cd, stk_nm, sec_nm, ex_cd
from stock
where stk_nm like '동일%'
group by ex_cd;

-- 4.08 GROUP BY, 섹터별로 데이터 건수를 구하는 SQL
select
  T1.sec_nm,
  count(*) CNT
from stock T1
where T1.stk_nm like '동일%'
group by T1.sec_nm
order by T1.sec_nm
;


-- 4.09 종목명이 삼성 or 현대로 시작하는 종목 group
select 
	substr(T1.stk_nm, 1, 2) STL_SUB_NAME, 
	count(*) CNT
from stock T1
where (T1.STK_NM like '삼성%' or T1.stk_nm like '현대%')
group by substr(T1.stk_nm, 1, 2)
order by substr(T1.stk_nm, 1, 2)
;

-- 일자를 연월로 변형한 후에 group by
select
	date_format(T1.dt, '%Y%m') YM, count(*)CNT
from history_dt T1
where T1.stk_cd = '005930'
group by date_format(T1.dt, '%Y%m')
order by YM asc
;

-- 여러 컬럼 group by
SELECT  T1.EX_CD ,T1.SEC_NM ,COUNT(*) CNT
FROM    STOCK T1
WHERE   T1.STK_NM LIKE '동일%'
GROUP BY T1.EX_CD ,T1.SEC_NM
ORDER BY T1.EX_CD ,T1.SEC_NM;


-- 5.07 기초코드
desc basecode_dv;

select * from basecode order by bas_cd_dv, srt_od;

select 
    T1.ex_cd, T2.bas_cd_nm EX_CD_NM
    ,T1.nat_cd, T3.bas_cd_nm NAT_CD_NM
    ,T1.stk_cd, T1.stk_nm
from 
    stock T1
    left outer join basecode T2
      on (T2.bas_cd_dv = 'EX_CD' and T2.bas_cd = T1.ex_cd)
    left outer join basecode T3
      on (T3.bas_cd_dv = 'NAT_CD' and T3.bas_cd = T1.nat_cd)
where T1.stk_nm in ('삼성전자', '서울반도체');


select
	T1.ex_cd,
    (select a.bas_cd_nm from basecode A where A.bas_cd_dv = 'ex_cd' and a.bas_cd = T1.ex_cd) EX_CD_NM,
    t1.nat_cd,
    (select a.bas_cd_nm from basecode A where a.bas_cd_dv = 'nat_cd' and a.bas_cd = t1.nat_cd) NAT_CD_NM,
    t1.stk_cd,
    t1.stk_nm
from stock T1
where T1.STK_NM in ('삼성전자', '서울반도체')
;



-- 5.09 등락률
select t1.stk_cd, t1.stk_nm, t2.dt, t2.c_prc, t2.chg_rt
from 
	stock t1
	inner join history_dt t2
		on (t2.stk_cd = t1.stk_cd)
where t1.stk_nm = '삼성전자'
and t2.dt = str_to_date('20190109', '%Y%m%d')
;

-- 등락률 계산
select t1.stk_cd, t1.stk_nm,
	round((t2.c_prc - t3.c_prc) / t3.c_prc * 100, 2) CHG_RT
from stock t1
	inner join history_dt t2
		on (t2.stk_cd = t1.stk_cd)
	inner join history_dt t3
		on (t3.stk_cd = t1.stk_cd)
where t1.stk_nm = '삼성전자'
and t2.dt = str_to_date('20190109', '%Y%m%d')
and t3.dt = str_to_date('20190108', '%Y%m%d')
;

-- 2020년 3월 19일에 가장 많이 빠진 종목을 차례대로 조회
select t1.stk_cd, t1.stk_nm,
	t2.dt, t2.c_prc,
    t3.dt, t3.c_prc,
    round((t2.c_prc - t3.c_prc) / t3.c_prc * 100, 2) CHG_RT
from stock t1
	inner join history_dt t2
		on t2.stk_cd = t1.stk_cd
	inner join history_dt t3
		on (t3.stk_cd = t1.stk_cd)
where 	t2.dt = str_to_date('20190319', '%Y%m%d')
and		t3.dt = str_to_date('20190102', '%Y%m%d')
order by chg_rt asc;

select t1.stk_cd, t1.stk_nm,
    t2.dt, t2.c_prc,
    t3.dt, t3.c_prc,
    round((t2.c_prc - t3.c_prc) / t3.c_prc * 100, 2) CHG_RT
from stock t1
    inner join history_dt t2
        on (t2.stk_cd = t1.stk_cd)
    inner join history_dt t3
        on t3.stk_cd = t1.stk_cd
where   t2.dt = str_to_date('20201230', '%Y%m%d')
and     t3.dt = str_to_date('20200319', '%Y%m%d')
order by CHG_RT desc;


-- 매도 시점으 history_dt(t_sell)를 한 번 더 조인해 매도한 일자의 종가를 가져와 수익률을 구할 수 있다.

select t1.stk_cd, t1.stk_nm,
    t_buy.dt BUY_DT, round(t_buy.c_prc, 1) BUY_PRC, round(t_buy.c_prc * 10, 1) BUY_AMT,
    t_sell.dt SELL_DT, round(t_sell.c_prc, 1) SELL_PRC, round(t_sell.c_prc * 10, 1) SELL_AMT,

    round((t_sell.c_prc - t_buy.c_prc) / t_buy.c_prc * 100, 2) 수익률,
	ROUND((T_SELL.C_PRC * 10) - (T_BUY.C_PRC * 10),1) 수익금
    -- 수익금, 이거 안되는 이유
--     SELL_AMT - BUY_AMT 수익금
from stock t1
    inner join history_dt t_buy
        on t_buy.stk_cd = t1.stk_cd
    inner join history_dt t_sell
        on t_sell.stk_cd = t1.stk_cd
where t1.stk_nm in ('삼성전자','카카오','LG화학')
and t_buy.dt = str_to_date('20190102', '%Y%m%d')
and t_sell.dt = str_to_date('20191227', '%Y%m%d')
order by 수익금 desc;