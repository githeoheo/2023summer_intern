-- cartesian product 확인
WITH Sample AS (
    SELECT 'A' AS COL1, 1 AS COL2
    UNION ALL SELECT 'B', 2
	UNION ALL SELECT 'C', 3
)
SELECT
	*
FROM Sample
CROSS JOIN
	(SELECT 1 as COL3 UNION ALL SELECT 2) Category
;

WITH Sample AS (
    SELECT 'A' AS COL1, 1 AS COL2
    UNION ALL SELECT 'B', 2
	UNION ALL SELECT 'C', 3
)
SELECT
    CASE COL3 
		WHEN 1 THEN COL1
		WHEN 2 THEN '합계'
	END AS COL1,
	SUM(COL2) AS COL2
FROM Sample
CROSS JOIN
	(SELECT 1 as COL3 UNION ALL SELECT 2) Category
GROUP BY
	CASE COL3
		WHEN 1 THEN COL1
		WHEN 2 THEN '합계'
	END
ORDER BY COL1
;


with Sample as
	(
    select 'A' COL1, 1 COL2
    UNION ALL SELECT 'B', 2
    )
select  coalesce(col1, '합계') as col1
	,sum(col2) as col2
from Sample 
group by col1 with Rollup;

-- 옛날 DB를 가정해서 

-- left join
with Sample as
(
    select 'A' COL1, 1 COL2
    UNION ALL SELECT 'B', 2
)
select 
	coalesce(b.col1, '합계') as col1, 
    sum(a.col2) as col2
from Sample a 
left join Sample b on a.col1 = b.col2
group by 1
;

with Sample as
(
    select 'A' COL1, 1 COL2
    UNION ALL SELECT 'B', 2
)
select 
	*
from Sample a 
left join Sample b on a.col1 = b.col2
;