select
	date,
	0.1 + avg(normalized_value) over(order by date
							  rows between 16 preceding and 7 following
							  ) as trend,
	count(normalized_value) over(order by date
								rows between 16 preceding and 7 following
								) as records_count
from
(
	select
		date,
		(2 * (diff - min_value) / (max_value - min_value)) -1 as normalized_value
	from
		trends,
		(
			select 
				min(diff) as min_value
				,max(diff) as max_value
			from trends
		) mn
) as normalized
;