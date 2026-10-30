SELECT
    d.date AS ds,
    SUM(d.total_passengers) AS y
FROM public.daily_passenger_stats AS d
WHERE d.route_id = '65'
  AND TRIM(d.operator) = 'Cty-BaoYen'
GROUP BY d.date
ORDER BY ds ASC;
