SELECT
    DATE_TRUNC('year', d.date)::date AS ds,
    SUM(d.total_passengers) AS y
FROM public.daily_passenger_stats AS d
WHERE d.route_id = '91'
  AND TRIM(d.operator) = 'XKSG'
GROUP BY DATE_TRUNC('year', d.date)
ORDER BY ds ASC;
